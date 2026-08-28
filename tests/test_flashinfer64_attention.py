import unittest

import torch
import torch.nn.functional as F

from dfsattn.flashinfer64_attention import (
    FlashInfer64Attention,
    _build_residual_topk,
    _ensure_flashinfer_vector_workspace,
    _select_hyvideo_core_tiles,
    compute_64_tile_scores,
    select_64_tiles_from_scores,
)


class FlashInfer64AttentionTest(unittest.TestCase):
    def test_top_p_one_selects_zero_probability_tiles(self) -> None:
        scores = torch.tensor([[[1.0, 0.0, 0.0]]])
        mask = select_64_tiles_from_scores(scores, top_p=1.0)
        self.assertTrue(bool(mask.all()))

    def test_top_p_one_core_only_matches_dense_with_tail_and_permutation(self) -> None:
        torch.manual_seed(0)
        sequence = 129
        q = torch.randn(1, 3, sequence, 16)
        k = torch.randn_like(q)
        v = torch.randn_like(q)
        video_len = 113
        video_perm = torch.randperm(video_len)

        backend = FlashInfer64Attention()
        actual = backend(
            q,
            k,
            v,
            video_perm=video_perm,
            video_len=video_len,
            tile_top_p=1.0,
            token_top_k=0,
            token_top_ratio=0.1,
            refresh_route=True,
            record_density=True,
        )
        expected = F.scaled_dot_product_attention(q, k, v)

        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
        self.assertEqual(backend.last_stats["core_interactions"], 3 * sequence**2)
        self.assertEqual(backend.last_stats["total_possible"], 3 * sequence**2)

    def test_padding_suffix_is_an_independent_varlen_segment(self) -> None:
        torch.manual_seed(4)
        sequence = 129
        valid_sequence = 120
        q = torch.randn(1, 3, sequence, 16)
        k = torch.randn_like(q)
        v = torch.randn_like(q)
        backend = FlashInfer64Attention()
        actual = backend(
            q,
            k,
            v,
            video_perm=torch.randperm(113),
            video_len=113,
            route_mode="topk_topp",
            tile_top_p=0.25,
            tile_top_ratio=1.0,
            token_top_k=None,
            token_top_ratio=0.1,
            token_top_p=0.0,
            valid_sequence=valid_sequence,
            refresh_route=True,
            record_density=True,
        )
        expected = torch.cat(
            (
                F.scaled_dot_product_attention(
                    q[:, :, :valid_sequence],
                    k[:, :, :valid_sequence],
                    v[:, :, :valid_sequence],
                ),
                F.scaled_dot_product_attention(
                    q[:, :, valid_sequence:],
                    k[:, :, valid_sequence:],
                    v[:, :, valid_sequence:],
                ),
            ),
            dim=2,
        )
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
        allowed = 3 * (valid_sequence**2 + (sequence - valid_sequence) ** 2)
        self.assertEqual(backend.last_stats["core_interactions"], allowed)
        self.assertEqual(backend.last_stats["total_possible"], allowed)

    def test_fa3_vector_workspaces_grow_from_exact_plan_sizes(self) -> None:
        class FakeWrapper:
            _backend = "fa3"
            _paged_kv_indices_buf = torch.empty(12, dtype=torch.int32)
            _paged_kv_indptr_buf = torch.empty(7, dtype=torch.int32)
            _vector_sparse_indices_buffer = torch.empty(4, dtype=torch.int32)
            _vector_sparse_indptr_buffer = torch.empty(3, dtype=torch.int32)
            _float_workspace_buffer = torch.empty(1, dtype=torch.uint8)
            _int_workspace_buffer = torch.empty(1, dtype=torch.uint8)

            def reset_workspace_buffer(self, **kwargs) -> None:
                self._vector_sparse_indices_buffer = kwargs[
                    "vector_sparse_indices_buffer"
                ]
                self._vector_sparse_indptr_buffer = kwargs[
                    "vector_sparse_indptr_buffer"
                ]

        wrapper = FakeWrapper()
        _ensure_flashinfer_vector_workspace(wrapper)
        self.assertGreater(
            wrapper._vector_sparse_indices_buffer.numel(),
            wrapper._paged_kv_indices_buf.numel(),
        )
        self.assertGreater(
            wrapper._vector_sparse_indptr_buffer.numel(),
            wrapper._paged_kv_indptr_buf.numel(),
        )

    def test_tile_topk_core_plus_token_topp_matches_union_mask(self) -> None:
        torch.manual_seed(3)
        sequence = 129
        q = torch.randn(1, 2, sequence, 16)
        k = torch.randn_like(q)
        v = torch.randn_like(q)
        backend = FlashInfer64Attention()

        actual = backend(
            q,
            k,
            v,
            video_perm=None,
            video_len=None,
            route_mode="topk_topp",
            tile_top_p=0.25,
            tile_top_ratio=0.2,
            token_top_k=None,
            token_top_ratio=0.1,
            token_top_p=0.5,
            refresh_route=True,
            record_density=True,
        )
        self.assertTrue(bool((backend.route.core_mask.sum(dim=-1) == 1).all()))

        scores = compute_64_tile_scores(q[0], k[0])
        indices, valid, _ = _build_residual_topk(
            scores,
            backend.route.core_mask,
            sequence=sequence,
            token_top_k=None,
            token_top_ratio=0.1,
            token_top_p=0.5,
        )
        union = backend.route.core_mask.repeat_interleave(64, 1).repeat_interleave(
            64, 2
        )[:, :sequence, :sequence]
        residual = torch.zeros_like(union)
        head, row, slot = valid.nonzero(as_tuple=True)
        residual[head, row, indices[head, row, slot]] = True
        union |= residual
        logits = torch.matmul(q[0], k[0].transpose(1, 2)) / (q.shape[-1] ** 0.5)
        logits.masked_fill_(~union, float("-inf"))
        expected = torch.matmul(torch.softmax(logits, dim=-1), v[0])[None]

        torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)
        self.assertEqual(
            backend.last_stats["core_interactions"]
            + backend.last_stats["residual_token_interactions"],
            int(union.sum()),
        )

    def test_hyvideo_text_blocks_and_queries_are_always_dense(self) -> None:
        # Five 64-token blocks; video ends inside block 3, so blocks 3-4 are
        # dense text/boundary blocks. At ratio 0.6 the total budget is three:
        # two forced text blocks plus one ranked fully-video block.
        scores = torch.tensor(
            [[[0.30, 0.10, 0.20, 0.25, 0.15]]], dtype=torch.float32
        ).expand(2, 5, 5).clone()
        mask = _select_hyvideo_core_tiles(
            scores,
            route_mode="topk_topp",
            tile_top_p=0.9,
            tile_top_ratio=0.6,
            sequence=300,
            video_len=220,
        )
        self.assertTrue(bool(mask[:, :3, 3:].all()))
        self.assertTrue(bool(mask[:, 3:, :].all()))
        self.assertTrue(bool((mask[:, :3, :3].sum(dim=-1) == 1).all()))


if __name__ == "__main__":
    unittest.main()
