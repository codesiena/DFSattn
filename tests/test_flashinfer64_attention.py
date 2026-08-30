import unittest

import torch
import torch.nn.functional as F

from dfsattn.flashinfer64_attention import (
    FlashInfer64Attention,
    K_MACRO,
    MICRO,
    Q_MACRO,
    _compact_residual_mask,
    _ensure_flashinfer_vector_workspace,
    _promote_residual_microtiles,
    _select_residual_to_total_mass,
    _select_hyvideo_core_tiles,
    compute_64_tile_scores,
    select_64_tiles_from_scores,
)
from dfsattn.utils.timing import AttentionTimingRecorder


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
            route_mode="topp_topk",
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

    def test_macro_topk_plus_micro_topp_matches_union_mask(self) -> None:
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
            reuse_route=True,
        )
        self.assertTrue(bool((backend.route.core_mask.sum(dim=-1) == 1).all()))

        union = backend.route.core_mask.repeat_interleave(Q_MACRO, 1).repeat_interleave(
            K_MACRO, 2
        )[:, :sequence, :sequence]
        residual = backend.route.residual_mask.repeat_interleave(
            MICRO, 1
        ).repeat_interleave(MICRO, 2)[:, :sequence, :sequence]
        self.assertFalse(bool((union & residual).any()))
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
        # Q uses 128-token macro blocks and K uses 96-token macro blocks.
        # video_len=220 gives one fully-video Q block and two fully-video K
        # blocks; all boundary/text blocks are forced dense.
        scores = torch.tensor(
            [[[0.40, 0.20, 0.25, 0.15]]], dtype=torch.float32
        ).expand(2, 3, 4).clone()
        mask = _select_hyvideo_core_tiles(
            scores,
            route_mode="topk_topp",
            tile_top_p=0.9,
            tile_top_ratio=0.6,
            sequence=300,
            video_len=220,
        )
        self.assertEqual(tuple(mask.shape), (2, 3, 4))
        self.assertTrue(bool(mask[:, :1, 2:].all()))
        self.assertTrue(bool(mask[:, 1:, :].all()))
        self.assertTrue(bool((mask[:, :1, :2].sum(dim=-1) == 1).all()))

    def test_occupancy_promotion_removes_residual_overlap(self) -> None:
        # One Q128/K96 macro contains 48 Q16/K16 microtiles.  Uniform mass at
        # p=1 selects all 48, which must promote the macro and clear Residual.
        micro = torch.full((1, 8, 6), 1.0 / 6)
        core = torch.zeros((1, 1, 1), dtype=torch.bool)
        residual = _select_residual_to_total_mass(
            micro, core, total_top_p=1.0,
        )
        core, residual, promoted = _promote_residual_microtiles(
            residual,
            core,
            promotion_threshold=24,
        )
        compact = _compact_residual_mask(residual)
        self.assertTrue(bool(core.all()))
        self.assertFalse(bool(residual.any()))
        self.assertEqual(compact.shape[-1], 0)
        self.assertEqual(promoted, 1)

    def test_residual_uses_absolute_mass_without_renormalizing_complement(self) -> None:
        # Core covers 0.70 original mass.  A total target of 0.80 requires only
        # 0.10 more, so the single rejected 0.12 tile is sufficient.  If the
        # 0.30 complement were renormalized, several rejected tiles would be kept.
        row = torch.tensor([
            0.20, 0.15, 0.12, 0.10, 0.08, 0.05,
            0.12, 0.08, 0.05, 0.03, 0.01, 0.01,
        ])
        micro = row.view(1, 1, 12).expand(1, 8, 12).clone()
        core = torch.tensor([[[True, False]]])
        residual = _select_residual_to_total_mass(
            micro, core, total_top_p=0.80,
        )
        self.assertTrue(bool((residual.sum(-1) == 1).all()))
        self.assertTrue(bool(residual[..., 6].all()))

    def test_route_is_reused_but_expanded_plan_is_rebuilt(self) -> None:
        torch.manual_seed(9)
        q = torch.randn(1, 2, 193, 16)
        k, v = torch.randn_like(q), torch.randn_like(q)
        backend = FlashInfer64Attention()
        plan_calls = 0
        original_plan = backend._plan_core

        def counted_plan(*args, **kwargs):
            nonlocal plan_calls
            plan_calls += 1
            return original_plan(*args, **kwargs)

        backend._plan_core = counted_plan
        common = dict(
            video_perm=None,
            video_len=None,
            route_mode="topk_topp",
            tile_top_p=0.25,
            tile_top_ratio=0.5,
            token_top_k=None,
            token_top_ratio=0.1,
            token_top_p=0.2,
            promotion_threshold=24,
            reuse_route=True,
        )
        backend(q, k, v, refresh_route=True, **common)
        route = backend.route
        backend(q, k, v, refresh_route=False, **common)
        self.assertIs(backend.route, route)
        self.assertEqual(plan_calls, 2)

    def test_bounded_memory_mode_drops_route_after_each_call(self) -> None:
        torch.manual_seed(10)
        q = torch.randn(1, 1, 129, 16)
        k, v = torch.randn_like(q), torch.randn_like(q)
        backend = FlashInfer64Attention()
        backend(
            q, k, v,
            video_perm=None,
            video_len=None,
            route_mode="topk_topp",
            tile_top_p=0.25,
            tile_top_ratio=0.5,
            token_top_k=None,
            token_top_ratio=0.1,
            token_top_p=0.5,
            promotion_threshold=24,
            reuse_route=False,
            refresh_route=True,
        )
        self.assertIsNone(backend.route)

    def test_hierarchical_steps_have_individual_timing_phases(self) -> None:
        torch.manual_seed(12)
        q = torch.randn(1, 1, 193, 16)
        k, v = torch.randn_like(q), torch.randn_like(q)
        recorder = AttentionTimingRecorder(enabled=True)
        FlashInfer64Attention()(
            q, k, v,
            video_perm=None,
            video_len=None,
            route_mode="topk_topp",
            tile_top_p=0.25,
            tile_top_ratio=0.5,
            token_top_k=None,
            token_top_ratio=0.1,
            token_top_p=0.8,
            promotion_threshold=24,
            refresh_route=True,
            timing_recorder=recorder,
        )
        phases = {row["phase"] for row in recorder.rows()}
        self.assertTrue({
            "flashinfer_fine_score",
            "flashinfer_core_score",
            "flashinfer_core_select",
            "flashinfer_residual_select",
            "flashinfer_promotion",
            "flashinfer_residual_compact",
            "flashinfer_plan",
            "flashinfer_core_run",
            "flashinfer_residual_micro_run",
            "flashinfer_lse_merge",
        }.issubset(phases))


if __name__ == "__main__":
    unittest.main()
