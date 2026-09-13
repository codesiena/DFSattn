import json
import os
import tempfile
import unittest
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from dfsattn.attention_hyvideo import _compute_flashinfer64_tile_schedule
from dfsattn.flashinfer64_attention import (
    FlashInfer64Attention,
    K_MACRO,
    MICRO,
    Q_MACRO,
    _build_residual_csr,
    _build_residual_csr_from_fine_indices,
    _build_rode_center_csr,
    _ensure_flashinfer_vector_workspace,
    _compute_head_occupancy_tau_stats,
    compute_peak_aware_micro_tile_scores,
    _fine_topk_occupancy_route,
    _force_dense_heads,
    _load_high_omission_heads,
    _merge_lse_active_rows,
    _promote_residual_microtiles,
    _replay_residual_mask,
    _select_residual_to_total_mass,
    _select_hyvideo_core_tiles,
    compute_64_tile_scores,
    select_64_tiles_from_scores,
    select_fine_topk_from_scores,
)
from dfsattn.utils.timing import AttentionTimingRecorder
from analyze_macro_topk_error import fixed_macro_topk, macro_structure


class FlashInfer64AttentionTest(unittest.TestCase):
    def test_dynamic_macro_ratio_matches_native_dfsattn_schedule(self) -> None:
        expected = {
            11: (False, 0.30),
            12: (True, 0.30),
            23: (False, 0.30),
            24: (True, 0.20),
            35: (False, 0.20),
            36: (True, 0.10),
            47: (False, 0.10),
            48: (False, 0.10),
            49: (False, 0.10),
        }
        for step_idx, expected_value in expected.items():
            actual = _compute_flashinfer64_tile_schedule(
                step_idx=step_idx,
                skip_steps=12,
                cache_interval=12,
                tile_top_ratio=0.30,
                sparsity_dcrt=0.10,
                dynamic_tile_ratio=True,
            )
            self.assertEqual(actual[0], expected_value[0])
            self.assertAlmostEqual(actual[1], expected_value[1])

    def test_fixed_macro_ratio_keeps_periodic_route_refresh(self) -> None:
        for step_idx, refresh in ((12, True), (24, True), (36, True), (48, True)):
            actual_refresh, ratio = _compute_flashinfer64_tile_schedule(
                step_idx=step_idx,
                skip_steps=12,
                cache_interval=12,
                tile_top_ratio=0.195,
                sparsity_dcrt=0.10,
                dynamic_tile_ratio=False,
            )
            self.assertEqual(actual_refresh, refresh)
            self.assertAlmostEqual(ratio, 0.195)

    def test_sampled_lse_micro_scores_are_normalized_and_chunkable(self) -> None:
        torch.manual_seed(36)
        q = torch.randn(2, 33, 8)
        k = torch.randn_like(q)
        scores = compute_peak_aware_micro_tile_scores(
            q, k, temperature=1.0, q_chunk_size=2,
        )
        self.assertEqual(tuple(scores.shape), (2, 3, 3))
        self.assertTrue(bool(torch.isfinite(scores).all()))
        torch.testing.assert_close(
            scores.sum(-1), torch.ones_like(scores.sum(-1)), atol=1e-6, rtol=1e-6,
        )

    def test_topp_topk_residual_check_is_limited_to_configured_heads(self) -> None:
        torch.manual_seed(35)
        sequence = 193
        q = torch.randn(1, 2, sequence, 16)
        k, v = torch.randn_like(q), torch.randn_like(q)
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as handle:
            handle.write("# test risk prior\nLayer2 / Head1\n")
            path = handle.name
        try:
            parsed, _ = _load_high_omission_heads(path)
            self.assertEqual(parsed, {2: (1,)})
            backend = FlashInfer64Attention()
            backend(
                q, k, v,
                video_perm=None,
                video_len=None,
                route_mode="topp_topk",
                tile_top_p=0.1,
                token_top_k=None,
                token_top_ratio=0.1,
                token_top_p=0.0,
                promotion_threshold=48,
                high_omission_heads_file=path,
                layer_idx=2,
                refresh_route=True,
                reuse_route=True,
            )
            residual = backend.route.residual_mask
            self.assertEqual(int(residual[0].sum()), 0)
            self.assertGreater(int(residual[1].sum()), 0)
        finally:
            os.unlink(path)

    def test_topk_topp_sampled_lse_scorer_runs_only_for_risk_head(self) -> None:
        torch.manual_seed(37)
        q = torch.randn(1, 2, 193, 16)
        k, v = torch.randn_like(q), torch.randn_like(q)
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as handle:
            handle.write("Layer2 / Head1\n")
            path = handle.name
        try:
            backend = FlashInfer64Attention()
            backend(
                q, k, v,
                video_perm=None,
                video_len=None,
                route_mode="topk_topp",
                tile_top_p=0.1,
                tile_top_ratio=0.16,
                token_top_k=None,
                token_top_ratio=0.1,
                token_top_p=0.0,
                promotion_threshold=48,
                high_omission_heads_file=path,
                residual_scorer="sampled_lse",
                layer_idx=2,
                refresh_route=True,
                reuse_route=True,
            )
            self.assertEqual(int(backend.route.residual_mask[0].sum()), 0)
            self.assertGreater(int(backend.route.residual_mask[1].sum()), 0)
        finally:
            os.unlink(path)

    def test_replay_mask_is_core_disjoint(self) -> None:
        micro = torch.zeros((1, 16, 18))
        core = torch.zeros((1, 2, 3), dtype=torch.bool)
        core[0, 0, 1] = True
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as handle:
            json.dump({"rows": {"12:12": [[0, 1, 7], [0, 9, 17]]}}, handle)
            path = handle.name
        try:
            replay = _replay_residual_mask(
                micro, core, replay_file=path, step_idx=12, layer_idx=12,
            )
            self.assertFalse(bool(replay[0, 1, 7]))  # already covered by Core
            self.assertTrue(bool(replay[0, 9, 17]))
            self.assertEqual(int(replay.sum()), 1)
        finally:
            os.unlink(path)

    def test_replay_mask_uses_latest_prior_anchor(self) -> None:
        micro = torch.zeros((1, 16, 18))
        core = torch.zeros((1, 2, 3), dtype=torch.bool)
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as handle:
            json.dump({
                "rows": {
                    "12:12": [[0, 1, 7]],
                    "24:12": [[0, 2, 8]],
                    "36:12": [[0, 3, 9]],
                }
            }, handle)
            path = handle.name
        try:
            before_first = _replay_residual_mask(
                micro, core, replay_file=path, step_idx=11, layer_idx=12,
            )
            from_12 = _replay_residual_mask(
                micro, core, replay_file=path, step_idx=23, layer_idx=12,
            )
            from_24 = _replay_residual_mask(
                micro, core, replay_file=path, step_idx=35, layer_idx=12,
            )
            from_36 = _replay_residual_mask(
                micro, core, replay_file=path, step_idx=49, layer_idx=12,
            )
            self.assertEqual(int(before_first.sum()), 0)
            self.assertTrue(bool(from_12[0, 1, 7]))
            self.assertTrue(bool(from_24[0, 2, 8]))
            self.assertTrue(bool(from_36[0, 3, 9]))
        finally:
            os.unlink(path)

    def test_replay_mask_full_q16_row_sentinel(self) -> None:
        micro = torch.zeros((1, 2, 12))
        core = torch.zeros((1, 1, 2), dtype=torch.bool)
        core[0, 0, 0] = True
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as handle:
            json.dump({"rows": {"12:12": [[0, 1, -1]]}}, handle)
            path = handle.name
        try:
            replay = _replay_residual_mask(
                micro, core, replay_file=path, step_idx=12, layer_idx=12,
            )
            self.assertEqual(int(replay[0, 1].sum()), 6)
        finally:
            os.unlink(path)

    def test_replay_route_is_reused_between_refresh_steps(self) -> None:
        torch.manual_seed(34)
        q = torch.randn(1, 1, 193, 16)
        k, v = torch.randn_like(q), torch.randn_like(q)
        backend = FlashInfer64Attention()
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as handle:
            json.dump({"rows": {"12:12": [[0, 0, 12]]}}, handle)
            path = handle.name
        previous = os.environ.get("FLASHINFER64_REPLAY_MASK_FILE")
        os.environ["FLASHINFER64_REPLAY_MASK_FILE"] = path
        common = dict(
            video_perm=None, video_len=None, route_mode="topk_topp",
            tile_top_p=0.25, tile_top_ratio=0.5,
            token_top_k=None, token_top_ratio=0.1, token_top_p=0.9,
            promotion_threshold=24, reuse_route=True, layer_idx=12,
        )
        try:
            backend(q, k, v, refresh_route=True, step_idx=12, **common)
            route = backend.route
            backend(q, k, v, refresh_route=False, step_idx=13, **common)
            self.assertIs(backend.route, route)
        finally:
            if previous is None:
                os.environ.pop("FLASHINFER64_REPLAY_MASK_FILE", None)
            else:
                os.environ["FLASHINFER64_REPLAY_MASK_FILE"] = previous
            os.unlink(path)

    def test_topp_topk_dense_head_override_forces_full_macro_support(self) -> None:
        torch.manual_seed(33)
        sequence = 129
        q = torch.randn(1, 2, sequence, 16)
        k, v = torch.randn_like(q), torch.randn_like(q)
        backend = FlashInfer64Attention()
        actual = backend(
            q,
            k,
            v,
            video_perm=None,
            video_len=None,
            route_mode="topp_topk",
            tile_top_p=0.25,
            tile_top_ratio=0.5,
            token_top_k=None,
            token_top_ratio=0.1,
            token_top_p=0.0,
            dense_layer=12,
            dense_heads=(1,),
            layer_idx=12,
            reuse_route=True,
            refresh_route=True,
            record_density=True,
        )
        route = backend.route
        self.assertTrue(bool(route.core_mask[1].all()))
        self.assertFalse(bool(route.core_mask[0].all()))
        expected_dense = F.scaled_dot_product_attention(
            q[:, 1], k[:, 1], v[:, 1]
        )
        torch.testing.assert_close(actual[0, 1], expected_dense[0], atol=1e-5, rtol=1e-5)
        self.assertEqual(
            backend.last_stats["core_interactions"],
            int(route.core_mask.repeat_interleave(Q_MACRO, 1)
                .repeat_interleave(K_MACRO, 2)[:, :sequence, :sequence].sum()),
        )

    def test_dense_head_override_is_layer_scoped(self) -> None:
        core = torch.zeros((2, 1, 2), dtype=torch.bool)
        _force_dense_heads(core, (1,))
        self.assertTrue(bool(core[1].all()))
        self.assertFalse(bool(core[0].any()))

    def test_rode_center_csr_maps_one_edge_per_residual_microtile(self) -> None:
        route = SimpleNamespace(
            residual_indices=torch.tensor([1, 3, 0], dtype=torch.int32),
            residual_indptr=torch.tensor([0, 2, 2, 3, 3], dtype=torch.int32),
        )
        packed = _build_rode_center_csr(route, heads=1, sequence=64)
        self.assertEqual(packed["columns_cpu"].tolist(), [24, 56, 8])
        self.assertEqual(packed["edge_rows"].tolist(), [8, 8, 40])
        self.assertEqual(packed["active_q16_rows"].tolist(), [0, 2])
        self.assertEqual(packed["nnz"], 3)

    def test_per_head_occupancy_and_tau_stats_reuse_fixed_support(self) -> None:
        micro_scores = torch.full((1, 8, 6), 1.0 / 6.0)
        fine_indices = torch.zeros((1, 8, 2), dtype=torch.int32)
        fine_valid = torch.ones_like(fine_indices, dtype=torch.bool)
        occupancy = torch.tensor([[[16]]], dtype=torch.uint8)
        occupancy_stats, tau_stats = _compute_head_occupancy_tau_stats(
            micro_scores,
            fine_indices,
            fine_valid,
            occupancy,
            sequence=128,
            video_len=None,
        )
        self.assertEqual(len(occupancy_stats), 1)
        self.assertEqual(occupancy_stats[0]["occupancy_histogram"][16], 1)
        self.assertEqual(occupancy_stats[0]["weighted_occupancy"], 16.0)
        self.assertEqual(len(tau_stats), 6)
        tau8 = next(item for item in tau_stats if item["tau"] == 8)
        tau24 = next(item for item in tau_stats if item["tau"] == 24)
        self.assertEqual(tau8["promoted_macro_count"], 1)
        self.assertEqual(tau8["residual_microtiles"], 0)
        self.assertEqual(tau24["promoted_macro_count"], 0)
        self.assertEqual(tau24["residual_microtiles"], 16)

    def test_fine_topk_occupancy_uses_uint8_and_partitions_supports(self) -> None:
        scores = torch.zeros(1, 8, 6)
        scores[0, :, 0] = 0.9
        scores[0, :, 1] = 0.1
        core, fine_indices, residual_valid, occupancy = _fine_topk_occupancy_route(
            scores,
            fine_top_k=2,
            occupancy_threshold=8,
            sequence=128,
            video_len=None,
        )
        self.assertEqual(occupancy.dtype, torch.uint8)
        self.assertEqual(int(occupancy[0, 0, 0]), 16)
        self.assertTrue(bool(core[0, 0, 0]))
        self.assertFalse(bool(residual_valid.any()))
        self.assertEqual(tuple(fine_indices.shape), (1, 8, 2))

    def test_fine_topk_low_occupancy_stays_in_residual(self) -> None:
        scores = torch.zeros(1, 8, 6)
        scores[0, :, 0] = 0.9
        scores[0, :, 1] = 0.1
        core, fine_indices, residual_valid, _ = _fine_topk_occupancy_route(
            scores,
            fine_top_k=1,
            occupancy_threshold=9,
            sequence=128,
            video_len=None,
        )
        mask, indices, indptr, buckets, active_rows, _ = _build_residual_csr_from_fine_indices(
            fine_indices,
            residual_valid,
            q_micro_blocks=8,
            k_micro_blocks=6,
            core_mask=core,
            build_mask=True,
        )
        self.assertFalse(bool(core.any()))
        self.assertEqual(indices.tolist(), [0] * 8)
        self.assertEqual(indptr.tolist(), list(range(0, 9)))
        self.assertEqual([cap for cap, _ in buckets], [4])
        self.assertEqual(active_rows.tolist(), list(range(8)))
        self.assertEqual(int(mask.sum()), 8)

    def test_fine_topk_attention_matches_exact_union_and_has_no_empty_core_nan(self) -> None:
        torch.manual_seed(31)
        sequence = 129
        q = torch.randn(1, 2, sequence, 16)
        k, v = torch.randn_like(q), torch.randn_like(q)
        backend = FlashInfer64Attention()
        actual = backend(
            q, k, v,
            video_perm=None,
            video_len=None,
            route_mode="fine_topk_occupancy",
            tile_top_p=0.25,
            fine_top_k=2,
            token_top_k=None,
            token_top_ratio=0.1,
            token_top_p=0.9,
            promotion_threshold=48,
            refresh_route=True,
            reuse_route=True,
            record_density=True,
        )
        route = backend.route
        support = route.core_mask.repeat_interleave(Q_MACRO, 1).repeat_interleave(
            K_MACRO, 2
        )[:, :sequence, :sequence]
        residual = route.residual_mask.repeat_interleave(MICRO, 1).repeat_interleave(
            MICRO, 2
        )[:, :sequence, :sequence]
        self.assertFalse(bool((support & residual).any()))
        logits = torch.matmul(q[0], k[0].transpose(1, 2)) / (q.shape[-1] ** 0.5)
        union = support | residual
        logits.masked_fill_(~union, float("-inf"))
        expected = torch.matmul(torch.softmax(logits, -1), v[0])[None]
        torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)

    def test_macro_structure_uses_48_way_entropy_and_boundary_valid_count(self) -> None:
        scores = torch.zeros(1, 17, 17)
        scores[0, 0, 0] = 1.0
        structure = macro_structure(scores)
        self.assertEqual(tuple(structure["M_g"].shape), (1, 3, 3))
        self.assertEqual(int(structure["valid_microtiles"][0, -1, -1]), 5)
        self.assertEqual(float(structure["M_g"][0, 0, 0]), 1.0)
        self.assertLess(float(structure["H_g"][0, 0, 0]), 0.1)
        self.assertEqual(float(structure["C_g"][0, 0, 0]), 1.0)

    def test_fixed_macro_topk_preserves_hyvideo_dense_tail_policy(self) -> None:
        scores = torch.tensor([[[0.9, 0.1, 0.0, 0.0]]])
        mask = fixed_macro_topk(scores, 0.2, sequence=300, video_len=220)
        self.assertTrue(bool(mask[:, 1:, :].all()))
        self.assertTrue(bool(mask[:, :1, 2:].all()))

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
            reset_calls = 0

            def reset_workspace_buffer(self, **kwargs) -> None:
                self.reset_calls += 1
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
        _ensure_flashinfer_vector_workspace(wrapper)
        self.assertEqual(wrapper.reset_calls, 1)

    def test_core_only_skips_all_residual_work(self) -> None:
        torch.manual_seed(21)
        sequence = 193
        q = torch.randn(1, 2, sequence, 16)
        k, v = torch.randn_like(q), torch.randn_like(q)
        backend = FlashInfer64Attention()
        actual = backend(
            q, k, v,
            video_perm=None,
            video_len=None,
            route_mode="topk_topp",
            tile_top_p=0.25,
            tile_top_ratio=0.5,
            token_top_k=None,
            token_top_ratio=0.1,
            token_top_p=1.0,
            promotion_threshold=1,
            reuse_route=True,
            core_only=True,
            refresh_route=True,
            record_density=True,
        )
        route = backend.route
        self.assertTrue(route.core_only)
        self.assertEqual(route.residual_indices.numel(), 0)
        self.assertEqual(route.residual_active_rows.numel(), 0)
        self.assertEqual(route.residual_buckets, ())
        self.assertEqual(backend.last_stats["residual_token_interactions"], 0)

        support = route.core_mask.repeat_interleave(Q_MACRO, 1).repeat_interleave(
            K_MACRO, 2
        )[:, :sequence, :sequence]
        logits = torch.matmul(q[0], k[0].transpose(1, 2)) / (q.shape[-1] ** 0.5)
        logits.masked_fill_(~support, float("-inf"))
        expected = torch.matmul(torch.softmax(logits, dim=-1), v[0])[None]
        torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)

    def test_active_row_merge_does_not_touch_inactive_q16_rows(self) -> None:
        torch.manual_seed(22)
        core_out = torch.randn(1, 33, 8)
        residual_out = torch.randn_like(core_out)
        core_lse = torch.randn(1, 33)
        residual_lse = torch.randn(1, 33)
        route = SimpleNamespace(
            residual_active_rows=torch.tensor([1], dtype=torch.int32)
        )
        original = core_out.clone()
        actual = _merge_lse_active_rows(
            core_out, core_lse, residual_out, residual_lse, route
        )
        full = torch.logaddexp(core_lse, residual_lse)
        expected_active = (
            torch.exp(core_lse[..., None] - full[..., None]) * original
            + torch.exp(residual_lse[..., None] - full[..., None]) * residual_out
        )
        expected = original.clone()
        expected[:, 16:32] = expected_active[:, 16:32]
        torch.testing.assert_close(actual, expected)

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
        indices, indptr, buckets, stats = _build_residual_csr(residual)
        self.assertTrue(bool(core.all()))
        self.assertFalse(bool(residual.any()))
        self.assertEqual(indices.numel(), 0)
        self.assertEqual(indptr.tolist(), [0] * (residual.shape[0] * residual.shape[1] + 1))
        self.assertEqual(buckets, ())
        self.assertEqual(stats[0], 0.0)
        self.assertEqual(promoted, 1)

    def test_residual_csr_and_length_buckets_preserve_mask(self) -> None:
        residual = torch.zeros((2, 3, 10), dtype=torch.bool)
        residual[0, 0, [1, 7]] = True
        residual[0, 2, [0, 2, 4, 6, 8]] = True
        residual[1, 1, [3]] = True
        indices, indptr, buckets, stats = _build_residual_csr(residual)

        rebuilt = torch.zeros_like(residual).view(6, 10)
        for row in range(6):
            cols = indices[indptr[row] : indptr[row + 1]].long()
            rebuilt[row, cols] = True
        torch.testing.assert_close(rebuilt.view_as(residual), residual)
        self.assertEqual([cap for cap, _ in buckets], [4, 8])
        bucket_rows = {cap: rows.tolist() for cap, rows in buckets}
        self.assertEqual(bucket_rows[4], [0, 4])
        self.assertEqual(bucket_rows[8], [2])
        self.assertAlmostEqual(stats[0], 8.0 / 6.0)
        self.assertEqual(stats[1], 0.5)
        self.assertEqual(stats[3], 5.0)
        self.assertAlmostEqual(stats[4], 0.5)

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

    def test_cpu_fallback_reuses_route_but_rebuilds_reference_plan(self) -> None:
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
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as handle:
            handle.write("Layer0 / Head0\n")
            path = handle.name
        try:
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
                high_omission_heads_file=path,
                layer_idx=0,
                refresh_route=True,
                timing_recorder=recorder,
            )
        finally:
            os.unlink(path)
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
