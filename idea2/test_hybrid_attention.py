"""CPU correctness tests for the same-mask Hybrid-DFSAttn prototype."""

from __future__ import annotations

import sys
from pathlib import Path
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dfsattn.hybrid_block_attention import fine_sparse_attention, hybrid_sparse_attention, partition_block_mask


class HybridAttentionTest(unittest.TestCase):
    def test_same_logical_mask_and_partial_macro_tile(self) -> None:
        torch.manual_seed(7)
        # Sequence 80 exercises both a promoted 64x64 region and tail padding.
        q = torch.randn(1, 2, 80, 16)
        k = torch.randn_like(q)
        v = torch.randn_like(q)
        mask = torch.zeros(2, 5, 5, dtype=torch.bool)
        mask[:, :4, :4] = True       # occupancy 16: promoted at n*=8
        mask[:, 4, 4] = True         # tail query stays on the residual path

        candidate, plan = hybrid_sparse_attention(q, k, v, mask, threshold=8, return_plan=True)
        reference = fine_sparse_attention(q, k, v, mask)
        self.assertEqual(plan.core_tiles, 2)
        self.assertEqual(plan.residual_tiles, 2)
        self.assertTrue(torch.allclose(candidate, reference, rtol=2e-5, atol=2e-6))

    def test_threshold_keeps_low_occupancy_region_residual(self) -> None:
        mask = torch.zeros(1, 4, 4, dtype=torch.bool)
        mask[0, torch.arange(4), torch.arange(4)] = True
        plan = partition_block_mask(mask, threshold=8)
        self.assertEqual(plan.core_tiles, 0)
        self.assertEqual(plan.residual_tiles, 4)


if __name__ == "__main__":
    unittest.main()
