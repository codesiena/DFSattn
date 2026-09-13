import math
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analyze_hyvideo_k16_individual_addback import (  # noqa: E402
    compute_k16_addbacks,
)


def _attention(q, k, v, indices):
    logits = q.float() @ k[indices].float().T / math.sqrt(q.shape[-1])
    return torch.softmax(logits, -1) @ v[indices].float()


def test_independent_k16_addback_matches_bruteforce():
    generator = torch.Generator().manual_seed(7)
    sequence, dim = 192, 16
    q = torch.randn(sequence, dim, generator=generator)
    k = torch.randn(sequence, dim, generator=generator)
    v = torch.randn(sequence, dim, generator=generator)
    core = torch.tensor([True, False])
    q16 = 2

    result = compute_k16_addbacks(q, k, v, core, q16, candidate_batch=2)
    q_slice = q[q16 * 16:(q16 + 1) * 16]
    dense = _attention(q_slice, k, v, torch.arange(sequence))
    core_ids = torch.arange(96)
    core_out = _attention(q_slice, k, v, core_ids)
    core_error = torch.linalg.vector_norm(core_out - dense) / torch.linalg.vector_norm(dense)
    assert torch.allclose(
        torch.tensor(result["eq_core_direct"]), core_error, atol=1e-6, rtol=1e-6
    )

    for local in range(6):
        tile_ids = torch.arange(96 + local * 16, 96 + (local + 1) * 16)
        output = _attention(q_slice, k, v, torch.cat((core_ids, tile_ids)))
        error = torch.linalg.vector_norm(output - dense) / torch.linalg.vector_norm(dense)
        assert torch.allclose(result["tile_errors"][6 + local], error, atol=1e-6, rtol=1e-6)

    macro_out = _attention(q_slice, k, v, torch.arange(sequence))
    macro_error = torch.linalg.vector_norm(macro_out - dense) / torch.linalg.vector_norm(dense)
    assert torch.allclose(result["macro_errors"][1], macro_error, atol=1e-6, rtol=1e-6)
