"""Partial one-to-one assignment with enough dummy slots for unmatched tokens."""

import torch


@torch.no_grad()
def hungarian_matches(scores, pair_mask, mask_a, mask_b, null_a, null_b):
    """Maximize sum of positive gains over the unmatched baseline.

    A match has gain S_ij - (u_i+u_j)/2. Unmatched tokens have zero gain.
    An (n+m)-square matrix permits every token to be independently unmatched.
    This does not force min(n,m) real matches or filter a forced assignment
    afterwards. Only indices are returned; callers gather differentiable scores.
    """
    from scipy.optimize import linear_sum_assignment
    import numpy as np

    device = scores.device
    cpu_scores, cpu_allowed = scores.detach().double().cpu(), pair_mask.cpu()
    cpu_a, cpu_b = mask_a.cpu(), mask_b.cpu()
    cpu_null_a, cpu_null_b = null_a.detach().double().cpu(), null_b.detach().double().cpu()
    matches = []
    for batch_index in range(scores.shape[0]):
        ids_a = cpu_a[batch_index].nonzero(as_tuple=False).flatten()
        ids_b = cpu_b[batch_index].nonzero(as_tuple=False).flatten()
        n, m = len(ids_a), len(ids_b)
        if not n or not m:
            matches.append(torch.empty(0, 2, dtype=torch.long, device=device))
            continue
        local_scores = cpu_scores[batch_index][ids_a[:, None], ids_b]
        baseline = (cpu_null_a[batch_index, ids_a, None] + cpu_null_b[batch_index, None, ids_b]) / 2
        gain = local_scores - baseline
        allowed = cpu_allowed[batch_index][ids_a[:, None], ids_b] & (gain > 0)
        utility = np.zeros((n + m, n + m), dtype=np.float64)
        utility[:n, :m] = torch.where(allowed, gain, -1e6).numpy()
        rows, columns = linear_sum_assignment(utility, maximize=True)
        real = (rows < n) & (columns < m)
        rows, columns = torch.from_numpy(rows[real]), torch.from_numpy(columns[real])
        keep = allowed[rows, columns]
        indices = torch.stack((ids_a[rows[keep]], ids_b[columns[keep]]), dim=-1)
        matches.append(indices.to(device))
    return matches
