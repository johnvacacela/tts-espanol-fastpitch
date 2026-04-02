import numpy as np
from numba import jit, prange

@jit(nopython=True)
def mas_width1(log_attn_map):
    """mas with hardcoded width=1"""
    neg_inf = log_attn_map.dtype.type(-np.inf)
    log_p = log_attn_map.copy()
    log_p[0, 1:] = neg_inf
    for i in range(1, log_p.shape[0]):
        prev_log1 = neg_inf
        for j in range(log_p.shape[1]):
            prev_log2 = log_p[i-1, j]
            log_p[i, j] += max(prev_log1, prev_log2)
            prev_log1 = prev_log2
    opt = np.zeros_like(log_p)
    one = opt.dtype.type(1)
    j = log_p.shape[1]-1
    for i in range(log_p.shape[0]-1, 0, -1):
        opt[i, j] = one
        if j > 0 and log_p[i-1, j-1] >= log_p[i-1, j]:
            j -= 1
            if j == 0:
                opt[1:i, j] = one
                break
    opt[0, j] = one
    return opt

@jit(nopython=True, parallel=True)
def b_mas(b_log_attn_map, in_lens, out_lens, width=1):
    assert width == 1
    attn_out = np.zeros_like(b_log_attn_map)
    for b in prange(b_log_attn_map.shape[0]):
        out = mas_width1(b_log_attn_map[b, 0, :out_lens[b], :in_lens[b]])
        attn_out[b, 0, :out_lens[b], :in_lens[b]] = out
    return attn_out
