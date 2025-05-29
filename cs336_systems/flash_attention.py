import torch
import einops

TILE_SIZE: int = 16

class FlashAttentionPy(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, is_causal=False):
        DEVICE = Q.device
        num_tiles = Q.shape[-2] // TILE_SIZE
        scale = Q.shape[-1] ** -0.5
        assert Q.shape == K.shape
        O = torch.zeros_like(Q)
        L = torch.zeros(*Q.shape[:-1], device=DEVICE) # logsumexp, N_q

        row = zip(
            torch.split(Q, TILE_SIZE, dim=-2),
            O.split(TILE_SIZE, dim=-2),
            L.split(TILE_SIZE, dim=-1),
        )
        for i, (q, o, l) in enumerate(row):
            m = torch.full(l.shape, float('-inf'), device=DEVICE)
            col = zip(
                K.split(TILE_SIZE, dim=-2),
                V.split(TILE_SIZE, dim=-2)
            )
            for j, (k, v) in enumerate(col):
                attn = einops.einsum(q, k, "... b_q d, ... b_k d -> ... b_q b_k") * scale
                block_row_max = torch.amax(attn, dim=-1)
                new_m = torch.maximum(m, block_row_max)
                attn = torch.exp(attn - new_m.unsqueeze(-1))
                assert new_m.shape == m.shape, new_m.shape
                m_diff = torch.exp(m - new_m)
                l.mul_(m_diff)
                l.add_(torch.sum(attn, dim=-1))
                m.copy_(new_m)
                o.mul_(m_diff.unsqueeze(-1))
                o.add_(einops.einsum(attn, v, "... b_q b_k, ... b_k d -> ... b_q d"))
            o.div_(l.unsqueeze(-1))
            l.copy_(m + torch.log(l))

        ctx.save_for_backward(L)
        return O

import triton
import triton.language as tl

@triton.jit
def flash_attn_kernel(
    q_ptr, k_ptr, v_ptr, o_ptr, l_ptr,
    q_stride_batch, q_stride_n, q_stride_d, 
    l_stride_batch, l_stride_n,
    N,
    TILE_SIZE: tl.constexpr,
    D: tl.constexpr
):
    # Parallelize over rows in batches
    row_idx = tl.program_id(0) * TILE_SIZE + tl.arange(0, TILE_SIZE)
    b_idx = tl.program_id(1)
    
    # Separate indices for sequence and feature dimensions
    seq_idx = tl.arange(0, TILE_SIZE)  # For sequence dimension
    feat_idx = tl.arange(0, D)         # For feature dimension
    
    # Calculate batch offsets
    q_batch_offset = b_idx * q_stride_batch
    l_batch_offset = b_idx * l_stride_batch
    
    # Initialize accumulators
    m = tl.full((TILE_SIZE,), float('-inf'), dtype=tl.float32)
    l_accum = tl.zeros((TILE_SIZE,), dtype=tl.float32)
    o = tl.zeros((TILE_SIZE, D), dtype=tl.float32)
    
    # Load query tile - shape (TILE_SIZE, D)
    q_offsets = q_batch_offset + row_idx[:, None] * q_stride_n + feat_idx[None, :] * q_stride_d
    q_mask = (row_idx < N)[:, None] & (feat_idx < D)[None, :]
    q = tl.load(q_ptr + q_offsets, mask=q_mask, other=0.0)
    
    # Loop over key/value tiles
    num_tiles = tl.cdiv(N, TILE_SIZE)
    for tile_idx in range(num_tiles):
        # Calculate column indices for this tile
        col_start = tile_idx * TILE_SIZE
        col_indices = col_start + seq_idx
        
        # Load key & value tile - shape (TILE_SIZE, D)
        k_offsets = q_batch_offset + col_indices[:, None] * q_stride_n + feat_idx[None, :] * q_stride_d
        k_mask = (col_indices < N)[:, None] & (feat_idx < D)[None, :]
        k = tl.load(k_ptr + k_offsets, mask=k_mask, other=0.0)
        v = tl.load(v_ptr + k_offsets, mask=k_mask, other=0.0)
        
        # Compute attention scores - shape (TILE_SIZE, TILE_SIZE)
        attn = tl.dot(q, tl.trans(k)) * (D ** -0.5)
        
        # Update with numerical stability
        row_max = tl.max(attn, axis=1)
        new_m = tl.maximum(m, row_max)
        exp_attn = tl.exp(attn - new_m[:, None])
        exp_m_diff = tl.exp(m - new_m)
        
        # Update accumulators
        l_accum = l_accum * exp_m_diff + tl.sum(exp_attn, axis=1)
        o = o * exp_m_diff[:, None] + tl.dot(exp_attn, v)  # Now shapes match
        m = new_m
    
    # Normalize output
    EPSILON = 1e-6
    o = o / (l_accum[:, None] + EPSILON)
    
    # Store output
    tl.store(o_ptr + q_offsets, o, mask=q_mask)
    
    # Store log-sum-exp
    l_offsets = l_batch_offset + row_idx * l_stride_n
    l_mask = row_idx < N
    tl.store(l_ptr + l_offsets, m + tl.log(l_accum), mask=l_mask)

class FlashAttentionTriton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, is_causal=False):
        DEVICE = Q.device
        B, N, D = Q.shape
        
        # Initialize output and LSE buffers
        O = torch.empty_like(Q)
        L = torch.empty((B, N), device=DEVICE, dtype=torch.float32)
        
        # Calculate grid size
        grid = (triton.cdiv(N, TILE_SIZE), B)
        
        flash_attn_kernel[grid](
            Q, K, V, O, L,
            Q.stride(0), Q.stride(1), Q.stride(2),
            L.stride(0), L.stride(1),
            N,
            TILE_SIZE=TILE_SIZE, # type: ignore
            D=D # type: ignore
        )
        
        ctx.save_for_backward(Q, K, V, O, L)
        return O
