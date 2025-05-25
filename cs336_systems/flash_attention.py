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
    B, N,
    TILE_SIZE: tl.constexpr,
    D: tl.constexpr
):
    row_tile_idx = tl.program_id(0)
    b_idx = tl.program_id(1)
    q_block_ptr = tl.make_block_ptr(
        q_ptr,
        (B, N, D),
        (q_stride_batch, q_stride_n, q_stride_d),
        offsets=(b_idx, row_tile_idx * TILE_SIZE, 0),
        block_shape=(1, TILE_SIZE, D),
        order=(2, 1, 0,)
    )
    o_block_ptr = tl.make_block_ptr(
        o_ptr,
        (B, N, D),
        (q_stride_batch, q_stride_n, q_stride_d),
        offsets=(b_idx, row_tile_idx * TILE_SIZE, 0),
        block_shape=(1, TILE_SIZE, D),
        order=(2, 1, 0,)
    )
    l_block_ptr = tl.make_block_ptr(
        l_ptr,
        (B, N),
        (l_stride_batch, l_stride_n,),
        offsets=(b_idx, row_tile_idx * TILE_SIZE,),
        block_shape=(1, TILE_SIZE,),
        order=(1, 0,)
    )
    q = tl.load(q_block_ptr)
    o = tl.load(o_block_ptr)
    l = tl.load(l_block_ptr)
    m = tl.full((TILE_SIZE,), float('-inf'), dtype=tl.float32)
    for col_tile_idx in range(0, N // TILE_SIZE):
        k_block_ptr = tl.make_block_ptr(
            k_ptr,
            (B, N, D),
            (q_stride_batch, q_stride_n, q_stride_d),
            offsets=(b_idx, col_tile_idx * TILE_SIZE, 0),
            block_shape=(1, TILE_SIZE, D),
            order=(2, 1, 0,)
        )
        v_block_ptr = tl.make_block_ptr(
            v_ptr,
            (B, N, D),
            (q_stride_batch, q_stride_n, q_stride_d),
            offsets=(b_idx, col_tile_idx * TILE_SIZE, 0),
            block_shape=(1, TILE_SIZE, D),
            order=(2, 1, 0,)
        )
        k = tl.load(k_block_ptr)
        v = tl.load(v_block_ptr)

        attn = tl.dot(q, tl.trans(k, 0, 2, 1)) * (D ** -0.5)
        block_row_max = tl.max(attn, axis=1)
        new_m = tl.maximum(m, block_row_max)
        attn = tl.exp(attn - new_m[:, None])
        m_diff = tl.exp(m - new_m)
        l *= m_diff
        l += tl.sum(attn, axis=1)
        m = new_m
        o *= m_diff[:, None]
        o += tl.dot(attn, v)
    
    tl.store(o_block_ptr, o / l[:, None])
    tl.store(l_block_ptr, m + tl.log(l))

class FlashAttentionTriton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, is_causal=False):
        DEVICE = Q.device
        B, N, D = Q.shape
        assert Q.shape == K.shape
        assert Q.shape == V.shape
        assert Q.is_cuda and K.is_cuda and V.is_cuda, "FlashAttentionTriton only supports CUDA tensors"
        assert Q.shape[-2] % TILE_SIZE == 0, "The second last dimension of Q must be divisible by TILE_SIZE"
        assert K.shape[-2] % TILE_SIZE == 0, "The second last dimension of K must be divisible by TILE_SIZE"
        assert V.shape[-2] % TILE_SIZE == 0, "The second last dimension of V must be divisible by TILE_SIZE"
        O = torch.zeros_like(Q)
        L = torch.zeros((B, N), device=DEVICE)
        flash_attn_kernel[(N // TILE_SIZE, B)](
            Q, K, V, O, L,
            Q.stride(0), Q.stride(1), Q.stride(2),
            L.stride(0), L.stride(1),
            B, N,
            TILE_SIZE, # type: ignore
            D # type: ignore
        )
        
        ctx.save_for_backward(L)
        return O
