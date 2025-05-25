import torch
import einops

class FlashAttentionPy(torch.autograd.Function):
    def forward(ctx, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, is_causal=False):
        DEVICE = Q.device
        TILE_SIZE = 16
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
                attn = torch.exp(attn - m.unsqueeze(-1))
                assert new_m.shape == m.shape, new_m.shape
                m_diff = torch.exp(new_m - m)
                l.mul_(m_diff)
                l.add_(torch.sum(attn, dim=-1))
                m.copy_(new_m)
                o.mul_(m_diff.unsqueeze(-1))
                o.add_(einops.einsum(attn, v, "... b_q b_k, ... b_k d -> ... b_q d"))
            o.div_(l.unsqueeze(-1))
            l.copy_(m + torch.log(l))

        print(f'O = {O}, L = {L}')
        ctx.save_for_backward(L)
        return O



