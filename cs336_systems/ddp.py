import os
import torch
import torch.multiprocessing as mp
import torch.distributed as dist
import time
from torch.nn.parallel import DistributedDataParallel as DDP

from cs336_basics.model import BasicsTransformerLM
VOCAB=2048
CTX_LEN=20
D_MODEL=256
LAYERS=6
NUM_HEADS=4
D_FF=384
ROPE_THETA=10000
def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "2044"
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def run_model(rank, world_size, data):
    setup(rank, world_size)
    model = DDP(BasicsTransformerLM(VOCAB, CTX_LEN, D_MODEL, LAYERS, NUM_HEADS, D_FF, ROPE_THETA))
    optimizer = torch.optim.AdamW(model.parameters())
    res = model.forward(data[rank:rank + 1, :])
    lossFunc = torch.nn.MSELoss()
    loss = lossFunc(res, torch.zeros_like(res))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

if __name__ == "__main__":
    world_size = 4
    data = torch.randint(0, VOCAB, (4, CTX_LEN))
    mp.spawn(fn=run_model, args=(world_size, data), nprocs=world_size, join=True)
