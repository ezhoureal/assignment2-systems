import os
import torch
import torch.multiprocessing as mp
import torch.distributed as dist
import time

def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "2044"
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def demo_reduce(rank, world_size, dataSize):
    setup(rank, world_size)
    data = torch.rand(size=dataSize, dtype=torch.float32)
    if rank == 0:
        start_time = time.time()
    dist.all_reduce(data, op=dist.ReduceOp.SUM, async_op=False)
    if rank == 0:
        exec_time = time.time() - start_time
        print(f'after reduce, time used = {exec_time}')

SHAPE_1MB = (100, 100, 25)
SHAPE_10MB = (100, 100, 250)
SHAPE_100MB = (1000, 100, 250)
SHAPE_1GB = (1000, 1000, 250)

if __name__ == "__main__":
    world_size = 4
    mp.spawn(fn=demo_reduce, args=(world_size, SHAPE_1GB), nprocs=world_size, join=True)
