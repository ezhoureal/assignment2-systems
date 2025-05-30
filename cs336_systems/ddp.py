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
STEPS=5
def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "2044"
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def run_model(rank, world_size, data):
    setup(rank, world_size)
    model = BasicsTransformerLM(VOCAB, CTX_LEN, D_MODEL, LAYERS, NUM_HEADS, D_FF, ROPE_THETA)
    optimizer = torch.optim.AdamW(model.parameters())
    communication_time = 0.0
    total_time = 0.0
    for step in range(STEPS):
        start_time = time.time()

        res = model.forward(data[rank:rank + 1, :])
        lossFunc = torch.nn.MSELoss()
        loss = lossFunc(res, torch.zeros_like(res))
        optimizer.zero_grad()
        loss.backward()

        com_start_time = time.time()
        grads = [param.grad.data for param in model.parameters() if param.grad is not None]
        flattened_grads = torch._utils._flatten_dense_tensors(grads)
        dist.all_reduce(flattened_grads)
        unflatted_grads = torch._utils._unflatten_dense_tensors(flattened_grads, grads)
        for grad, unflat_grad in zip(grads, unflatted_grads):
            grad.copy_(unflat_grad)
        communication_time += time.time() - com_start_time

        optimizer.step()

        total_time += time.time() - start_time
    if rank == 0:
        print(f'total time = {total_time}, communication time = {communication_time}')

class DDPWrapper(torch.nn.Module):

    def __init__(self, module: torch.nn.Module):
        super().__init__()
        self.module = module
        self.handles = []
        for param in module.parameters():
            dist.broadcast(param.data, src=0, async_op=False)

        def grad_hook(grad):
            print(f'hook')
            handle = dist.all_reduce(grad, op=dist.ReduceOp.AVG, async_op=False)
            self.handles.append(handle)

        for param in module.parameters():
            if param.requires_grad:
                param.register_post_accumulate_grad_hook(grad_hook)


    def forward(self, *inputs, **kwargs):
        for buffer in self.module.buffers():
            dist.broadcast(buffer.data, src=0, async_op=False)
        return self.module.forward(*inputs, **kwargs)
    
    def finish_gradient_synchronization(self):
        print(f'handle count = {self.handles.__len__()}')
        for handle in self.handles:
            handle.wait()
        self.handles.clear()

if __name__ == "__main__":
    world_size = 4
    data = torch.randint(0, VOCAB, (4, CTX_LEN))
    mp.spawn(fn=run_model, args=(world_size, data), nprocs=world_size, join=True)
