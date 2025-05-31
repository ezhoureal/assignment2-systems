import os
import torch
import torch.multiprocessing as mp
import torch.distributed as dist
import time

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

class ParamBucket:
    def __init__(self, max_size: int):
        self.params = []
        self.max_size = max_size
        self.cur_size = 0
        self.back_propped = 0
        self.handle = None

    def add_param(self, param):
        assert param.requires_grad, param
        self.params.append(param)
        self.cur_size += param.numel() * param.element_size()

    def is_full(self):
        return self.cur_size >= self.max_size
    
    def is_empty(self):
        return self.cur_size > 0
    
    def prepare_back_prop(self):
        self.back_propped = 0
        self.handle = None
    
    def handle_grad_hook(self, param):
        self.back_propped += 1
        if self.back_propped == len(self.params):
            # Flatten and concatenate all gradients
            grads = [p.grad for p in self.params]
            flattened_grads = torch._utils._flatten_dense_tensors(grads)
            
            # Average gradients across workers
            flattened_grads /= dist.get_world_size()
            
            # Batch all_reduce
            self.handle = dist.all_reduce(flattened_grads, op=dist.ReduceOp.SUM, async_op=True)
            
            # Store flattened grads to unflatten after sync
            self.flattened_grads = flattened_grads
    
    def wait_for_grad_sync(self):
        if self.handle is not None:
            self.handle.wait()
            # Unflatten gradients back to individual parameters
            unflattened_grads = torch._utils._unflatten_dense_tensors(self.flattened_grads, [p.grad for p in self.params])
            for param, grad in zip(self.params, unflattened_grads):
                param.grad.copy_(grad)

class DDPWrapper(torch.nn.Module):
    def __init__(self, module: torch.nn.Module, bucket_size_byte: int = 4):
        super().__init__()
        self.module = module
        self.handles = []
        self.buckets: list[ParamBucket] = []
        self.param2Bucket = dict()
        
        def grad_hook(param):
            if param.grad is not None:
                bucket = self.param2Bucket[param]
                bucket.handle_grad_hook(param)
            
        # Initialize parameters (broadcast from rank 0)
        bucket = ParamBucket(bucket_size_byte)
        self.buckets.append(bucket)
        for param in self.module.parameters():
            dist.broadcast(param.data, src=0, async_op=False)
            if param.requires_grad:
                param.register_post_accumulate_grad_hook(grad_hook) 
                self.param2Bucket[param] = bucket
                bucket.add_param(param)
                if bucket.is_full():
                    bucket = ParamBucket(bucket_size_byte)
                    self.buckets.append(bucket)

        if not bucket.is_empty():
            self.buckets.append(bucket)

    def forward(self, *inputs, **kwargs):
        for bucket in self.buckets:
            bucket.prepare_back_prop()
        output = self.module(*inputs, **kwargs)
        # Synchronize buffers after forward
        for buffer in self.module.buffers():
            dist.broadcast(buffer.data, src=0, async_op=False)
        return output

    def finish_gradient_synchronization(self):
        """Call after backward pass (or after gradient accumulation)."""
        for bucket in self.buckets:
            bucket.wait_for_grad_sync()

if __name__ == "__main__":
    world_size = 4
    data = torch.randint(0, VOCAB, (4, CTX_LEN))
    mp.spawn(fn=run_model, args=(world_size, data), nprocs=world_size, join=True)
