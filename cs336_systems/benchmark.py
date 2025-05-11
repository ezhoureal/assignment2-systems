import time
from cs336_basics import model
import torch
class ModelSize:
    d_model: int
    d_ff: int
    num_layers: int
    num_heads: int
    def __init__(self, d_model, d_ff, num_layers, num_heads):
        self.d_model = d_model
        self.d_ff = d_ff
        self.num_layers = num_layers
        self.num_heads = num_heads

SMALL = ModelSize(d_model=768, d_ff=3072, num_layers=12, num_heads=12)
MEDIUM = ModelSize(d_model=1024, d_ff=4096, num_layers=24, num_heads=16)
LARGE = ModelSize(d_model=1280, d_ff=5120, num_layers=36, num_heads=20)
XL = ModelSize(d_model=1600, d_ff=6400, num_layers=48, num_heads=25)
XXL = ModelSize(d_model=2560, d_ff=10240, num_layers=32, num_heads=32)

W = 5
CONTEXT_LENGTH = 512
ROPE_THETA = 10000
VOCAB_SIZE = 32768
B = 1

device = (
    torch.device("cuda") if torch.cuda.is_available() else
    torch.device("cpu")
)

def benchmark(size: ModelSize, forward_only: bool, steps: int):
    m = model.BasicsTransformerLM(vocab_size=VOCAB_SIZE, 
                                  context_length=CONTEXT_LENGTH, 
                                  d_model=size.d_model, 
                                  num_layers=size.num_layers, 
                                  num_heads=size.num_heads, 
                                  d_ff=size.d_ff, 
                                  rope_theta=ROPE_THETA
                                  ).to(device)
    random_batch = torch.randint(0, VOCAB_SIZE, (B, CONTEXT_LENGTH), dtype=torch.int32, device=device)
    stats: list[int] = []
    for i in range(W + steps):
        # warmup
        start_time = time.time()
        pred = m(random_batch)
        if not forward_only:
            loss = torch.mean(pred)
            loss.backward()
        if device == torch.device('cuda'):
            torch.cuda.synchronize()
        end_time = time.time()
        if i >= W:
            # record
            stats.append(end_time - start_time)
    print(stats)
    avg = sum(stats) / len(stats)
    std = (sum((x - avg) ** 2 for x in stats) / len(stats)) ** 0.5
    print(f'avg time = {avg:.3f} seconds, std = {std:.3f}')

benchmark(LARGE, False, 5)
