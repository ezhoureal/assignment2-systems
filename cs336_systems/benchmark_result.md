All experiments ran with BatchSize = 1 on my RTX 3070
## Forward Only

| Model Size | Average Time | Std Dev    |
|:-----------|:------------|:-----------|
| Small      | 0.037s      | 0.000      |
| Medium     | 0.088s      | 0.013      |
| Large      | 41.711s     | 23.602     |
| XL         | OutOfMemoryError |         |

*Observed exponential increase in Large model due to memory bottleneck.*

## Forward & Backward

| Model Size | Average Time                                   | Std Dev |
|:-----------|:-----------------------------------------------|:--------|
| Small      | 0.092s                                        | 0.002   |
| Medium     | 0.226s                                        | 0.010   |
| Large      | TLDR (by theory this should take 3 times as long) |        |

## Triton Flash Attention
Setup:
- batch_size = 4
- n_queries = 128
- n_keys = 128
- D = 64

triton time: **0.0027s**, pytorch (no flash attention) time: 0.0363s (10x faster)

### changing D to 256:

triton time: **0.0028s**, pytorch time: 0.0299s

## Distributed

| Data Size | Average Time |
|:----------|:------------|
| 1MB       | 0.0027s     |
| 10MB      | 0.0116s     |
| 100MB     | 0.0686s     |
| 1GB       | 0.847s      |

### naive_ddp
step = 5, model = LLM, world_size = 4, sequence_length = 20, B = 1 (for each worker)

total time = 0.5839047431945801, communication time = 0.26487088203430176

with batched communication (flatten_dense_tensors), communication time shrunk to 0.14197254180908203 (~50% decrease)