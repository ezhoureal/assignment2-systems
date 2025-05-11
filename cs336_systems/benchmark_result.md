All experiments ran with BatchSize = 1 on my RTX 3070

## Forward only

| Model Size | Average Time | Std Dev |
|------------|--------------|---------|
| Small      | 0.037s       | 0.000   |
| Medium     | 0.088s       | 0.013   |
| Large      | 41.711s      | 23.602  |
| XL         | OutOfMemoryError | 


observed exponential increase in Large model due to memory bottleneck

## Forward & Backward

| Model Size | Average Time | Std Dev |
|------------|--------------|---------|
| Small      | 0.092s       | 0.002   |
| Medium     | 0.226s       | 0.010   |
| Large      | TLDR (by theory this should take 3 times as long) | 