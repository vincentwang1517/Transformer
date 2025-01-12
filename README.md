
## [Mastering LLM Techniques: Inference Optimization](https://developer.nvidia.com/blog/mastering-llm-techniques-inference-optimization/)

```python
## LLM inference
prefill vs. decode 

## Prefill / KV cache
encoder output (seqlen, d_model) -> 

## KV Cache
```

One common optimization for the decode phase is KV caching. The decode phase generates a single token at each time step, but each token depends on the key and value tensors of all previous tokens (including the input tokens’ KV tensors computed at prefill, and any new KV tensors computed until the current time step). 
