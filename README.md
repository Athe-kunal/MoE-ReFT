Start training job

```
uv run torchrun --nnodes 1 --nproc_per_node 2 moe_reft/train.py
```

SCOPE OF THE PROJECT

01/02/2026

1. MoE finetuning with ReFT, Full parameter and LoRA (build from scratch)
2. GSM8k and Ultrachatfeedback dataset finetuning.
3. Integrate flash attention and other optimizations to make it faster