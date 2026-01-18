# MoE-ReFT

Experiments with Representation Fine-Tuning (ReFT) on Mixture-of-Experts models, specifically [OLMoE-1B-7B](https://huggingface.co/allenai/OLMoE-1B-7B-0125).

## Project Scope

1. MoE fine-tuning with ReFT (LoReFT/DiReFT interventions), full parameter, and LoRA
2. GSM8K math reasoning dataset fine-tuning
3. Analysis of expert activation patterns with/without interventions

## Quick Start

### Training

```bash
# ReFT intervention training (2 GPUs)
uv run torchrun --nnodes 1 --nproc_per_node 2 moe_reft/train.py
```

Config: `moe_reft/configs/olmoe.yaml`

### Expert Analysis

Compare base model vs. fine-tuned model expert activations on GSM8K test set:

```bash
# Run analysis (base model on GPU 2, intervention model on GPU 3)
uv run python -m moe_reft.analyze_experts

# Quick test
uv run python -m moe_reft.analyze_experts --max-batches 10
```

## Results

Results in `analysis_results/`:
- `expert_distribution_comparison.png` - Aggregated expert usage histogram
- `expert_distribution_per_layer.png` - Per-layer (16 layers × 64 experts) heatmaps
- `analysis_results.json` - Numerical results

**Key findings (GSM8K test set):**
- Base model loss: 0.81, Intervention model loss: 0.91
- All 64 experts remain active in both models
- Interventions shift expert routing patterns but don't disable experts
- This was a failed experiment with the test loss actually increase on the test set and same experts are getting activated for both base and interventions finetuned model

## Structure

```
moe_reft/
├── train.py              # DDP training script
├── analyze_experts.py    # Expert activation analysis
├── interventions.py      # LoReFT/DiReFT implementations
├── olmoe/                # Custom OLMoE model with intervention hooks
└── configs/olmoe.yaml    # Training configuration
```
