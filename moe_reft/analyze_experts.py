# Aim of this file:
# 1. Load GSM8K test split
# 2. Load base OLMoE model and checkpoint with interventions
# 3. Evaluate test loss for both models
# 4. Analyze and visualize expert activation distribution

from __future__ import annotations

import os
from typing import Any
from collections import defaultdict

import torch
from torch import nn
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from loguru import logger
from transformers import AutoTokenizer
from tqdm import tqdm

from moe_reft import sft_dataset
from moe_reft.olmoe import modeling_olmoe, configuration_olmoe, load_weights


def create_base_model(
    hf_model_name_or_path: str = "allenai/OLMoE-1B-7B-0125",
    map_dtype: torch.dtype = torch.float32,
    map_device: torch.device | None = None,
) -> nn.Module:
    """
    Create a base OLMoE model without interventions (uses Identity layers).
    """
    # Create config without interventions (intervention_config=None triggers Identity layers)
    base_config = configuration_olmoe.OlmoeInterventionsConfig(
        interventions_config=None,
        full_parameter_finetuning=True,  # This ensures Identity layers are used
    )

    # Initialize model
    base_model = modeling_olmoe.OlmoeForCausalLM(base_config)

    # Load pretrained weights
    _, base_model = load_weights.load_hf_into_custom_model(
        hf_model_name_or_path=hf_model_name_or_path,
        custom_model=base_model,
        intervention_patterns=[],  # No intervention patterns for base model
        full_parameter_finetuning=True,
        map_dtype=map_dtype,
        map_device=map_device,
        trust_remote_code=False,
    )

    return base_model


def load_intervention_model(
    pt_file: str,
    hf_model_name_or_path: str = "allenai/OLMoE-1B-7B-0125",
    map_dtype: torch.dtype = torch.float32,
    map_device: torch.device | None = None,
) -> nn.Module:
    """
    Load the model with interventions from a checkpoint.
    """
    model, report = load_weights.load_pretrained_with_interventions_from_checkpoint(
        pt_file=pt_file,
        hf_model_name_or_path=hf_model_name_or_path,
        map_dtype=map_dtype,
        map_device=map_device,
        trust_remote_code=False,
        full_parameter_finetuning=False,
    )
    logger.info(f"Intervention model loaded: {report.summary()}")
    return model


def load_gsm8k_test_dataset(
    tokenizer_model_name: str = "allenai/OLMoE-1B-7B-0125-Instruct",
) -> tuple[sft_dataset.SFTDataset, Any]:
    """
    Load GSM8K test split using SFTDataset.
    """
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_model_name)
    response_template = sft_dataset.extract_response_template(tokenizer)
    response_template_ids = tokenizer(response_template)["input_ids"]

    message_extractor = sft_dataset.KeyBasedMessageExtractor(
        user_key="question",
        assistant_key="answer",
        system_message="You are a helpful math tutor. Solve step by step.",
    )

    val_dataset = sft_dataset.SFTDataset(
        source="openai/gsm8k",
        tokenizer=tokenizer,
        response_template_ids=response_template_ids,
        message_extractor=message_extractor,
        split="test",
        name="main",
    )

    return val_dataset, tokenizer


class ExpertTracker:
    """
    Hook-based tracker for expert activations during forward passes.
    """

    def __init__(self, model: nn.Module):
        self.model = model
        self.expert_counts: dict[int, dict[int, int]] = defaultdict(lambda: defaultdict(int))
        self.total_tokens_per_layer: dict[int, int] = defaultdict(int)
        self.hooks: list[Any] = []
        self._register_hooks()

    def _register_hooks(self) -> None:
        """Register forward hooks on all MoE blocks."""
        for layer_idx, layer in enumerate(self.model.model.layers):
            if hasattr(layer, "mlp") and hasattr(layer.mlp, "gate"):
                hook = layer.mlp.register_forward_hook(self._make_hook(layer_idx))
                self.hooks.append(hook)

    def _make_hook(self, layer_idx: int):
        """Create a hook function for a specific layer."""

        def hook(module, inputs, outputs):
            # outputs is (final_hidden_states, router_logits)
            _, router_logits = outputs
            if router_logits is not None:
                # router_logits: (batch_size * seq_len, num_experts)
                routing_weights = torch.nn.functional.softmax(router_logits.float(), dim=-1)
                top_k = module.top_k
                _, top_k_index = torch.topk(routing_weights, top_k, dim=-1)

                # Count expert usage
                num_tokens = top_k_index.shape[0]
                self.total_tokens_per_layer[layer_idx] += num_tokens

                for expert_indices in top_k_index:
                    for expert_idx in expert_indices.tolist():
                        self.expert_counts[layer_idx][expert_idx] += 1

        return hook

    def reset(self) -> None:
        """Reset all counters."""
        self.expert_counts = defaultdict(lambda: defaultdict(int))
        self.total_tokens_per_layer = defaultdict(int)

    def remove_hooks(self) -> None:
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def get_expert_distribution(self) -> dict[int, np.ndarray]:
        """
        Get the expert distribution per layer.
        Returns a dict mapping layer_idx to an array of expert counts.
        """
        num_experts = 64  # OLMoE has 64 experts
        distributions = {}

        for layer_idx, expert_dict in self.expert_counts.items():
            counts = np.zeros(num_experts)
            for expert_idx, count in expert_dict.items():
                counts[expert_idx] = count
            distributions[layer_idx] = counts

        return distributions

    def get_aggregated_distribution(self) -> np.ndarray:
        """
        Get the aggregated expert distribution across all layers.
        """
        num_experts = 64
        total_counts = np.zeros(num_experts)

        for layer_idx, expert_dict in self.expert_counts.items():
            for expert_idx, count in expert_dict.items():
                total_counts[expert_idx] += count

        return total_counts


def compute_loss(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    max_batches: int | None = None,
) -> float:
    """
    Compute the average loss over the dataloader.
    """
    model.eval()
    total_loss = 0.0
    total_batches = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Computing loss")):
            if max_batches is not None and batch_idx >= max_batches:
                break

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )

            if hasattr(outputs, "loss") and outputs.loss is not None:
                total_loss += outputs.loss.item()
            total_batches += 1

    avg_loss = total_loss / max(total_batches, 1)
    return avg_loss


def collect_expert_statistics(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    max_batches: int | None = None,
) -> ExpertTracker:
    """
    Run inference and collect expert activation statistics.
    """
    model.eval()
    tracker = ExpertTracker(model)

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Collecting expert stats")):
            if max_batches is not None and batch_idx >= max_batches:
                break

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            _ = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

    return tracker


def plot_expert_distributions(
    base_distribution: np.ndarray,
    intervention_distribution: np.ndarray,
    save_path: str,
) -> None:
    """
    Plot histogram comparing expert distributions between base and intervention models.
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    num_experts = len(base_distribution)
    x = np.arange(num_experts)
    width = 0.35

    # Plot 1: Side-by-side comparison
    ax1 = axes[0, 0]
    ax1.bar(x - width / 2, base_distribution, width, label="Base Model", alpha=0.8, color="steelblue")
    ax1.bar(x + width / 2, intervention_distribution, width, label="+ Interventions", alpha=0.8, color="coral")
    ax1.set_xlabel("Expert Index")
    ax1.set_ylabel("Activation Count")
    ax1.set_title("Expert Activation Distribution Comparison")
    ax1.legend()
    ax1.set_xticks(x[::4])  # Show every 4th tick

    # Plot 2: Normalized distributions (as percentages)
    ax2 = axes[0, 1]
    base_norm = base_distribution / max(base_distribution.sum(), 1) * 100
    intervention_norm = intervention_distribution / max(intervention_distribution.sum(), 1) * 100
    ax2.bar(x - width / 2, base_norm, width, label="Base Model", alpha=0.8, color="steelblue")
    ax2.bar(x + width / 2, intervention_norm, width, label="+ Interventions", alpha=0.8, color="coral")
    ax2.set_xlabel("Expert Index")
    ax2.set_ylabel("Activation Percentage (%)")
    ax2.set_title("Normalized Expert Activation Distribution")
    ax2.legend()
    ax2.set_xticks(x[::4])

    # Plot 3: Difference in activations
    ax3 = axes[1, 0]
    difference = intervention_distribution - base_distribution
    colors = ["green" if d > 0 else "red" for d in difference]
    ax3.bar(x, difference, color=colors, alpha=0.8)
    ax3.axhline(y=0, color="black", linestyle="-", linewidth=0.5)
    ax3.set_xlabel("Expert Index")
    ax3.set_ylabel("Difference (Intervention - Base)")
    ax3.set_title("Change in Expert Activations Due to Interventions")
    ax3.set_xticks(x[::4])

    # Plot 4: Summary statistics
    ax4 = axes[1, 1]
    ax4.axis("off")

    # Calculate statistics
    base_active = np.sum(base_distribution > 0)
    intervention_active = np.sum(intervention_distribution > 0)
    base_entropy = -np.sum((base_norm / 100 + 1e-10) * np.log(base_norm / 100 + 1e-10))
    intervention_entropy = -np.sum((intervention_norm / 100 + 1e-10) * np.log(intervention_norm / 100 + 1e-10))

    base_gini = gini_coefficient(base_distribution)
    intervention_gini = gini_coefficient(intervention_distribution)

    stats_text = f"""
    Summary Statistics
    ==================
    
    Active Experts (count > 0):
      - Base Model:       {base_active} / {num_experts}
      - + Interventions:  {intervention_active} / {num_experts}
    
    Total Activations:
      - Base Model:       {int(base_distribution.sum()):,}
      - + Interventions:  {int(intervention_distribution.sum()):,}
    
    Distribution Entropy (higher = more uniform):
      - Base Model:       {base_entropy:.4f}
      - + Interventions:  {intervention_entropy:.4f}
    
    Gini Coefficient (lower = more equal):
      - Base Model:       {base_gini:.4f}
      - + Interventions:  {intervention_gini:.4f}
    
    Top 5 Most Used Experts:
      Base:         {np.argsort(base_distribution)[-5:][::-1].tolist()}
      Interventions: {np.argsort(intervention_distribution)[-5:][::-1].tolist()}
    """

    ax4.text(
        0.1,
        0.9,
        stats_text,
        transform=ax4.transAxes,
        fontsize=11,
        verticalalignment="top",
        fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved expert distribution plot to {save_path}")


def plot_per_layer_distributions(
    base_tracker: ExpertTracker,
    intervention_tracker: ExpertTracker,
    save_path: str,
) -> None:
    """
    Plot per-layer expert distribution heatmaps.
    """
    base_dist = base_tracker.get_expert_distribution()
    intervention_dist = intervention_tracker.get_expert_distribution()

    num_layers = len(base_dist)
    num_experts = 64

    # Create matrices for heatmaps
    base_matrix = np.zeros((num_layers, num_experts))
    intervention_matrix = np.zeros((num_layers, num_experts))

    for layer_idx in range(num_layers):
        if layer_idx in base_dist:
            base_matrix[layer_idx] = base_dist[layer_idx]
        if layer_idx in intervention_dist:
            intervention_matrix[layer_idx] = intervention_dist[layer_idx]

    # Normalize per layer
    base_matrix_norm = base_matrix / (base_matrix.sum(axis=1, keepdims=True) + 1e-10)
    intervention_matrix_norm = intervention_matrix / (intervention_matrix.sum(axis=1, keepdims=True) + 1e-10)

    fig, axes = plt.subplots(1, 3, figsize=(20, 8))

    # Base model heatmap
    im1 = axes[0].imshow(base_matrix_norm, aspect="auto", cmap="viridis")
    axes[0].set_xlabel("Expert Index")
    axes[0].set_ylabel("Layer Index")
    axes[0].set_title("Base Model - Expert Usage per Layer")
    plt.colorbar(im1, ax=axes[0], label="Normalized Usage")

    # Intervention model heatmap
    im2 = axes[1].imshow(intervention_matrix_norm, aspect="auto", cmap="viridis")
    axes[1].set_xlabel("Expert Index")
    axes[1].set_ylabel("Layer Index")
    axes[1].set_title("+ Interventions - Expert Usage per Layer")
    plt.colorbar(im2, ax=axes[1], label="Normalized Usage")

    # Difference heatmap
    diff_matrix = intervention_matrix_norm - base_matrix_norm
    im3 = axes[2].imshow(diff_matrix, aspect="auto", cmap="RdBu_r", vmin=-0.05, vmax=0.05)
    axes[2].set_xlabel("Expert Index")
    axes[2].set_ylabel("Layer Index")
    axes[2].set_title("Difference (Intervention - Base)")
    plt.colorbar(im3, ax=axes[2], label="Difference")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved per-layer distribution plot to {save_path}")


def gini_coefficient(arr: np.ndarray) -> float:
    """Calculate Gini coefficient for measuring inequality in distribution."""
    arr = arr.flatten()
    if arr.sum() == 0:
        return 0.0
    arr = np.sort(arr)
    n = len(arr)
    index = np.arange(1, n + 1)
    return (2 * np.sum(index * arr) / (n * np.sum(arr))) - (n + 1) / n


def main(
    checkpoint_path: str = "math_gsm8k/checkpoint-epoch0.pt",
    hf_model_name: str = "allenai/OLMoE-1B-7B-0125",
    tokenizer_model: str = "allenai/OLMoE-1B-7B-0125-Instruct",
    batch_size: int = 4,
    max_batches: int | None = None,  # Set to None to use full dataset
    output_dir: str = "analysis_results",
    base_device_str: str = "cuda:2",
    intervention_device_str: str = "cuda:3",
) -> None:
    """
    Main analysis function.
    """
    os.makedirs(output_dir, exist_ok=True)

    base_device = torch.device(base_device_str if torch.cuda.is_available() else "cpu")
    intervention_device = torch.device(intervention_device_str if torch.cuda.is_available() else "cpu")
    dtype = torch.float32  # Use float32 for stability

    logger.info(f"Using base model device: {base_device}")
    logger.info(f"Using intervention model device: {intervention_device}")
    logger.info(f"Using dtype: {dtype}")

    # 1. Load GSM8K test dataset
    logger.info("Loading GSM8K test dataset...")
    test_dataset, tokenizer = load_gsm8k_test_dataset(tokenizer_model)
    logger.info(f"Test dataset size: {len(test_dataset)}")

    collate_fn = sft_dataset.SFTDataCollator(tokenizer)
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4,
    )

    # 2. Load base model (without interventions) on GPU 2
    logger.info(f"Loading base OLMoE model on {base_device}...")
    base_model = create_base_model(
        hf_model_name_or_path=hf_model_name,
        map_dtype=dtype,
        map_device=base_device,
    )
    base_model.to(base_device)
    base_model.eval()

    # 3. Load model with interventions from checkpoint on GPU 3
    logger.info(f"Loading intervention model from {checkpoint_path} on {intervention_device}...")
    intervention_model = load_intervention_model(
        pt_file=checkpoint_path,
        hf_model_name_or_path=hf_model_name,
        map_dtype=dtype,
        map_device=intervention_device,
    )
    intervention_model.to(intervention_device)
    intervention_model.eval()

    # 4. Compute test loss for both models
    logger.info("Computing test loss for base model...")
    base_loss = compute_loss(base_model, test_loader, base_device, max_batches=max_batches)
    logger.info(f"Base Model Test Loss: {base_loss:.4f}")

    logger.info("Computing test loss for intervention model...")
    intervention_loss = compute_loss(intervention_model, test_loader, intervention_device, max_batches=max_batches)
    logger.info(f"Intervention Model Test Loss: {intervention_loss:.4f}")

    # 5. Collect expert activation statistics
    logger.info("Collecting expert statistics for base model...")
    base_tracker = collect_expert_statistics(base_model, test_loader, base_device, max_batches=max_batches)

    logger.info("Collecting expert statistics for intervention model...")
    intervention_tracker = collect_expert_statistics(
        intervention_model, test_loader, intervention_device, max_batches=max_batches
    )

    # 6. Get aggregated distributions
    base_distribution = base_tracker.get_aggregated_distribution()
    intervention_distribution = intervention_tracker.get_aggregated_distribution()

    # 7. Plot and save results
    logger.info("Generating plots...")

    # Main comparison plot
    plot_expert_distributions(
        base_distribution,
        intervention_distribution,
        os.path.join(output_dir, "expert_distribution_comparison.png"),
    )

    # Per-layer heatmap
    plot_per_layer_distributions(
        base_tracker,
        intervention_tracker,
        os.path.join(output_dir, "expert_distribution_per_layer.png"),
    )

    # Clean up hooks
    base_tracker.remove_hooks()
    intervention_tracker.remove_hooks()

    # Save numerical results
    results = {
        "base_model_loss": base_loss,
        "intervention_model_loss": intervention_loss,
        "loss_improvement": base_loss - intervention_loss,
        "loss_improvement_percent": (base_loss - intervention_loss) / base_loss * 100 if base_loss > 0 else 0,
        "base_active_experts": int(np.sum(base_distribution > 0)),
        "intervention_active_experts": int(np.sum(intervention_distribution > 0)),
        "base_distribution": base_distribution.tolist(),
        "intervention_distribution": intervention_distribution.tolist(),
    }

    import json

    results_path = os.path.join(output_dir, "analysis_results_256.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved numerical results to {results_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("ANALYSIS SUMMARY")
    print("=" * 60)
    print(f"\nTest Loss Comparison:")
    print(f"  Base Model:          {base_loss:.4f}")
    print(f"  + Interventions:     {intervention_loss:.4f}")
    print(
        f"  Improvement:         {base_loss - intervention_loss:.4f} ({results['loss_improvement_percent']:.2f}%)"
    )
    print(f"\nActive Experts:")
    print(f"  Base Model:          {results['base_active_experts']} / 64")
    print(f"  + Interventions:     {results['intervention_active_experts']} / 64")
    print(f"\nResults saved to: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Analyze expert distributions in OLMoE models")
    parser.add_argument(
        "--checkpoint",
        "-c",
        type=str,
        default="math_gsm8k_256/checkpoint-epoch0.pt",
        help="Path to the intervention checkpoint file",
    )
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default="allenai/OLMoE-1B-7B-0125",
        help="HuggingFace model name for base weights",
    )
    parser.add_argument(
        "--tokenizer", "-t", type=str, default="allenai/OLMoE-1B-7B-0125-Instruct", help="Tokenizer model name"
    )
    parser.add_argument("--batch-size", "-b", type=int, default=4, help="Batch size for inference")
    parser.add_argument(
        "--max-batches",
        type=int,
        default=None,
        help="Maximum number of batches to process (None for full dataset)",
    )
    parser.add_argument(
        "--output-dir", "-o", type=str, default="analysis_results", help="Output directory for results"
    )
    parser.add_argument(
        "--base-device",
        type=str,
        default="cuda:2",
        help="Device for base model (default: cuda:2)",
    )
    parser.add_argument(
        "--intervention-device",
        type=str,
        default="cuda:3",
        help="Device for intervention model (default: cuda:3)",
    )

    args = parser.parse_args()

    main(
        checkpoint_path=args.checkpoint,
        hf_model_name=args.model,
        tokenizer_model=args.tokenizer,
        batch_size=args.batch_size,
        max_batches=args.max_batches,
        output_dir=args.output_dir,
        base_device_str=args.base_device,
        intervention_device_str=args.intervention_device,
    )
