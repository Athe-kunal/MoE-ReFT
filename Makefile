.PHONY: train

train:
	export PYTHONPATH=. ; \
	export CUDA_VISIBLE_DEVICES=2,3 ; \
	uv run torchrun --nnodes 1 --nproc_per_node 2 moe_reft/train.py
