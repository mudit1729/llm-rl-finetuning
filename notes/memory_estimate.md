# Phase 4 memory estimate (Qwen3-1.7B, 1 trainer GPU + 1 inference GPU)

Model: 28 layers, hidden 2048, MLP 6144, 16 Q / 8 KV heads × 128, vocab 151,936, tied
embeddings, ~1.72B params. RL config: seq_len 8192 (packed micro-batches), batch 128 rollouts.

## Trainer GPU, full fine-tune (FSDP on 1 GPU, AdamW)

| Item | Size |
|---|---|
| fp32 master weights | 6.9 GB |
| fp32 gradients | 6.9 GB |
| AdamW m + v (fp32) | 13.8 GB |
| **Model + optimizer state** | **~27.6 GB** |
| Activations, 8192-token micro-batch, no activation checkpointing (≈34·h bytes/token/layer with flash-attn: 8192 × 28 × 70 KB) | ~16 GB |
| Logits + loss buffers (8192 × 151,936; bf16 + fp32) | ~5–8 GB |
| **Peak** | **~50 GB** |

- **80 GB card: fits with ~30 GB headroom**, no activation checkpointing needed. This is the example's default trainer config.
- 48 GB card: needs activation checkpointing (`[trainer.model.ac]`), which cuts activations to ~2–3 GB, for a peak of ~35 GB.
- 24 GB card: does not fit (the optimizer state alone exceeds it).

## Trainer GPU, LoRA (rank 32 on all linear layers)

Frozen bf16 base 3.4 GB + ~35M adapter params with fp32 states (~0.6 GB) + the same activations and logits (~21–24 GB)
→ **~25–28 GB peak**. Fits on 48 GB, or on 24 GB with activation checkpointing.

## Inference GPU (vLLM, bf16)

Weights 3.4 GB. KV cache per token = 2 × 28 layers × 8 KV heads × 128 × 2 bytes ≈ 115 KB.
With ~65 GB free for cache on an 80 GB card, that is ~570k tokens. A full batch of 128 games at
≤ ~4.5k tokens each is ~576k tokens, so roughly all rollouts of a step can be in flight. A 48 GB card would hold about half.

## Checkpoint disk

Full-FT DCP checkpoint (weights + optimizer state) ≈ 28 GB. `keep_last = 1` plus the HF export
(3.4 GB per model) fits in the 100 GB instance disk alongside ~25 GB of uv cache and venv.

**Recommendation: 2× A100 80 GB** (or H100 80 GB). Both runs fit without activation checkpointing,
and the inference GPU has KV room for the whole batch.
