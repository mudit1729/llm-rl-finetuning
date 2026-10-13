# Phase 3: reverse-text smoke test (2026-10-12)

Instance 55551593: 2× RTX 3090 (Oregon), image nvidia/cuda:13.0.1-cudnn-devel-ubuntu24.04,
prime-rl 1e53181 (torch 2.13.0+cu130, flash-attn 2.8.3, vLLM 0.31.0). Job: `scripts/jobs/smoke.sh`.
Pipeline: base eval → SFT (our own) → export DCP→HF → RL from our SFT weights → export → eval.

| Stage | Reward | Notes |
|---|---|---|
| Base Qwen3-0.6B, eval 20×3 | **0.047** | README: ~0.05; 68% truncated |
| SFT online eval, step 0 / 25 / 50 / 75 / 100 | 0.052 / 0.013 / 0.437 / 0.631 / 0.656 | 100 steps, ~20 s/step on a 3090 |
| RL train reward, step 1 → 20 | 0.073 → 0.825 | 128 rollouts/step, ~30 s/step |
| RL online eval (256 ex., 128 tok), v0 / v5 / v10 / v15 / v20 | 0.161 / 0.611 / 0.751 / 0.773 / 0.786 | |
| **Final RL model, same eval as base** | **0.813** | README: ~0.8; 3% truncated |

RL train reward by step: 0.073 0.137 0.124 0.224 0.429 0.500 0.580 0.654 0.653 0.722 0.692 0.681 0.694 0.746 0.769 0.721 0.792 0.781 0.817 0.825

Time: 1.30 h.
Setup took 6 min (`uv sync` 2.5 min). SFT took 36 min; RL plus exports and evals took 18 min.
About 10 min were lost to idle time while I was blocked locally, and about 5 min to the two fixes below.

Issues fixed
1. `uv run` re-resolves the project on each call and refetched flash-attn from GitHub, which was
   returning 504s. All runs now set `UV_NO_SYNC=1` (setup does the one sync).
2. The bare CUDA image lacks `python3.12-dev`, and Triton JIT-compiles a C helper at startup (Python.h).
   It is now installed in vast_setup.sh.

Verified tooling: create → setup in tmux → jobs in tmux with logs → dashboard over SSH (port 7788) →
idle watchdog (counted idle minutes, went quiet during jobs) → pull (30 MB, no checkpoints) →
destroy, confirmed by `vastai show instances`.
Artifacts: `artifacts/wordle-rl-smoke-55551593/` (gitignored).
