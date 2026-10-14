# Teaching Qwen3-1.7B to play Wordle

SFT warmup plus RL with [prime-rl](https://github.com/PrimeIntellect-ai/prime-rl), on the verifiers `wordle` taskset. Win rate is measured on 20 games whose secret words are never seen in training.

**Result:** 0% → **15%** (full fine-tune) and **21%** (LoRA) wins on prime-rl's example eval. Stripping the model's reasoning before the game reads the move, as training does, gives **40.5%** and **42.5%**.

![Win rate before vs after training](results/plots/win_rate.png)

## Results

200 held-out games (20 words × 10 tries), temperature 1.0, 1,024 tokens per turn. The 95% interval resamples the 20 games.

| Model | Win rate, example eval | Win rate, thinking removed | Avg reward (example / removed) |
|---|---|---|---|
| Qwen3-1.7B base | 0% | 0% | 0.17 / 0.21 |
| + SFT warmup | 2.5% | 12.5% | 0.30 / 0.40 |
| + RL, full fine-tune | 15.0% (10–20) | **40.5%** (28–53) | 0.43 / 0.66 |
| + RL, LoRA r32 | **21.0%** (12–31) | **42.5%** (30–55) | 0.47 / 0.65 |
| + RL, per-turn credit (run 2) | 17.0% (10–24) | 41.5% (28–56) | 0.45 / 0.64 |

![RL training reward](results/plots/train_reward.png)

Full tables: [results/results.md](results/results.md). One-page summary: [notes/REPORT.md](notes/REPORT.md).

## The task

- Each game is a seeded TextArena `Wordle-v0` board. The model reasons, then guesses a word in brackets (`[crane]`), and gets G/Y/X feedback for each letter.
- Six guesses. One invalid move (repeat, non-word, wrong length) gets a warning; a second ends the game.
- Reward is 1.0 for a win, otherwise `(greens + 0.5 × yellows) / 5` of the last valid guess.

## What we found

1. **The eval read moves out of the model's thinking.** The example eval server has no reasoning parser, so the game takes the first bracketed word anywhere in the reply, often a guess quoted mid-thought. Training strips the thinking. Adding `--vllm.reasoning-parser qwen3` to the eval server is the only change between the two result columns.
2. **The example's eval words weren't held out.** The secret words come from only 156 nouns, eval seeds 0–19 are also the first training seeds, and 98 of the 1,000 SFT games end on eval words. We held the 18 eval words out of both SFT and RL ([configs/heldout_split.json](configs/heldout_split.json)).
3. **No memorization.** RL models do no better on training words than on held-out words.
4. **LoRA matched full fine-tuning** step for step, using 9 GiB of trainer memory instead of 30.
5. **Main remaining failure:** about 1 turn in 5 runs out of the 1,024-token budget mid-thought.
6. **Turns-to-solve didn't drop** (about 4.1–4.5 guesses per win after SFT and after RL).
7. **Per-turn credit fixed validity, not wins.** Cut-off turns fell from 15% to 1.4% and invalid-move losses from 88 to 39 of 200, but the win rate held at about 41%: the model now runs out of guesses because its information-gain bonus favors exploratory guesses over consistent ones ([notes/REPORT.md](notes/REPORT.md)).

## Training schedule

| Stage | Data | Steps × batch | LR | Wall time |
|---|---|---|---|---|
| SFT warmup | 902 games (willcb/V3-wordle, eval words removed) | 20 × 64 | 1e-5 | 7.5 min |
| RL, full fine-tune | 5,325 held-out-safe seeds | 200 × 128 (8 words × 16 tries) | 1e-6 | 3.6 h |
| RL, LoRA r32 α64 | same | 200 × 128 | 1e-5 | 4.7 h |

Each RL step uses prime-rl's default GRPO advantages (reward minus the mean of the 16 tries at the same word) and the IPO loss. Context is 8,192 tokens per game.

## Architecture

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="results/plots/architecture-dark.png">
  <img alt="Architecture: the Mac launches jobs over SSH on a GPU machine; the orchestrator dispatches games to the env server, which plays them against vLLM on GPU 0; rollouts and advantages go to the trainer on GPU 1, which sends new weights to vLLM over NCCL and writes checkpoints; results are pulled back with rsync." src="results/plots/architecture-light.png">
</picture>

The env server runs each game and calls vLLM for every model turn; the orchestrator batches finished games, computes group advantages and feeds the trainer. Diagram made with [archify](https://github.com/tt-a1i/archify).

## What broke

| Problem | Fix |
|---|---|
| prime-rl at this commit needs CUDA 13 | `nvidia/cuda:13.0.1-cudnn-devel-ubuntu24.04`, hosts with CUDA ≥ 13 drivers |
| `uv run` refetched wheels from GitHub (504s) | `UV_NO_SYNC=1` after setup |
| Triton couldn't compile its helper | install `python3.12-dev` |
| One env worker: 0.87 s per new game stalled the rest | static pool of 16 workers (3.5 min → 70 s per step) |
| 21 GB checkpoints on a 100 GB disk | delete optimizer state after export |

## Reproduce

```bash
make setup                                   # Mac tooling (uv)
make split                                   # held-out split → configs
make gpu-offers PROFILE=wordle HOURS=9       # list 2-GPU offers; nothing is rented
make gpu-up OFFER=<id> HOURS=9 && make gpu-setup
make data eval-base sft eval-sft rl eval-rl rl-lora eval-lora
make gpu-down PHASE=4 && make report
```

`make help` lists every target. Pinned: prime-rl `1e53181`, verifiers `2cb0a6d`, torch 2.13+cu130, vLLM 0.31, textarena 0.7.4.
