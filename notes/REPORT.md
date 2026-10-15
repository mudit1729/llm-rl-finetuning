# Qwen3-1.7B learns Wordle with prime-rl: report

**Bottom line.** Training worked, and then hit a ceiling. On 200 games over 18 secret words the model
never saw in training, plain SFT + RL raised the win rate from **0% to 15% (full fine-tune) and 21% (LoRA)**
on prime-rl's example eval, or **40.5% and 42.5%** when the eval strips the model's reasoning the way
training does. Structured-thinking SFT plus reweighted per-turn RL (run 3) reached **45.5%** on the example
eval, and **41.0%** in a rerun: level with the earlier models like for like. Late in training the run 3
model wins about 88% of games on its 138 training words, so the ceiling now is memorizing those words,
not validity or infrastructure. Turns-to-solve did not go down.

## Results (200 held-out games: 20 words × 10 tries, T = 1.0, 1,024 tokens per turn)

| Model | Win rate, example eval | Win rate, reasoning stripped | Avg reward (example / stripped) | Guesses to solve | Format errors / turn (example) | Training GPU-h |
|---|---|---|---|---|---|---|
| Qwen3-1.7B base | 0.0% | 0.0% | 0.17 / 0.21 | no wins | 61.9% | 0 |
| + SFT warmup (20 steps) | 2.5% (0–6) | 12.5% (8–18) | 0.30 / 0.40 | 4.08 | 13.9% | 0.16 |
| + RL, full fine-tune (200 steps) | 15.0% (10–20) | 40.5% (28–53) | 0.43 / 0.66 | 4.27 | 9.4% | 7.44 |
| + RL, LoRA r32 (200 steps) | 21.0% (12–31) | **42.5%** (30–55) | 0.47 / 0.65 | 4.16 | 11.0% | 9.56 |
| + RL, per-turn credit (run 2) | 17.0% (10–24) | 41.5% (28–56) | 0.45 / 0.64 | 4.53 | 2.4% | 4.94 |
| Structured-thinking SFT (run 3, 40 steps) | 11.5% (5–20) | 0.5%† | 0.42 / 0.24 | 4.57* | 4.1% | 0.31 |
| + RL, reweighted per-turn credit (run 3) | **45.5%** (30–61) | 2.5%† | 0.66 / 0.29 | 4.36* | 12.3% | 10.31 |
| + same RL, rerun on a second machine | 41.0% (28–56) | 5.5%† | 0.66 / 0.33 | 4.29* | 6.6% | 4.80 |

95% intervals resample the 20 eval games. Guesses to solve: reasoning-stripped eval, except * (example
eval). Full tables: `results/results.md`. Plots: `results/plots/`.

† Not comparable. With the parser on, the eval client sends earlier turns back without their thinking.
The structured-thinking models were trained only on games that keep it, and without it they stop giving
an answer. Training and the example eval both keep it.

## Run 2: per-turn credit (the main bet from notes/IDEAS.md)

**Setup.** A new `wordle-turns` env scores every turn:
- information gain over the 156-word list;
- penalties for rejected moves, turns cut off at the token limit, thinking past 768 tokens, and guesses that contradict earlier clues;
- a win reward scaled by speed.

A small prime-rl patch (`turn_grpo`) gives each turn's tokens its own advantage: the GRPO outcome advantage plus discounted per-turn credit. The SFT recipe and the 200 full-fine-tune RL steps are otherwise the same as the baseline.

**Results** (thinking removed, 200 held-out games):

| | Baseline full RL | Per-turn credit |
|---|---|---|
| Win rate | 40.5% | 41.5% |
| Turns cut off at 1,024 tokens | 15% | **1.4%** |
| Format errors per turn | 22.5% | **9.1%** |
| Games lost to invalid moves | 88 | **39** |
| Games lost after all 6 guesses | 31 | 78 |
| Guesses consistent with earlier clues | 54% | 47% |
| Guesses that could still be the answer | 14% | 12% |

**Takeaway.** The cut-off and validity fixes worked. Run 2 also trained about 30% faster, because thinking got shorter. The win rate did not move: the games it rescued now run out of guesses, because the model uses its clues *worse*. The information-gain bonus pays for exploratory guesses that cannot be the answer, and the clue-violation penalty is too weak to offset it. Run 3 acted on this.

## Run 3: structured thinking

**Setup.**
- **SFT data** (`scripts/make_cot_data.py`): a solver plays 2,560 games on the 138 training words, guessing only common English words (one Qwen3 token each) and never an eval word. Each turn's thinking lists the clues, writes down what they imply (known letters, misplaced letters, ruled-out letters), spells the candidate out, then checks its length, that it is new, and every clue, in about 90 tokens instead of about 400. About half the turns first reject a near miss. The messages match the env exactly, and tests check that every written clue is true and no eval word appears.
- **RL:** run 2's env and `turn_grpo`, with the clue-violation penalty raised (0.2 → 0.5), the information-gain weight cut (0.5 → 0.1) and the turn weight cut (0.5 → 0.3). 200 full-fine-tune steps on the same seeds.

**Results** (each model's best eval: reasoning stripped for the earlier models, example eval for run 3):

| Held-out | Full RL | LoRA | Run 2 | Run 3 | Run 3 rerun |
|---|---|---|---|---|---|
| Win rate | 40.5% | 42.5% | 41.5% | **45.5%** | 41.0% |
| Games lost to two invalid moves | 44.0% | 43.5% | 19.5% | 19.5% | **14.0%** |
| Games lost after all 6 guesses | 15.5% | 14.0% | 39.0% | 35.0% | 45.0% |
| Later guesses that fit every earlier clue | 28.6% | 28.5% | 22.4% | **33.4%** | 24.9% |
| Turns cut off at 1,024 tokens | 15.2% | 16.6% | 1.4% | 6.4% | 2.1% |

Clue fit counts every guess after the first against all earlier feedback, the same way for all five; the run 2 table above used a different count.

**Takeaways.**
1. **The format fixed validity, not clue use.** The structured SFT model alone lost only 6.5% of games to invalid moves. But it marks every check ✓ while only 11% of its later guesses fit the clues; 57% follow a wrong clue summary. Matching each letter to its G/Y/X mark on another line and counting positions is hard for a 1.7B model that reads words as tokens.
2. **A first version of the data had no length or repeat checks,** and that SFT model lost 48% of games to invalid moves, mostly wrong-length words. Spelling each candidate out and checking length and repeats first cut it to 6.5%.
3. **RL lifted the example eval from 11.5% to 41–45.5%,** partly because the structured thinking never quotes a bracketed word, so no move leaks out of it. Like for like, the structured runs are level with the earlier models.
4. **It learns the 138 training words.** Over the rerun's last 10 steps, which had no failed games, it won 88% of training-word games against 41% held out. Training games go through prime-rl's rollout path rather than the eval harness, so the two aren't identical measurements, but the gap is far wider than the earlier models showed (item 2 below).
5. **Inference outages were not the main cause.** The first machine lost 12,217 training games during outages (prime-rl drops failed games, mostly long losing ones, which biases training). The rerun lost 7,073, mostly in steps 60–130, and scored the same within noise.

## Success criteria

- **Win rate up:** met. 0% → 15–21% (example eval) and 41–45.5% with structured thinking; 0% → 40–42% with reasoning stripped.
- **Average reward up:** met. 0.17 → 0.43–0.47, or 0.66 with structured thinking; 0.21 → 0.65–0.66 with reasoning stripped.
- **Turns-to-solve down:** not met. The base model never wins, so it has no baseline. From SFT to RL, valid
  guesses per win stayed flat (4.1–4.6 in every model).

## What we found

1. **The example's eval words were not held out.** Eval seeds 0–19 are also the first training seeds;
   all 156 secret words recur constantly; 98 of the 1,000 SFT games end on eval words. We held out the
   18 eval words everywhere (RL: 5,325 seeds over 138 words; SFT: 902 games), and verified it.
2. **The first RL models generalized; run 3 memorizes.** On 20 *training* words the first RL models score
   about the same as on held-out words (full: 15.5% vs 15.0%; LoRA: 11.0% vs 21.0%). Run 3 wins about 88%
   on training words late in training against 41–45.5% held out.
3. **The example eval and training feed the game different text.** In training, prime-rl's renderer
   splits off `<think>` and the game sees only the answer. In the eval, the vLLM server has no reasoning
   parser, so the game reads the first bracketed word anywhere in the reply. That is often a guess the
   model quoted while thinking ("the feedback for [crane]…"), so it gets rejected as a repeat. Adding
   `--vllm.reasoning-parser qwen3` to the eval server changes nothing else and moves full RL from 15% to
   40.5%. The base model's reward with it (0.21) matches prime-rl's published baseline (~0.2), which
   suggests the published numbers were measured this way.
4. **LoRA matched full fine-tuning.** The training reward tracked step for step, and the evals agree within
   their intervals. It used 9 GiB of trainer memory against 30 GiB, but each prime-rl step was slower (~80 s vs ~62 s).
5. **Where the first RL models failed:** with reasoning stripped, 18% of their turns run out of the
   1,024-token budget mid-thought and produce no move. Runs 2 and 3 cut that to 1–6%; the remaining
   losses are games that run out of guesses because most later guesses break a clue.
6. **Compared with prime-rl's ~60%:** ours is 40–45% on truly held-out words, with half the RL data
   (25.6k vs 51.2k games) and one inference GPU instead of six.

## How it ran

- **Code:** prime-rl `1e53181` (verifiers `2cb0a6d`), torch 2.13 cu130, vLLM 0.31.
- **Setup:** Mac for code; rented GPUs for training: 2× RTX 3090 for the smoke test (reverse-text: 0.047 →
  0.813, matching the README), 2× A100-SXM4-80GB for phase 4, run 2 and the run 3 rerun, and 2× A100 PCIe
  40GB for run 3 (the full-FT trainer peaks at ~30 GiB). One inference GPU and one trainer GPU, batch 128
  games (8 words × 16 tries), seq 8,192.
- **Fixes along the way:**
  - CUDA 13 image (not 12).
  - `UV_NO_SYNC=1` (GitHub 504s).
  - `python3.12-dev` for Triton.
  - A 16-worker env pool. TextArena takes 0.87 s of synchronous CPU to build a game, which stalled the default single worker; a step went from 3.5 min to ~70 s.
  - Deleting optimizer state after export to fit the 100 GB disk.
  - On GPUs without NVLink, the first weight sync hung: prime-rl turns NCCL P2P/SHM off per process after vLLM has already started NCCL with them on. Setting `NCCL_P2P_DISABLE=0 NCCL_SHM_DISABLE=0` before launch fixes it.
- **Not fixed:** both run 3 machines capped the container's CPU (20.5 and 30.7 CPUs) while showing 64–128. Every training game starts its own Python process, and run 3's short games finish fast enough to exhaust the cap; vLLM then missed router health checks and the router failed in-flight games. Lowering the game workers' priority helped for about 10 minutes.

| Phase | Hardware | Hours |
|---|---|---|
| 3 smoke test | 2× RTX 3090 | 1.30 |
| 4 Wordle (incl. 9 diagnostic evals) | 2× A100 SXM4 80GB | 10.26 |
| Run 2 · per-turn credit | 2× A100 SXM4 80GB | 2.85 |
| Run 3 · structured thinking (SFT twice, RL, 6 evals) | 2× A100 PCIe 40GB | 12.25 (about 6 idle after the job finished) |
| Run 3 rerun · SFT, RL, 2 evals | 2× A100 SXM4 80GB | 2.74 |

## Next steps

1. **Train on a wider pool of secret words,** so answers can't be learned: a custom taskset that draws secrets from thousands of common words, with the 18 eval words still excluded. This is the main bet now.
2. **Make clue checks lookups,** by pairing each letter with its mark in the thinking (`t:X r:X a:G`) and numbering the spelled letters.
3. **Fix the CPU load before renting again:** run games without a new Python process each, or with fewer env workers, and confirm in a 10-step trial.
4. **Evaluate on more held-out words.** 20 words give intervals of about ±15 points, too wide to resolve the differences runs 2–3 produce.
5. **Keep the exported SFT weights** (3.4 GB) when a later run may build on them.

Artifacts on the Mac: eval traces, metrics and logs in `artifacts/wordle-rl-wordle-*/` (phase 4:
`55623081`, run 2: `55849417`, run 3: `55959398`, rerun: `56052036`), and the phase 4 LoRA adapter in
`…55623081/outputs/wordle-rl-lora/lora_adapter_step_200/`. Full-model weights were not kept.
