"""Phase 5: results table and plots from the pulled Phase 4 artifacts.

    uv run python scripts/make_report.py [artifacts/wordle-rl-wordle-<id>]

Writes
  results/results.md            model | win rate | avg reward | avg turns | format-error rate | GPU-hours
  results/eval_summaries.json   every metric from scripts/summarize_eval.py, per model
  results/plots/train_reward.png   RL training reward per step (full fine-tune vs LoRA)
  results/plots/win_rate.png       held-out win rate per model, 95% CI (bootstrap over the 20 games)

GPU-hours are the training compute that produced each model (cumulative: SFT for the SFT
model, SFT + RL for the RL models), from each run's metrics.jsonl timestamps x 2 GPUs.
"""

import json
import random
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from summarize_eval import find_trace_files, game_stats, read_lines, summarize  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results"
PLOTS = OUT / "plots"

MODELS = [  # (label, eval run, training runs that produced it)
    ("Qwen3-1.7B (base)", "wordle-eval-base", []),
    ("+ SFT warmup", "wordle-eval-sft", ["wordle-sft"]),
    ("+ SFT + RL (full)", "wordle-eval-rl", ["wordle-sft", "wordle-rl"]),
    ("+ SFT + RL (LoRA r32)", "wordle-eval-lora", ["wordle-sft", "wordle-rl-lora"]),
    ("+ SFT + RL (per-turn credit)", "wordle-eval-turns", ["wordle-sft", "wordle-rl-turns"]),
]
SEEN = [  # diagnostic: the same eval on 20 training games (configs/wordle/eval_seen.toml)
    ("+ SFT + RL (full)", "wordle-eval-rl", "wordle-eval-rl-seen"),
    ("+ SFT + RL (LoRA r32)", "wordle-eval-lora", "wordle-eval-lora-seen"),
]
PARSED = {  # diagnostic: eval.toml with vLLM's qwen3 reasoning parser on the server
    "wordle-eval-base": "wordle-eval-base-parsed", "wordle-eval-sft": "wordle-eval-sft-parsed",
    "wordle-eval-rl": "wordle-eval-rl-parsed", "wordle-eval-lora": "wordle-eval-lora-parsed",
    "wordle-eval-turns": "wordle-eval-turns-parsed",
}
TRAIN_KEY = re.compile(r"^train/agg/all/[^/]+/reward/mean$")

# Reference palette, light mode (dataviz skill references/palette.md)
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
SERIES = {"wordle-rl": ("#2a78d6", "full fine-tune"), "wordle-rl-lora": ("#eb6834", "LoRA r32"),
          "wordle-rl-turns": ("#1baf7a", "per-turn credit")}
# Run 2 scores games differently (outcome = win x speed), so its curve uses the game's own reward.
GAME_REWARD_KEY = re.compile(r"^train/agg/all/[^/]+/metrics/game_reward/mean$")


def latest_artifacts() -> Path:
    found = sorted((ROOT / "artifacts").glob("wordle-rl-wordle-*"), key=lambda p: p.stat().st_mtime)
    if not found:
        raise SystemExit("no artifacts/wordle-rl-wordle-* (run: make vast-pull)")
    return found[-1]


def run_dir(art: Path, run: str) -> Path:
    """The newest outputs/<run> across all instance artifact folders (runs span instances)."""
    found = sorted((ROOT / "artifacts").glob(f"wordle-rl-wordle-*/outputs/{run}"), key=lambda p: p.stat().st_mtime)
    return found[-1] if found else art / "outputs" / run


def metrics_rows(run_dir: Path) -> list[dict]:
    f = run_dir / "monitors" / "file" / "metrics.jsonl"
    return [json.loads(line) for line in f.read_text().splitlines() if line.strip()] if f.exists() else []


def run_hours(run_dir: Path) -> float | None:
    times = [r["time"] for r in metrics_rows(run_dir) if isinstance(r.get("time"), (int, float))]
    return (max(times) - min(times)) / 3600 if len(times) > 1 else None


def train_curve(run_dir: Path) -> list[tuple[int, float]]:
    rows = metrics_rows(run_dir)
    key = GAME_REWARD_KEY if any(GAME_REWARD_KEY.match(k) for r in rows for k in r) else TRAIN_KEY
    pts = {}
    for r in rows:
        for k, v in r.items():
            if key.match(k) and isinstance(v, (int, float)):
                pts[r["step"]] = v
    return sorted(pts.items())


def eval_games(run_dir: Path) -> list[dict]:
    games = []
    for f in find_trace_files([str(run_dir)], all_runs=False):
        for line in read_lines(f):
            if line.strip():
                games.extend(game_stats(t) for t in json.loads(line).get("traces", []))
    return games


def win_ci(games: list[dict], n_boot: int = 4000) -> tuple[float, float]:
    """95% CI for win rate, bootstrapping over games (seeds): the 10 attempts at one word move together."""
    by_seed: dict = {}
    for g in games:
        by_seed.setdefault(g["seed"], []).append(g["win"])
    seeds, rng, rates = list(by_seed), random.Random(0), []
    for _ in range(n_boot):
        sample = [w for s in rng.choices(seeds, k=len(seeds)) for w in by_seed[s]]
        rates.append(sum(sample) / len(sample))
    rates.sort()
    return rates[int(0.025 * n_boot)], rates[int(0.975 * n_boot) - 1]


def style(ax) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def plot_train(art: Path) -> Path | None:
    curves = {run: train_curve(run_dir(art, run)) for run in SERIES}
    curves = {k: v for k, v in curves.items() if v}
    if not curves:
        return None
    fig, ax = plt.subplots(figsize=(7.5, 4.2), facecolor=SURFACE)
    style(ax)
    ends = {}
    for run, pts in curves.items():
        color, label = SERIES[run]
        steps, vals = zip(*pts)
        w = 10  # trailing mean, so the trend reads through step-to-step noise
        smooth = [sum(vals[max(0, i - w + 1):i + 1]) / len(vals[max(0, i - w + 1):i + 1]) for i in range(len(vals))]
        ax.plot(steps, vals, color=color, linewidth=1, alpha=0.3)
        ax.plot(steps, smooth, color=color, linewidth=2, label=f"{label} (10-step mean)")
        ends[run] = (steps[-1], smooth[-1], label)
    # End labels, spread so none overlap: at least 12 points apart, lowest first.
    order = sorted(ends.values(), key=lambda e: e[1])
    to_px = ax.transData.transform
    placed = []
    for x_end, y_end, label in order:
        y_px = to_px((x_end, y_end))[1]
        if placed and y_px < placed[-1] + 12:
            y_px = placed[-1] + 12
        placed.append(y_px)
        dy = y_px - to_px((x_end, y_end))[1]
        ax.annotate(f"{label}  {y_end:.2f}", (x_end, y_end), xytext=(6, dy * 72 / fig.dpi), textcoords="offset points",
                    va="center", fontsize=9, color=INK)
    ax.set_xlabel("RL step (128 games per step)", color=INK2, fontsize=9)
    ax.set_ylabel("mean training reward (game's own scale)", color=INK2, fontsize=9)
    ax.set_title("Wordle RL training reward, Qwen3-1.7B from our SFT warmup", color=INK, fontsize=11, loc="left")
    ax.legend(frameon=False, fontsize=9, labelcolor=INK2, loc="lower right")
    ax.margins(x=0.02)
    fig.subplots_adjust(right=0.8)
    path = PLOTS / "train_reward.png"
    fig.savefig(path, dpi=160, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_win(rows: list[dict]) -> Path:
    rows = [r for r in rows if r["games"]]
    fig, ax = plt.subplots(figsize=(7.5, 4.2), facecolor=SURFACE)
    style(ax)
    groups = [("example eval: raw text to the game", "#2a78d6", lambda r: r)]
    if any(r.get("parsed") for r in rows):
        groups.append(("reasoning stripped (as in training)", "#eb6834", lambda r: r.get("parsed")))
    width = 0.7 / len(groups)
    tops = [0.0]
    for g, (label, color, pick) in enumerate(groups):
        for i, r in enumerate(rows):
            d = pick(r)
            if not d:
                continue
            xi = i + (g - (len(groups) - 1) / 2) * width
            rate = 100 * d["win_rate"]
            lo, hi = 100 * (d["win_rate"] - d["ci"][0]), 100 * (d["ci"][1] - d["win_rate"])
            tops.append(rate + hi)
            ax.bar(xi, rate, width=width, color=color, edgecolor=SURFACE, linewidth=2, label=label if i == 0 else None)
            ax.errorbar(xi, rate, yerr=[[lo], [hi]], fmt="none", ecolor=INK2, elinewidth=1, capsize=3)
            ax.annotate(f"{rate:.1f}%", (xi, rate + hi), xytext=(0, 3),
                        textcoords="offset points", ha="center", fontsize=9, color=INK)
    if len(groups) > 1:
        ax.legend(frameon=False, fontsize=8.5, labelcolor=INK2, loc="upper right")
    x = range(len(rows))
    ax.axhline(60, color=INK2, linewidth=1, linestyle=(0, (4, 3)))
    ax.annotate("prime-rl example, eval words not held out: ~60%", (-0.45, 60), xytext=(0, 4), annotation_clip=False,
                textcoords="offset points", ha="left", fontsize=8.5, color=INK2)
    short = {"Qwen3-1.7B (base)": "Base", "+ SFT warmup": "SFT", "+ SFT + RL (full)": "SFT + RL\nfull",
             "+ SFT + RL (LoRA r32)": "SFT + RL\nLoRA r32", "+ SFT + RL (per-turn credit)": "SFT + RL\nper-turn credit"}
    ax.set_xticks(list(x), [short.get(r["model"], r["model"]) for r in rows], fontsize=9, color=INK)
    ax.set_ylabel("win rate on 200 held-out games (%)", color=INK2, fontsize=9)
    ax.set_ylim(0, max(100, max(tops) + 15))
    ax.set_title("Wordle win rate before vs after training (95% CI over the 20 held-out games)",
                 color=INK, fontsize=11, loc="left")
    path = PLOTS / "win_rate.png"
    fig.savefig(path, dpi=160, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    return path


def main() -> None:
    art = Path(sys.argv[1]) if len(sys.argv) > 1 else latest_artifacts()
    PLOTS.mkdir(parents=True, exist_ok=True)
    inst = json.loads((ROOT / ".vast" / "history" / f"{art.name.rsplit('-', 1)[1]}.json").read_text()) \
        if (ROOT / ".vast" / "history" / f"{art.name.rsplit('-', 1)[1]}.json").exists() else None
    if inst is None and (ROOT / ".vast" / "instance.json").exists():
        inst = json.loads((ROOT / ".vast" / "instance.json").read_text())
    n_gpus = int(inst["gpu"].split("x")[0]) if inst else 2

    hours = {run: run_hours(run_dir(art, run)) for run in ("wordle-sft", "wordle-rl", "wordle-rl-lora", "wordle-rl-turns")}
    rows = []
    for label, eval_run, train_runs in MODELS:
        games = eval_games(run_dir(art, eval_run)) if run_dir(art, eval_run).exists() else []
        s = summarize(games) if games else {}
        h = sum(hours.get(r) or 0 for r in train_runs)
        pg = eval_games(run_dir(art, PARSED[eval_run])) if run_dir(art, PARSED[eval_run]).exists() else []
        parsed = {**summarize(pg), "ci": win_ci(pg)} if pg else None
        rows.append({"model": label, "eval_run": eval_run, "games": len(games), **s, "parsed": parsed,
                     "ci": win_ci(games) if games else None,
                     "train_hours": h, "gpu_hours": h * n_gpus})

    def pct(v):
        return "—" if v is None else f"{100 * v:.1f}%"

    def num(v, nd=2):
        return "—" if v is None or v != v else f"{v:.{nd}f}"

    lines = [
        "# Results: Qwen3-1.7B on Wordle (prime-rl)",
        "",
        "Eval: `configs/wordle/eval.toml` for every model: 20 held-out games (seeds 0-19, 18 secret words",
        "never seen in SFT or RL) × 10 rollouts = 200 games, temperature 1.0, 1024 tokens per turn.",
        "",
        "| Model | Win rate (95% CI) | Avg reward | Avg turns | Guesses to solve | Format-error rate | GPU-hours |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        if not r["games"]:
            lines.append(f"| {r['model']} | not run | | | | | |")
            continue
        ci = f" ({100 * r['ci'][0]:.0f}–{100 * r['ci'][1]:.0f}%)"
        lines.append(f"| {r['model']} | {pct(r['win_rate'])}{ci} | {num(r['avg_reward'])} | {num(r['avg_turns'])} "
                     f"| {num(r['guesses_to_solve']) if r['win_rate'] else 'no wins'} | {pct(r['format_error_rate'])} | {num(r['gpu_hours'])} |")
    seen_rows = []
    for label, held_run, seen_run in SEEN:
        held = next((r for r in rows if r["eval_run"] == held_run and r["games"]), None)
        sg = eval_games(run_dir(art, seen_run)) if run_dir(art, seen_run).exists() else []
        if held and sg:
            ss = summarize(sg)
            seen_rows.append(f"| {label} | {pct(held['win_rate'])} | {pct(ss['win_rate'])} | {num(held['avg_reward'])} "
                             f"| {num(ss['avg_reward'])} |")
    prow = [r for r in rows if r.get("parsed")]
    if prow:
        lines += ["", "### Same eval with reasoning stripped (diagnostic)", "",
                  "`configs/wordle/eval.toml` unchanged; the vLLM server adds `--vllm.reasoning-parser qwen3`, so the game",
                  "sees only the answer after `</think>`, as in training (prime-rl's renderer splits it there). The example",
                  "eval passes the raw text, and the game plays the first bracketed word, often a guess quoted in the thinking.", "",
                  "| Model | Win rate (95% CI) | Avg reward | Avg turns | Guesses to solve | Format-error rate |",
                  "|---|---|---|---|---|---|"]
        for r in prow:
            d = r["parsed"]
            lines.append(f"| {r['model']} | {pct(d['win_rate'])} ({100 * d['ci'][0]:.0f}–{100 * d['ci'][1]:.0f}%) | "
                         f"{num(d['avg_reward'])} | {num(d['avg_turns'])} | "
                         f"{num(d['guesses_to_solve']) if d['win_rate'] else 'no wins'} | {pct(d['format_error_rate'])} |")
    if seen_rows:
        lines += ["", "### Held-out vs training words (diagnostic)", "",
                  "Same eval settings on 20 training games (`configs/wordle/eval_seen.toml`), 10 tries each.", "",
                  "| Model | Win rate, held-out | Win rate, training words | Reward, held-out | Reward, training words |",
                  "|---|---|---|---|---|"] + seen_rows
    lines += [
        "",
        "- **Win rate**: games solved within 6 guesses. **Avg reward**: TextArena game reward (1.0 win; otherwise",
        "  (greens + 0.5 × yellows) / 5 of the last valid guess). **Avg turns**: model turns per game, invalid",
        "  attempts included. **Guesses to solve**: valid guesses in won games (1-6), the usual Wordle score.",
        "- **Format-error rate**: share of model turns that did not produce a well-formed move: cut off at the",
        "  1024-token limit, or rejected for format (no `[word]`, or not 5 letters). A cut-off `<think>` has no",
        "  `</think>`, so vLLM returns the whole thought as the answer and the game reads the first bracketed word",
        "  in it (usually a quoted earlier guess) as a repeat; those turns count as format errors here.",
        f"- **GPU-hours**: training compute behind each model, cumulative (SFT; SFT + RL), from run",
        f"  timestamps × {n_gpus} GPUs. Setup and eval time are not included.",
        "",
        "Plots: `plots/train_reward.png`, `plots/win_rate.png`.",
    ]
    (OUT / "results.md").write_text("\n".join(lines) + "\n")
    (OUT / "eval_summaries.json").write_text(json.dumps(rows, indent=1, default=str))
    print("\n".join(lines))
    for p in (plot_train(art), plot_win(rows)):
        if p:
            print(f"wrote {p.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
