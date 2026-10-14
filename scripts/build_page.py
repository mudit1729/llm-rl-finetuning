"""Refresh the data block of notes/experiment.html (the experiment write-up page).

    ./scripts/vast_sync.sh pull            # optional: fresh metrics + eval traces from the instance
    uv run python scripts/build_page.py    # rewrites the JSON between the DATA markers

Everything shown as a number on the page comes from here: eval traces (summarize_eval),
RL metrics.jsonl (make_report.train_curve) and the held-out split.
"""

import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_report import MODELS, PARSED, eval_games, run_dir, run_hours, train_curve, win_ci  # noqa: E402
from summarize_eval import summarize  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
PAGE = ROOT / "notes" / "experiment.html"
BEGIN, END = "<!--DATA-BEGIN-->", "<!--DATA-END-->"
STEPS = 200


def artifacts_dir() -> Path | None:
    found = sorted((ROOT / "artifacts").glob("wordle-rl-wordle-*"), key=lambda p: p.stat().st_mtime)
    return found[-1] if found else None


def main() -> None:
    art = artifacts_dir()
    out = art / "outputs" if art else None
    evals = []
    for label, run, train_runs in MODELS:
        games = eval_games(run_dir(art, run)) if art and run_dir(art, run).exists() else []
        row = {"model": label, "run": run, "games": len(games)}
        if games:
            s = summarize(games)
            lo, hi = win_ci(games)
            row.update({k: s[k] for k in ("win_rate", "avg_reward", "avg_turns", "turns_to_solve", "guesses_to_solve",
                                          "format_error_rate", "truncated_call_rate", "invalid_end_rate")})
            row.update({"ci": [lo, hi]})
            for k in ("turns_to_solve", "guesses_to_solve"):
                if row[k] != row[k]:  # NaN -> null
                    row[k] = None
        for key, other in (("parsed", PARSED.get(run)), ("seen", run + "-seen")):
            og = eval_games(run_dir(art, other)) if art and other and run_dir(art, other).exists() else []
            if og:
                o = summarize(og)
                row[key] = {k: (None if o[k] != o[k] else o[k]) for k in
                            ("win_rate", "avg_reward", "avg_turns", "guesses_to_solve", "format_error_rate", "truncated_call_rate")}
                row[key]["ci"] = list(win_ci(og))
        evals.append(row)

    rl = {}
    for key, run in (("full", "wordle-rl"), ("lora", "wordle-rl-lora"), ("turns", "wordle-rl-turns")):
        d = run_dir(art, run) if art else None
        pts = train_curve(d) if d and d.exists() else []
        rl[key] = {"curve": [[s, round(v, 4)] for s, v in pts], "steps": STEPS,
                   "hours": round(run_hours(d) or 0, 2) if pts else None}

    split = json.loads((ROOT / "configs" / "heldout_split.json").read_text())
    data = {
        "updated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "evals": evals,
        "rl": rl,
        "words": {"eval": split["held_out_words"], "train": split["train_words"],
                  "excluded_seeds": len(split["excluded_train_seeds"]),
                  "train_seeds": split["train_range"][1] - split["train_range"][0] - len(split["excluded_train_seeds"])},
    }
    html = PAGE.read_text()
    a, b = html.index(BEGIN) + len(BEGIN), html.index(END)
    block = '\n<script type="application/json" id="data">' + json.dumps(data, separators=(",", ":")) + "</script>\n"
    PAGE.write_text(html[:a] + block + html[b:])
    full = rl["full"]["curve"]
    print(f"page data updated: evals {[e['games'] for e in evals]}, full RL steps {len(full)}, "
          f"LoRA steps {len(rl['lora']['curve'])}")


if __name__ == "__main__":
    main()
