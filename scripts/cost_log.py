"""Keep COST.md (local, gitignored) current and enforce the budget cap.

    python3 scripts/cost_log.py total                      # spent so far
    python3 scripts/cost_log.py check --planned 3.20       # exit 1 if spent + planned > cap
    python3 scripts/cost_log.py add --phase 3 --instance 123 --gpu "1x RTX 4090" \
        --dph 0.41 --hours 0.62 --notes "reverse-text smoke"

Rows are estimates (runtime x $/hr at destroy time); reconcile against
`vastai show invoices -c` and edit the row if they differ.
"""

import argparse
import re
import sys
from datetime import date
from pathlib import Path

COST_MD = Path(__file__).resolve().parent.parent / "COST.md"
ROW = re.compile(r"^\|\s*\d{4}-\d{2}-\d{2}\s*\|")


TEMPLATE = """# vast.ai cost log (local only; gitignored)

Budget cap: **${cap:.2f}** (stop and ask before exceeding).

| Date | Phase | Instance | GPU | $/hr | Hours | Cost | Notes |
|---|---|---|---|---|---|---|---|

**Total spent: $0.00** | Remaining: ${cap:.2f}
"""


def read():
    if not COST_MD.exists():  # fresh clone: start a local log; cap from $VAST_BUDGET_USD (default 40)
        import os
        COST_MD.write_text(TEMPLATE.replace("{cap:.2f}", f"{float(os.environ.get('VAST_BUDGET_USD', 40)):.2f}"))
    text = COST_MD.read_text()
    cap = float(re.search(r"Budget cap: \*\*\$([\d.]+)\*\*", text).group(1))
    rows = [line for line in text.splitlines() if ROW.match(line)]
    spent = sum(float(line.split("|")[7].strip().lstrip("$")) for line in rows)
    return text, cap, rows, spent


def write(text: str, cap: float, rows: list[str], spent: float) -> None:
    lines = [line for line in text.splitlines() if not ROW.match(line) and not line.startswith("**Total spent")]
    header_end = max(i for i, line in enumerate(lines) if line.startswith("|---"))
    lines[header_end + 1:header_end + 1] = rows
    while lines and not lines[-1].strip():
        lines.pop()
    lines += ["", f"**Total spent: ${spent:.2f}** | Remaining: ${cap - spent:.2f}", ""]
    COST_MD.write_text("\n".join(lines))


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("total")
    c = sub.add_parser("check")
    c.add_argument("--planned", type=float, required=True)
    a = sub.add_parser("add")
    for f in ("phase", "instance", "gpu", "notes"):
        a.add_argument(f"--{f}", default="")
    a.add_argument("--dph", type=float, required=True)
    a.add_argument("--hours", type=float, required=True)
    args = ap.parse_args()

    text, cap, rows, spent = read()
    if args.cmd == "total":
        print(f"spent ${spent:.2f} of ${cap:.2f} cap (remaining ${cap - spent:.2f})")
    elif args.cmd == "check":
        total = spent + args.planned
        print(f"spent ${spent:.2f} + planned ${args.planned:.2f} = ${total:.2f} (cap ${cap:.2f})")
        sys.exit(0 if total <= cap else 1)
    else:
        cost = args.dph * args.hours
        rows.append(f"| {date.today()} | {args.phase} | {args.instance} | {args.gpu} | ${args.dph:.3f} "
                    f"| {args.hours:.2f} | ${cost:.2f} | {args.notes} |")
        write(text, cap, rows, spent + cost)
        print(f"logged ${cost:.2f}; total now ${spent + cost:.2f} of ${cap:.2f}")


if __name__ == "__main__":
    main()
