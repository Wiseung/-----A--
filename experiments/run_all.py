from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description="Run bounded official evaluations over the selected cases.")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "2026_official" / "data")
    parser.add_argument("--official-root", type=Path, default=ROOT / "2026_official")
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--cases", nargs="*", help="case stems; default scans all official cases")
    parser.add_argument("--problems", type=int, nargs="+", choices=(1, 2, 3), default=[1, 2, 3])
    parser.add_argument("--ncores", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    parser.add_argument("--time-limit", type=float, default=300.0)
    parser.add_argument("--baseline-timeout", type=float, default=300.0)
    parser.add_argument("--max-evals", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no-improve", action="store_true")
    parser.add_argument("--retry-baseline", action="store_true")
    args = parser.parse_args()

    files = sorted(args.data_dir.glob("case_*.json"))
    if args.cases:
        selected = set(args.cases)
        files = [path for path in files if path.stem in selected]
        missing = selected - {path.stem for path in files}
        if missing:
            parser.error(f"unknown cases: {sorted(missing)}")
    if not files:
        parser.error(f"no official cases found in {args.data_dir}")

    summary_path = args.results_dir / "summary.json"
    completed = set()
    if args.resume and summary_path.is_file():
        previous = json.loads(summary_path.read_text(encoding="utf-8"))
        completed = {
            (row["case"], row["problem"], row["ncores"])
            for row in previous if row.get("status") == "evaluated"
        }

    total = len(files) * len(args.problems) * len(args.ncores)
    done = 0
    for graph in files:
        for problem in args.problems:
            for ncores in args.ncores:
                done += 1
                key = (graph.stem, problem, ncores)
                if key in completed:
                    print(f"[{done}/{total}] skip {graph.stem} p{problem} n{ncores}")
                    continue
                command = [
                    sys.executable, "-m", "solver.solver", str(graph.resolve()),
                    "--problem", str(problem), "--ncores", str(ncores),
                    "--time-limit", str(args.time_limit),
                    "--baseline-timeout", str(args.baseline_timeout),
                    "--max-evals", str(args.max_evals), "--seed", str(args.seed),
                    "--official-root", str(args.official_root.resolve()),
                    "--results-dir", str(args.results_dir.resolve()),
                ]
                if args.no_improve:
                    command.append("--no-improve")
                if args.retry_baseline:
                    command.append("--retry-baseline")
                print(f"[{done}/{total}] run {graph.stem} p{problem} n{ncores}")
                completed_run = subprocess.run(
                    command,
                    cwd=ROOT,
                    capture_output=True,
                    text=True,
                    check=False,
                )
                if completed_run.stdout:
                    print(completed_run.stdout, end="")
                if completed_run.returncode != 0:
                    print(completed_run.stderr, file=sys.stderr, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())