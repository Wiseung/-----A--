from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from solver.run_identity import (
    build_fingerprints,
    ensure_run_metadata,
    make_run_metadata,
    new_run_id,
    valid_run_label,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run bounded official evaluations over the selected cases.")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "2026_official" / "data")
    parser.add_argument("--official-root", type=Path, default=ROOT / "2026_official")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--results-dir", type=Path)
    parser.add_argument("--history-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--cases", nargs="*", help="case stems; default scans all official cases")
    parser.add_argument("--problems", type=int, nargs="+", choices=(1, 2, 3), default=[1, 2, 3])
    parser.add_argument("--ncores", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    parser.add_argument("--time-limit", type=float, default=300.0)
    parser.add_argument("--baseline-timeout", type=float, default=300.0)
    parser.add_argument("--evaluator-timeout", type=float, default=600.0)
    parser.add_argument("--max-evals", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--experiment-id", default="round2")
    parser.add_argument("--run-id")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no-improve", action="store_true")
    parser.add_argument("--retry-baseline", action="store_true")
    args = parser.parse_args()

    if (args.time_limit <= 0 or args.baseline_timeout <= 0
            or args.evaluator_timeout <= 0 or args.max_evals < 1):
        parser.error("time limits must be positive; max-evals must be at least 1")
    if not valid_run_label(args.experiment_id):
        parser.error("experiment-id must be a path-safe label")
    if args.resume and not args.run_id:
        parser.error("--resume requires --run-id from the original run")
    run_id = args.run_id or new_run_id()
    if not valid_run_label(run_id):
        parser.error("run-id must be a path-safe label")

    files = sorted(args.data_dir.glob("case_*.json"))
    if args.cases:
        selected = set(args.cases)
        files = [path for path in files if path.stem in selected]
        missing = selected - {path.stem for path in files}
        if missing:
            parser.error(f"unknown cases: {sorted(missing)}")
    if not files:
        parser.error(f"no official cases found in {args.data_dir}")

    data_dir = args.data_dir.resolve()
    official_root = args.official_root.resolve()
    config_path = args.config.resolve() if args.config else official_root / "data" / "config.txt"
    history_dir = args.history_dir.resolve()
    results_dir = (
        args.results_dir.resolve()
        if args.results_dir
        else ROOT / "results_round2" / args.experiment_id / run_id
    )
    fingerprints_by_case = {
        graph.stem: build_fingerprints(
            graph.resolve(), config_path, official_root, ROOT / "solver"
        )
        for graph in files
    }
    common_fingerprints = next(iter(fingerprints_by_case.values()))
    run_metadata = make_run_metadata(
        args.experiment_id,
        run_id,
        common_fingerprints,
        {
            "seed": args.seed,
            "time_limit_sec": args.time_limit,
            "baseline_timeout_sec": args.baseline_timeout,
            "evaluator_timeout_sec": args.evaluator_timeout,
            "max_evals": args.max_evals,
            "improve_enabled": not args.no_improve,
            "retry_baseline": args.retry_baseline,
            "history_dir": str(history_dir),
        },
    )
    batch_manifest = {
        "cases": [graph.stem for graph in files],
        "problems": args.problems,
        "ncores": args.ncores,
        "data_dir": data_dir.relative_to(ROOT).as_posix()
        if data_dir.is_relative_to(ROOT)
        else data_dir.as_posix(),
    }
    metadata_path = results_dir / "run_metadata.json"
    manifest_path = results_dir / "batch_manifest.json"
    if args.resume:
        if not metadata_path.is_file() or not manifest_path.is_file():
            parser.error("cannot resume: run metadata or batch manifest is missing")
        try:
            ensure_run_metadata(metadata_path, run_metadata)
            existing_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, ValueError) as error:
            parser.error(str(error))
        if existing_manifest != batch_manifest:
            parser.error("resume matrix does not match the original batch manifest")
    else:
        if metadata_path.exists() or manifest_path.exists():
            parser.error("run directory already has metadata; use --resume with the same --run-id")
        ensure_run_metadata(metadata_path, run_metadata)
        results_dir.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(
            json.dumps(batch_manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    summary_path = results_dir / "summary.json"
    completed = set()
    if args.resume and summary_path.is_file():
        previous = json.loads(summary_path.read_text(encoding="utf-8"))
        for row in previous:
            fingerprints = fingerprints_by_case.get(row.get("case"), {})
            if (row.get("status") == "evaluated"
                    and row.get("experiment_id") == args.experiment_id
                    and row.get("run_id") == run_id
                    and row.get("graph_hash") == fingerprints.get("graph_hash")
                    and row.get("config_hash") == fingerprints.get("config_hash")
                    and row.get("official_code_hash") == fingerprints.get("official_code_hash")
                    and row.get("solver_code_hash") == fingerprints.get("solver_code_hash")
                    and row.get("seed") == args.seed
                    and row.get("time_limit_sec") == args.time_limit
                    and row.get("baseline_timeout_sec") == args.baseline_timeout
                    and row.get("evaluator_timeout_sec") == args.evaluator_timeout
                    and row.get("max_evals") == args.max_evals
                    and row.get("improve_enabled") == (not args.no_improve)
                    and row.get("retry_baseline") == args.retry_baseline):
                completed.add((row["case"], row["problem"], row["ncores"]))

    total = len(files) * len(args.problems) * len(args.ncores)
    done = 0
    print(f"experiment_id={args.experiment_id} run_id={run_id} results_dir={results_dir}")
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
                    "--evaluator-timeout", str(args.evaluator_timeout),
                    "--max-evals", str(args.max_evals), "--seed", str(args.seed),
                    "--official-root", str(official_root),
                    "--config", str(config_path),
                    "--results-dir", str(results_dir),
                    "--history-dir", str(history_dir),
                    "--experiment-id", args.experiment_id,
                    "--run-id", run_id,
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
                    failure_path = results_dir / "process_failures.jsonl"
                    failure = {
                        "experiment_id": args.experiment_id,
                        "run_id": run_id,
                        "case": graph.stem,
                        "problem": problem,
                        "ncores": ncores,
                        "graph_hash": fingerprints_by_case[graph.stem]["graph_hash"],
                        "return_code": completed_run.returncode,
                        "stderr": completed_run.stderr,
                        "status": "process_failed",
                    }
                    with failure_path.open("a", encoding="utf-8") as stream:
                        stream.write(json.dumps(failure, ensure_ascii=False) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())