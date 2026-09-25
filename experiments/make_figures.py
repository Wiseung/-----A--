from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
WIDTH, HEIGHT = 960, 600
LEFT, RIGHT, TOP, BOTTOM = 88, 34, 72, 82
COLORS = {"p1": "#176B87", "p2": "#D97706", "l2": "#4F772D", "cache": "#9A3412"}


def _svg_text(x: float, y: float, text: str, size: int = 14, anchor: str = "middle") -> str:
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}" '
        f'font-family="Arial,sans-serif" font-size="{size}" fill="#202A33">'
        f"{html.escape(text)}</text>"
    )


def _series(rows: list[dict[str, Any]], problem: int, label: str) -> dict[int, tuple[float | None, int]]:
    output = {}
    for ncores in range(1, 6):
        values = []
        if problem == 1 and ncores == 1:
            cases = {
                row["case"] for row in rows
                if row.get("case", "").startswith("case_")
                and row.get("single_core_baseline")
            }
            values = [1.0] * len(cases)
        else:
            for row in rows:
                if not row.get("case", "").startswith("case_"):
                    continue
                if row.get("problem") == problem and row.get("ncores") == ncores:
                    value = row.get("speedup")
                    if value is not None:
                        values.append(value)
        output[ncores] = ((sum(values) / len(values)) if values else None, len(values))
    return output


def _paired_problem3_series(
    rows: list[dict[str, Any]],
) -> tuple[
    dict[int, tuple[float | None, int]],
    dict[int, tuple[float | None, int]],
    dict[int, tuple[float | None, int]],
]:
    identity_fields = (
        "experiment_id", "run_id", "case", "ncores", "graph_hash",
        "config_hash", "official_code_hash", "solver_code_hash", "seed",
        "time_limit_sec", "baseline_timeout_sec", "evaluator_timeout_sec",
        "max_evals", "improve_enabled", "retry_baseline",
    )
    paired: dict[tuple[Any, ...], dict[int, dict[str, Any]]] = {}
    for row in rows:
        if row.get("problem") not in (2, 3):
            continue
        key = tuple(row.get(field) for field in identity_fields)
        paired.setdefault(key, {})[row["problem"]] = row

    no_l2_values = {ncores: [] for ncores in range(1, 6)}
    l2_values = {ncores: [] for ncores in range(1, 6)}
    cache_values = {ncores: [] for ncores in range(1, 6)}
    for pair in paired.values():
        no_l2 = pair.get(2)
        l2 = pair.get(3)
        if (no_l2 is None or l2 is None
                or no_l2.get("status") != "evaluated"
                or l2.get("status") != "evaluated"):
            continue
        ncores = no_l2["ncores"]
        no_l2_makespan = no_l2.get("makespan")
        l2_makespan = l2.get("makespan")
        baseline = no_l2.get("single_core_baseline")
        if baseline is None:
            baseline = l2.get("single_core_baseline")
        if no_l2_makespan not in (None, 0) and baseline not in (None, 0):
            no_l2_values[ncores].append(baseline / no_l2_makespan)
        if l2_makespan not in (None, 0) and baseline not in (None, 0):
            l2_values[ncores].append(baseline / l2_makespan)
        if no_l2_makespan not in (None, 0) and l2_makespan not in (None, 0):
            cache_values[ncores].append(no_l2_makespan / l2_makespan)

    def summarize(values: dict[int, list[float]]) -> dict[int, tuple[float | None, int]]:
        return {
            ncores: (
                sum(items) / len(items) if items else None,
                len(items),
            )
            for ncores, items in values.items()
        }

    return summarize(no_l2_values), summarize(l2_values), summarize(cache_values)


def line_svg(title: str, y_label: str, lines: list[tuple[str, dict[int, tuple[float | None, int]], str]], output: Path) -> None:
    values = [value for _, data, _ in lines for value, _ in data.values() if value is not None]
    ymin = min([0.0, *values]) if y_label == "Cache speedup" else 0.0
    ymax = max([1.0, *values])
    ymax = ymax * 1.12 if ymax > 1 else 1.1
    plot_w, plot_h = WIDTH - LEFT - RIGHT, HEIGHT - TOP - BOTTOM
    x_at = lambda n: LEFT + (n - 1) * plot_w / 4
    y_at = lambda value: TOP + (ymax - value) * plot_h / max(ymax - ymin, 1e-9)
    chunks = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" viewBox="0 0 {WIDTH} {HEIGHT}">',
        '<rect width="100%" height="100%" fill="white"/>',
        _svg_text(WIDTH / 2, 34, title, 21),
    ]
    for tick in range(6):
        value = ymin + (ymax - ymin) * tick / 5
        y = y_at(value)
        chunks.append(f'<line x1="{LEFT}" y1="{y:.1f}" x2="{WIDTH-RIGHT}" y2="{y:.1f}" stroke="#D8DEE4"/>')
        chunks.append(_svg_text(LEFT - 12, y + 5, f"{value:.2f}", 12, "end"))
    chunks.append(f'<line x1="{LEFT}" y1="{TOP}" x2="{LEFT}" y2="{HEIGHT-BOTTOM}" stroke="#35424C"/>')
    chunks.append(f'<line x1="{LEFT}" y1="{HEIGHT-BOTTOM}" x2="{WIDTH-RIGHT}" y2="{HEIGHT-BOTTOM}" stroke="#35424C"/>')
    for ncores in range(1, 6):
        x = x_at(ncores)
        chunks.append(f'<line x1="{x:.1f}" y1="{TOP}" x2="{x:.1f}" y2="{HEIGHT-BOTTOM}" stroke="#EEF1F4"/>')
        chunks.append(_svg_text(x, HEIGHT - BOTTOM + 24, str(ncores), 13))
    chunks.append(_svg_text(WIDTH / 2, HEIGHT - 26, "NPU cores", 14))
    chunks.append(
        f'<text x="24" y="{HEIGHT/2:.1f}" transform="rotate(-90 24 {HEIGHT/2:.1f})" '
        'text-anchor="middle" font-family="Arial,sans-serif" font-size="14" fill="#202A33">'
        f"{html.escape(y_label)}</text>"
    )

    legend_y = TOP + 14
    for series_index, (label, data, color) in enumerate(lines):
        last = None
        legend_x = LEFT + 20 + series_index * 210
        chunks.append(f'<line x1="{legend_x}" y1="{legend_y}" x2="{legend_x+26}" y2="{legend_y}" stroke="{color}" stroke-width="3"/>')
        chunks.append(_svg_text(legend_x + 34, legend_y + 5, label, 12, "start"))
        for ncores in range(1, 6):
            value, count = data[ncores]
            if value is None:
                last = None
                continue
            point = (x_at(ncores), y_at(value))
            if last is not None:
                chunks.append(f'<line x1="{last[0]:.1f}" y1="{last[1]:.1f}" x2="{point[0]:.1f}" y2="{point[1]:.1f}" stroke="{color}" stroke-width="3"/>')
            chunks.append(f'<circle cx="{point[0]:.1f}" cy="{point[1]:.1f}" r="5" fill="{color}"/>')
            chunks.append(_svg_text(point[0], point[1] - 10, f"{value:.2f} (k={count})", 10))
            last = point
    chunks.append(_svg_text(WIDTH - RIGHT, HEIGHT - 7, "k = evaluated cases; missing combinations are omitted", 10, "end"))
    chunks.append("</svg>")
    output.write_text("\n".join(chunks), encoding="utf-8")


def scatter_svg(rows: list[dict[str, Any]], output: Path) -> None:
    points = [
        (row["added_copy_bytes"], row["makespan"])
        for row in rows
        if row.get("case", "").startswith("case_")
        and row.get("added_copy_bytes") is not None
        and row.get("makespan") is not None
    ]
    max_x = max((x for x, _ in points), default=1) or 1
    max_y = max((y for _, y in points), default=1) or 1
    plot_w, plot_h = WIDTH - LEFT - RIGHT, HEIGHT - TOP - BOTTOM
    chunks = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" viewBox="0 0 {WIDTH} {HEIGHT}">',
        '<rect width="100%" height="100%" fill="white"/>',
        _svg_text(WIDTH / 2, 34, "Makespan vs added DDR bytes", 21),
        f'<line x1="{LEFT}" y1="{TOP}" x2="{LEFT}" y2="{HEIGHT-BOTTOM}" stroke="#35424C"/>',
        f'<line x1="{LEFT}" y1="{HEIGHT-BOTTOM}" x2="{WIDTH-RIGHT}" y2="{HEIGHT-BOTTOM}" stroke="#35424C"/>',
        _svg_text(WIDTH / 2, HEIGHT - 26, "Added COPY bytes", 14),
        _svg_text(24, HEIGHT / 2, "Makespan (cycles)", 14),
    ]
    for x_value, y_value in points:
        x = LEFT + x_value / max_x * plot_w
        y = HEIGHT - BOTTOM - y_value / max_y * plot_h
        chunks.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3.5" fill="#176B87" fill-opacity="0.55"/>')
    chunks.append("</svg>")
    output.write_text("\n".join(chunks), encoding="utf-8")


def _figure_manifest(
    rows: list[dict[str, Any]],
    results_dir: Path,
    output_dir: Path,
    generated_files: list[str],
) -> dict[str, Any]:
    identities = sorted({
        (row.get("experiment_id"), row.get("run_id"))
        for row in rows
    })
    cases = sorted({
        row.get("case") for row in rows
        if str(row.get("case", "")).startswith("case_")
    })
    case_core = sorted({
        (row.get("case"), row.get("ncores"))
        for row in rows
        if str(row.get("case", "")).startswith("case_")
    })
    paired_no_l2, paired_l2, paired_cache = _paired_problem3_series(rows)
    paired_counts = {
        str(ncores): paired_cache[ncores][1]
        for ncores in sorted(paired_cache)
        if paired_cache[ncores][1]
    }
    source_summary = results_dir / "summary.json"
    try:
        source_summary_label = source_summary.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        source_summary_label = source_summary.as_posix()
    try:
        output_label = output_dir.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        output_label = output_dir.as_posix()
    return {
        "source_summary": source_summary_label,
        "output_dir": output_label,
        "row_count": len(rows),
        "case_count": len(cases),
        "case_core_count": len(case_core),
        "cases": cases,
        "run_identities": [
            {"experiment_id": experiment_id, "run_id": run_id}
            for experiment_id, run_id in identities
        ],
        "strict_problem3_pair_counts_by_core": paired_counts,
        "metric_definitions": {
            "speedup": "single_core_baseline / official multicore makespan",
            "cache_speedup": "paired Problem 2 makespan / paired Problem 3 makespan",
            "cache_pairing": "same run identity and same case/core; both rows status=evaluated",
            "scatter": "official makespan versus official added_copy_bytes",
        },
        "missing_data_rule": "missing or non-evaluated combinations are omitted from plotted means and annotated by k",
        "generated_files": generated_files,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate SVG figures from official evaluator summary rows.")
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--experiment-id")
    parser.add_argument("--run-id")
    args = parser.parse_args()
    summary_path = args.results_dir / "summary.json"
    rows = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.is_file() else []
    if args.experiment_id is not None:
        rows = [row for row in rows if row.get("experiment_id") == args.experiment_id]
    if args.run_id is not None:
        rows = [row for row in rows if row.get("run_id") == args.run_id]
    run_identities = {
        (row.get("experiment_id"), row.get("run_id")) for row in rows
    }
    if len(run_identities) > 1 and (args.experiment_id is None or args.run_id is None):
        parser.error("select one experiment and run when summary.json contains multiple runs")
    output_dir = args.output_dir or args.results_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    for problem in (1, 2):
        data = _series(rows, problem, "no_l2")
        line_svg(
            f"Problem {problem}: arithmetic mean speedup",
            "Speedup",
            [(f"Problem {problem}", data, COLORS[f"p{problem}"])],
            output_dir / f"problem_{problem}_speedup.svg",
        )
    paired_no_l2, paired_l2, paired_cache = _paired_problem3_series(rows)
    line_svg(
        "Problem 3: no-L2 vs read-only L2",
        "Speedup vs single-core baseline",
        [
            ("no L2", paired_no_l2, COLORS["p2"]),
            ("read-only L2", paired_l2, COLORS["l2"]),
        ],
        output_dir / "problem_3_l2_comparison.svg",
    )
    line_svg(
        "Problem 3: L2 / no-L2 makespan ratio",
        "Cache speedup",
        [("read-only L2", paired_cache, COLORS["cache"])],
        output_dir / "problem_3_cache_speedup.svg",
    )
    scatter_svg(rows, output_dir / "makespan_vs_added_copy.svg")
    generated_files = [
        "problem_1_speedup.svg",
        "problem_2_speedup.svg",
        "problem_3_l2_comparison.svg",
        "problem_3_cache_speedup.svg",
        "makespan_vs_added_copy.svg",
    ]
    (output_dir / "figure_manifest.json").write_text(
        json.dumps(
            _figure_manifest(rows, args.results_dir, output_dir, generated_files),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ) + "\n",
        encoding="utf-8",
    )
    print(f"wrote SVG figures to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())