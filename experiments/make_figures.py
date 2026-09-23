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


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate SVG figures from official evaluator summary rows.")
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    summary_path = args.results_dir / "summary.json"
    rows = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.is_file() else []
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
    line_svg(
        "Problem 3: no-L2 vs read-only L2",
        "Speedup vs single-core baseline",
        [
            ("no L2", _series(rows, 2, "no_l2"), COLORS["p2"]),
            ("read-only L2", _series(rows, 3, "l2"), COLORS["l2"]),
        ],
        output_dir / "problem_3_l2_comparison.svg",
    )
    cache_values = {n: (None, 0) for n in range(1, 6)}
    for ncores in range(1, 6):
        ratios = [
            row["cache_speedup"] for row in rows
            if row.get("case", "").startswith("case_")
            and row.get("problem") == 3 and row.get("ncores") == ncores
            and row.get("cache_speedup") is not None
        ]
        cache_values[ncores] = ((sum(ratios) / len(ratios)) if ratios else None, len(ratios))
    line_svg(
        "Problem 3: L2 / no-L2 makespan ratio",
        "Cache speedup",
        [("read-only L2", cache_values, COLORS["cache"])],
        output_dir / "problem_3_cache_speedup.svg",
    )
    scatter_svg(rows, output_dir / "makespan_vs_added_copy.svg")
    print(f"wrote SVG figures to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())