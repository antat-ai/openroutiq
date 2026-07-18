from __future__ import annotations

import html
import json
import math
from collections import defaultdict
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from openroutiq.benchmark.core import BenchmarkObservation, BenchmarkRun, BenchmarkSummary


INK = "#172033"
MUTED = "#667085"
GRID = "#D9DEE8"
BLUE = "#356AE6"
BLUE_LIGHT = "#DCE7FF"
GOLD = "#D49A00"
ORANGE = "#E36B2C"
PINK = "#C94C83"
OLIVE = "#7A8B32"
PALETTE = (BLUE, GOLD, ORANGE, PINK, OLIVE)


def _escape(value: Any) -> str:
    return html.escape(str(value), quote=True)


def _fmt_cost(value: float | None) -> str:
    if value is None:
        return "n/a"
    if value == 0:
        return "$0"
    if value < 0.0001:
        return f"${value:.2e}"
    if value < 0.01:
        return f"${value:.6f}"
    return f"${value:.4f}"


def _scale(value: float, low: float, high: float, start: float, end: float) -> float:
    if math.isclose(low, high):
        return (start + end) / 2
    return start + (value - low) * (end - start) / (high - low)


def _svg_frame(title: str, subtitle: str, body: str, *, height: int = 430) -> str:
    return f"""
    <figure class="chart-card">
      <figcaption><strong>{_escape(title)}</strong><span>{_escape(subtitle)}</span></figcaption>
      <svg viewBox="0 0 920 {height}" role="img" aria-label="{_escape(title)}">
        {body}
      </svg>
    </figure>
    """


def _accuracy_cost_chart(summaries: Sequence[BenchmarkSummary]) -> str:
    legend_rows = math.ceil(len(summaries) / 2)
    height = 430 + legend_rows * 24
    left, right, top, bottom = 90, 875, 45, 350
    costs = [summary.mean_cost_ci_high for summary in summaries]
    accuracies = [
        bound
        for summary in summaries
        for bound in (summary.accuracy_ci_low, summary.accuracy_ci_high)
    ]
    max_cost = max(costs, default=1) or 1
    min_accuracy = min(accuracies, default=0)
    max_accuracy = max(accuracies, default=1)
    padding = max(0.02, (max_accuracy - min_accuracy) * 0.15)
    y_low = max(0, min_accuracy - padding)
    y_high = min(1, max_accuracy + padding)
    if math.isclose(y_low, y_high):
        y_low, y_high = max(0, y_low - 0.05), min(1, y_high + 0.05)
    parts = [
        f'<line x1="{left}" y1="{bottom}" x2="{right}" y2="{bottom}" stroke="{INK}"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{bottom}" stroke="{INK}"/>',
    ]
    for index in range(6):
        y_value = y_low + (y_high - y_low) * index / 5
        y = _scale(y_value, y_low, y_high, bottom, top)
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{right}" y2="{y:.1f}" stroke="{GRID}"/>')
        parts.append(
            f'<text x="{left - 12}" y="{y + 4:.1f}" text-anchor="end" class="tick">{y_value:.0%}</text>'
        )
    for index in range(6):
        cost = max_cost * index / 5
        x = _scale(cost, 0, max_cost, left, right)
        parts.append(
            f'<text x="{x:.1f}" y="{bottom + 28}" text-anchor="middle" class="tick">{_escape(_fmt_cost(cost))}</text>'
        )
    frontier = sorted(
        (summary for summary in summaries if summary.pareto_optimal),
        key=lambda summary: summary.mean_cost,
    )
    if len(frontier) > 1:
        points = " ".join(
            f"{_scale(item.mean_cost, 0, max_cost, left, right):.1f},"
            f"{_scale(item.accuracy, y_low, y_high, bottom, top):.1f}"
            for item in frontier
        )
        parts.append(
            f'<polyline points="{points}" fill="none" stroke="{INK}" stroke-width="2" stroke-dasharray="7 5"/>'
        )
    for index, summary in enumerate(summaries):
        x = _scale(summary.mean_cost, 0, max_cost, left, right)
        y = _scale(summary.accuracy, y_low, y_high, bottom, top)
        color = PALETTE[index % len(PALETTE)]
        radius = 9 if summary.pareto_optimal else 7
        x_low = _scale(summary.mean_cost_ci_low, 0, max_cost, left, right)
        x_high = _scale(summary.mean_cost_ci_high, 0, max_cost, left, right)
        y_low_ci = _scale(summary.accuracy_ci_low, y_low, y_high, bottom, top)
        y_high_ci = _scale(summary.accuracy_ci_high, y_low, y_high, bottom, top)
        parts.append(
            f'<line x1="{x_low:.1f}" y1="{y:.1f}" x2="{x_high:.1f}" y2="{y:.1f}" stroke="{color}" stroke-width="2"/>'
        )
        parts.append(
            f'<line x1="{x:.1f}" y1="{y_low_ci:.1f}" x2="{x:.1f}" y2="{y_high_ci:.1f}" stroke="{color}" stroke-width="2"/>'
        )
        parts.append(
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{radius}" fill="{color}" stroke="{INK}" stroke-width="1.5"/>'
        )
        parts.append(
            f'<text x="{x:.1f}" y="{y + 3.5:.1f}" text-anchor="middle" class="point-number">{index + 1}</text>'
        )
    legend_top = 430
    legend_width = (right - left) / 2
    for index, summary in enumerate(summaries):
        column = index % 2
        row = index // 2
        x = left + column * legend_width
        y = legend_top + row * 24
        color = PALETTE[index % len(PALETTE)]
        parts.append(f'<circle cx="{x + 7:.1f}" cy="{y:.1f}" r="7" fill="{color}"/>')
        parts.append(
            f'<text x="{x + 7:.1f}" y="{y + 3.5:.1f}" text-anchor="middle" class="point-number">{index + 1}</text>'
        )
        parts.append(
            f'<text x="{x + 20:.1f}" y="{y + 4:.1f}" class="legend-label">{_escape(summary.router)}</text>'
        )
    parts.extend(
        [
            f'<text x="{(left + right) / 2}" y="{bottom + 52}" text-anchor="middle" class="axis-label">Mean cost per request (USD, lower is better)</text>',
            f'<text transform="translate(24 {(top + bottom) / 2}) rotate(-90)" text-anchor="middle" class="axis-label">Mean task accuracy (higher is better)</text>',
            f'<text x="{right}" y="{top + 12}" text-anchor="end" class="note">Bars: approximate 95% CI · dashed: observed Pareto frontier</text>',
        ]
    )
    return _svg_frame(
        "Accuracy × cost frontier",
        "Each point is one router over the same cases and candidate pool.",
        "".join(parts),
        height=height,
    )


def _ranked_bar_chart(
    summaries: Sequence[BenchmarkSummary],
    *,
    title: str,
    subtitle: str,
    value: str,
    formatter,
    lower_is_better: bool,
) -> str:
    ordered = sorted(
        summaries,
        key=lambda item: (getattr(item, value) is None, getattr(item, value) or 0),
        reverse=not lower_is_better,
    )
    values = [float(getattr(item, value) or 0) for item in ordered]
    maximum = max(values, default=1) or 1
    height = max(260, 95 + len(ordered) * 48)
    left = min(
        340,
        max(190, 10 + max((len(item.router) for item in ordered), default=0) * 7),
    )
    right, top = 855, 30
    parts: list[str] = []
    for index, (summary, amount) in enumerate(zip(ordered, values, strict=True)):
        y = top + index * 48
        bar_width = _scale(amount, 0, maximum, 0, right - left)
        parts.append(
            f'<text x="{left - 14}" y="{y + 18}" text-anchor="end" class="row-label">{_escape(summary.router)}</text>'
        )
        parts.append(
            f'<rect x="{left}" y="{y}" width="{max(2, bar_width):.1f}" height="25" rx="3" fill="{BLUE}"/>'
        )
        parts.append(
            f'<text x="{min(right - 5, left + bar_width + 9):.1f}" y="{y + 18}" class="value-label">{_escape(formatter(getattr(summary, value)))}</text>'
        )
    return _svg_frame(title, subtitle, "".join(parts), height=height)


def _slice_accuracy(
    observations: Iterable[BenchmarkObservation],
    field: str,
) -> tuple[list[str], list[str], dict[tuple[str, str], float]]:
    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    routers: set[str] = set()
    columns: set[str] = set()
    for row in observations:
        raw = getattr(row, field)
        label = str(raw or "unknown")
        routers.add(row.router)
        columns.add(label)
        grouped[(row.router, label)].append(row.accuracy)
    values = {key: sum(items) / len(items) for key, items in grouped.items()}
    return sorted(routers), sorted(columns), values


def _heatmap(
    observations: Sequence[BenchmarkObservation],
    *,
    field: str,
    title: str,
    subtitle: str,
) -> str:
    routers, columns, values = _slice_accuracy(observations, field)
    cell_width = max(68, min(130, 680 // max(1, len(columns))))
    cell_height = 38
    left = min(
        340,
        max(190, 10 + max((len(router) for router in routers), default=0) * 7),
    )
    top = 145 if field == "model_id" else 100
    width = max(920, left + len(columns) * cell_width + 30)
    height = max(270, top + len(routers) * cell_height + 45)
    parts: list[str] = []
    for column_index, label in enumerate(columns):
        x = left + column_index * cell_width + cell_width / 2
        if field == "model_id":
            parts.append(
                f'<text transform="translate({x:.1f} {top - 12}) rotate(-35)" '
                f'text-anchor="start" class="heat-label">{_escape(label)}</text>'
            )
        else:
            parts.append(
                f'<text x="{x:.1f}" y="{top - 15}" text-anchor="middle" '
                f'class="heat-label">{_escape(label[:24])}</text>'
            )
    for row_index, router in enumerate(routers):
        y = top + row_index * cell_height
        parts.append(
            f'<text x="{left - 12}" y="{y + 25}" text-anchor="end" class="row-label">{_escape(router)}</text>'
        )
        for column_index, label in enumerate(columns):
            x = left + column_index * cell_width
            amount = values.get((router, label))
            if amount is None:
                fill, text = "#F1F3F7", "N/A"
            else:
                opacity = 0.15 + 0.85 * amount
                fill, text = f"rgba(53,106,230,{opacity:.3f})", f"{amount:.1%}"
            parts.append(
                f'<rect x="{x}" y="{y}" width="{cell_width - 3}" height="{cell_height - 3}" rx="3" fill="{fill}"/>'
            )
            parts.append(
                f'<text x="{x + (cell_width - 3) / 2:.1f}" y="{y + 24}" text-anchor="middle" class="heat-value">{text}</text>'
            )
    return _svg_frame(title, subtitle, "".join(parts), height=height).replace(
        'viewBox="0 0 920', f'viewBox="0 0 {width}'
    )


def _summary_table(summaries: Sequence[BenchmarkSummary]) -> str:
    rows = []
    for summary in sorted(summaries, key=lambda item: (-item.accuracy, item.mean_cost)):
        rows.append(
            "<tr>"
            f"<th scope='row'>{_escape(summary.router)}</th>"
            f"<td>{summary.accuracy:.2%}<small>{summary.accuracy_ci_low:.2%}–{summary.accuracy_ci_high:.2%}</small></td>"
            f"<td>{_fmt_cost(summary.mean_cost)}<small>{_fmt_cost(summary.mean_cost_ci_low)}–{_fmt_cost(summary.mean_cost_ci_high)}</small></td>"
            f"<td>{_fmt_cost(summary.total_cost)}</td>"
            f"<td>{_fmt_cost(summary.cost_per_correct)}</td>"
            f"<td>{summary.coverage:.2%}</td>"
            f"<td>{summary.failure_rate:.2%}</td>"
            f"<td>{summary.constraint_violation_rate:.2%}</td>"
            f"<td>{summary.routing_p95_ms:.2f} ms<small>{_escape(summary.timing_scope)}</small></td>"
            f"<td>{summary.selection_stability:.2%}</td>"
            f"<td>{'n/a' if summary.calibration_error is None else f'{summary.calibration_error:.3f}'}</td>"
            f"<td>{summary.pareto_distance:.3f}</td>"
            f"<td>{'yes' if summary.pareto_optimal else 'no'}</td>"
            "</tr>"
        )
    return (
        "<div class='table-wrap'><table><thead><tr>"
        "<th>Router</th><th>Accuracy</th><th>Mean cost</th><th>Total cost</th>"
        "<th>Cost/correct</th><th>Coverage</th><th>Failures</th><th>Violations</th>"
        "<th>Measured p95</th>"
        "<th>Stability</th><th>Calibration ECE</th><th>Pareto distance</th><th>Pareto</th>"
        "</tr></thead><tbody>" + "".join(rows) + "</tbody></table></div>"
    )


def render_benchmark_report(run: BenchmarkRun, output: str | Path) -> Path:
    """Write a self-contained, dependency-free HTML benchmark report."""
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    summaries = run.summaries
    observations = run.observations
    charts = [
        _accuracy_cost_chart(summaries),
        _ranked_bar_chart(
            summaries,
            title="Cost per correct-equivalent answer",
            subtitle="Total inference cost divided by the sum of per-case accuracy; lower is better.",
            value="cost_per_correct",
            formatter=_fmt_cost,
            lower_is_better=True,
        ),
        _ranked_bar_chart(
            summaries,
            title="Accuracy per dollar",
            subtitle="Sum of per-case accuracy divided by total inference cost; higher is better.",
            value="accuracy_per_dollar",
            formatter=lambda value: "n/a" if value is None else f"{value:.2f}",
            lower_is_better=False,
        ),
        _heatmap(
            observations,
            field="task",
            title="Accuracy by task",
            subtitle="Mean task score for each router; empty cells indicate no covered observations.",
        ),
        _heatmap(
            observations,
            field="provider",
            title="Accuracy by selected provider",
            subtitle="Provider slices retain concrete model and reasoning variants in the source rows.",
        ),
        _heatmap(
            observations,
            field="model_id",
            title="Accuracy by selected model variant",
            subtitle="Every provider model and reasoning configuration keeps a distinct candidate ID.",
        ),
        _heatmap(
            observations,
            field="reasoning_level",
            title="Accuracy by selected reasoning level",
            subtitle="Results are sliced by the concrete reasoning variant selected for each request.",
        ),
    ]
    diagnostic = [
        summary
        for summary in summaries
        if summary.router.casefold().startswith("oracle")
        or "oracle upper bound" in summary.router.casefold()
    ]
    deployable = [summary for summary in summaries if summary not in diagnostic] or summaries
    best_accuracy = max(deployable, key=lambda item: item.accuracy, default=None)
    cheapest = min(deployable, key=lambda item: item.mean_cost, default=None)
    pareto = [summary.router for summary in deployable if summary.pareto_optimal]
    oracle = max(diagnostic, key=lambda item: item.accuracy, default=None)
    accuracy_fragment = (
        f"{_escape(best_accuracy.router)} {best_accuracy.accuracy:.2%}" if best_accuracy else "n/a"
    )
    cheapest_fragment = (
        f"{_escape(cheapest.router)} {_fmt_cost(cheapest.mean_cost)}" if cheapest else "n/a"
    )
    targets = run.metadata.get("measurement_targets", {})
    target_cards = ""
    if isinstance(targets, dict) and isinstance(
        targets.get("cost_budget_per_request"), (int, float)
    ):
        budget = float(targets["cost_budget_per_request"])
        within_budget = [summary for summary in deployable if summary.mean_cost <= budget]
        best_at_budget = max(within_budget, key=lambda item: item.accuracy, default=None)
        value = (
            "none eligible"
            if best_at_budget is None
            else f"{_escape(best_at_budget.router)} {best_at_budget.accuracy:.2%}"
        )
        target_cards += (
            f'<div class="kpi"><span>Accuracy at {_fmt_cost(budget)} budget</span>'
            f"<strong>{value}</strong></div>"
        )
    if isinstance(targets, dict) and isinstance(targets.get("accuracy_target"), (int, float)):
        accuracy_target = float(targets["accuracy_target"])
        at_target = [summary for summary in deployable if summary.accuracy >= accuracy_target]
        cheapest_at_target = min(at_target, key=lambda item: item.mean_cost, default=None)
        value = (
            "none eligible"
            if cheapest_at_target is None
            else f"{_escape(cheapest_at_target.router)} {_fmt_cost(cheapest_at_target.mean_cost)}"
        )
        target_cards += (
            f'<div class="kpi"><span>Cost at {accuracy_target:.0%} accuracy</span>'
            f"<strong>{value}</strong></div>"
        )
    generated_data = json.dumps(run.summary_dict(), separators=(",", ":")).replace("</", "<\\/")
    document = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{_escape(run.dataset)} benchmark</title>
  <style>
    :root {{ color-scheme: light; --ink:{INK}; --muted:{MUTED}; --grid:{GRID}; --blue:{BLUE}; }}
    * {{ box-sizing: border-box; }}
    body {{ margin:0; background:#F6F8FC; color:var(--ink); font:15px/1.5 Inter,Segoe UI,Arial,sans-serif; }}
    main {{ max-width:1180px; margin:auto; padding:42px 24px 70px; }}
    header {{ margin-bottom:28px; }}
    h1 {{ margin:0 0 6px; font-size:34px; letter-spacing:-.03em; }}
    h2 {{ margin:34px 0 12px; font-size:22px; }}
    .sub {{ color:var(--muted); margin:0; }}
    .kpis {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(200px,1fr)); gap:14px; margin:24px 0; }}
    .kpi,.chart-card,.table-wrap,.methods {{ background:#fff; border:1px solid #E2E6EE; border-radius:12px; box-shadow:0 5px 18px rgba(23,32,51,.05); }}
    .kpi {{ padding:18px; }} .kpi span {{ display:block; color:var(--muted); font-size:13px; }}
    .kpi strong {{ display:block; font-size:21px; margin-top:5px; }}
    .chart-card {{ margin:18px 0; padding:18px; overflow-x:auto; }}
    .chart-card figcaption {{ display:flex; flex-direction:column; margin-bottom:4px; }}
    .chart-card figcaption strong {{ font-size:18px; }} .chart-card figcaption span {{ color:var(--muted); font-size:13px; }}
    svg {{ width:100%; min-width:780px; height:auto; }}
    svg text {{ fill:{INK}; font-family:Inter,Segoe UI,Arial,sans-serif; }}
    .tick,.note {{ fill:{MUTED}; font-size:11px; }} .axis-label {{ font-size:12px; font-weight:600; }}
    .point-number {{ fill:#fff; font-size:9px; font-weight:750; }}
    .legend-label,.row-label {{ font-size:12px; font-weight:650; }} .value-label {{ font-size:11px; font-family:Consolas,monospace; }}
    .heat-label {{ font-size:10px; fill:{MUTED}; }} .heat-value {{ font-size:11px; font-weight:650; }}
    .table-wrap {{ overflow:auto; }} table {{ width:100%; border-collapse:collapse; min-width:900px; }}
    th,td {{ padding:11px 13px; text-align:right; border-bottom:1px solid #E8EBF1; font-variant-numeric:tabular-nums; }}
    td small {{ display:block; color:var(--muted); font-size:10px; }}
    th:first-child,td:first-child {{ text-align:left; }} thead th {{ background:#F7F9FC; font-size:12px; color:var(--muted); }}
    .methods {{ padding:20px; }} code {{ font-family:Consolas,monospace; }}
    @media (max-width:700px) {{ main {{ padding:26px 14px 50px; }} h1 {{ font-size:28px; }} }}
    @media print {{ body {{ background:#fff; }} .chart-card,.table-wrap,.kpi,.methods {{ box-shadow:none; break-inside:avoid; }} }}
  </style>
</head>
<body><main>
  <header><h1>{_escape(run.dataset)} benchmark</h1><p class="sub">Accuracy × cost comparison · generated {_escape(run.created_at)}</p></header>
  <section class="kpis">
    <div class="kpi"><span>Highest deployable accuracy</span><strong>{accuracy_fragment}</strong></div>
    <div class="kpi"><span>Lowest deployable mean cost</span><strong>{cheapest_fragment}</strong></div>
    <div class="kpi"><span>Deployable Pareto routers</span><strong>{_escape(", ".join(pareto) or "none")}</strong></div>
    {f'<div class="kpi"><span>Diagnostic oracle upper bound</span><strong>{_escape(oracle.router)} {oracle.accuracy:.2%}</strong></div>' if oracle else ""}
    <div class="kpi"><span>Cases × routers</span><strong>{len(set(row.case_id for row in observations))} × {len(summaries)}</strong></div>
    {target_cards}
  </section>
  <h2>Decision view</h2>
  {"".join(charts)}
  <h2>Exact results</h2>
  {_summary_table(summaries)}
  <h2>Method</h2>
  <section class="methods">
    <p><strong>Primary comparison:</strong> accuracy and inference cost are shown jointly. Pareto-optimal routers are not dominated by another router that is both at least as accurate and no more expensive.</p>
    <p><strong>Decision KPIs:</strong> diagnostic oracle/upper-bound configurations are excluded from deployable winners and targets. They remain in charts and exact results as unattainable references.</p>
    <p><strong>Accuracy:</strong> mean per-case score on a 0–1 scale. <strong>Cost:</strong> recorded provider cost in replay, or reported/frozen-price cost in live runs. Error bars are approximate normal 95% confidence intervals over case-level means; publish the raw rows for paired bootstrap or significance analysis.</p>
    <p><strong>Guardrails:</strong> coverage, constraint violations, failures, and measured latency are reported separately so a router cannot improve apparent cost by silently dropping difficult requests. Timing scope distinguishes routing-only, replay lookup, and combined end-to-end calls; those scopes should not be compared as if identical.</p>
    <p><strong>Calibration:</strong> when a router supplies predicted accuracy, calibration error is 10-bin equal-width expected calibration error (ECE; lower is better).</p>
    <p><strong>Audit files:</strong> this HTML embeds the compact run summary used by the charts. The sibling <code>benchmark-observations.csv</code> contains flat case-level rows, while <code>benchmark-results.json</code> retains the complete observations and metadata.</p>
  </section>
  <script type="application/json" id="benchmark-data">{generated_data}</script>
</main></body></html>
"""
    target.write_text(document, encoding="utf-8")
    return target
