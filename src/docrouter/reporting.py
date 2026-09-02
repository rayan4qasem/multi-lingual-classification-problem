"""Rendering results for the terminal.

Split from `cli.py` so command functions do argument handling and
orchestration only. Everything here takes plain domain objects and returns
Rich renderables — no I/O, no `sys.exit`, no knowledge of Typer — which is
what makes the tables testable without invoking a command.
"""

from __future__ import annotations

from rich.table import Table

from .evaluate import Report
from .fewshot import ExampleSet
from .labeling.store import StoreStats
from .models import Prediction
from .taxonomy import Taxonomy
from .threshold import CalibrationReport, ClassThreshold, SweepPoint


def institutions_table(taxonomy: Taxonomy) -> Table:
    table = Table(title=f"Institutions (taxonomy v{taxonomy.version})")
    table.add_column("id", style="cyan", no_wrap=True)
    table.add_column("الجهة", justify="right")
    table.add_column("english")
    for institution in taxonomy.institutions:
        table.add_row(institution.id, institution.name_ar, institution.name_en)
    return table


def routing_table(predictions: list[Prediction], taxonomy: Taxonomy, limit: int = 40) -> Table:
    table = Table(title="Routing")
    table.add_column("doc", style="cyan", no_wrap=True)
    table.add_column("الجهة", justify="right")
    table.add_column("conf", justify="right")
    table.add_column("")
    for prediction in predictions[:limit]:
        flag = "[yellow]مراجعة[/yellow]" if prediction.needs_review else ""
        table.add_row(
            prediction.doc_id,
            taxonomy.name_ar(prediction.institution_id),
            f"{prediction.confidence:.2f}",
            flag,
        )
    return table


def per_class_table(report: Report) -> Table:
    table = Table(title="Per class")
    table.add_column("institution", style="cyan")
    table.add_column("n", justify="right")
    table.add_column("P", justify="right")
    table.add_column("R", justify="right")
    table.add_column("F1", justify="right")
    for metrics in sorted(report.per_class, key=lambda m: m.f1):
        if metrics.support:
            table.add_row(
                metrics.institution_id,
                str(metrics.support),
                f"{metrics.precision:.2f}",
                f"{metrics.recall:.2f}",
                f"{metrics.f1:.2f}",
            )
    return table


def reliability_table(calibration: CalibrationReport) -> Table:
    table = Table(title="Reliability")
    table.add_column("confidence", style="cyan")
    table.add_column("n", justify="right")
    table.add_column("predicted", justify="right")
    table.add_column("actual", justify="right")
    table.add_column("gap", justify="right")
    for bin_ in calibration.bins:
        colour = "green" if abs(bin_.gap) < 0.1 else "yellow"
        table.add_row(
            f"{bin_.low:.1f}-{bin_.high:.1f}",
            str(bin_.n),
            f"{bin_.mean_confidence:.2f}",
            f"{bin_.accuracy:.2f}",
            f"[{colour}]{bin_.gap:+.2f}[/{colour}]",
        )
    return table


def sweep_table(
    points: list[SweepPoint],
    on_target: SweepPoint | None,
    cheapest: SweepPoint,
) -> Table:
    total = points[0].auto_routed + points[0].held if points else 0
    table = Table(title=f"Threshold sweep (n={total})")
    table.add_column("t", justify="right", style="cyan")
    table.add_column("auto", justify="right")
    table.add_column("coverage", justify="right")
    table.add_column("auto acc", justify="right")
    table.add_column("misrouted", justify="right")
    table.add_column("held", justify="right")
    table.add_column("cost", justify="right")
    table.add_column("pick", no_wrap=True)

    for point in points:
        marks = []
        if on_target and point.threshold == on_target.threshold:
            marks.append("[green]target[/green]")
        if point.threshold == cheapest.threshold:
            marks.append("[magenta]cheapest[/magenta]")
        table.add_row(
            f"{point.threshold:.2f}",
            str(point.auto_routed),
            f"{point.coverage:.0%}",
            f"{point.auto_accuracy:.1%}" if point.defined else "—",
            str(point.misrouted),
            str(point.held),
            f"{point.expected_cost:.0f}",
            " · ".join(marks),
        )
    return table


def class_thresholds_table(rows: list[ClassThreshold], target: float) -> Table:
    table = Table(title=f"Per-class cut-offs (target {target:.0%})")
    table.add_column("institution", style="cyan")
    table.add_column("n", justify="right")
    table.add_column("t", justify="right")
    table.add_column("coverage", justify="right")
    table.add_column("")
    for row in rows:
        table.add_row(
            row.institution_id,
            str(row.support),
            f"{row.threshold:.2f}" if row.threshold is not None else "[red]none[/red]",
            f"{row.coverage:.0%}" if row.threshold is not None else "—",
            "[yellow]thin[/yellow]" if row.thin else "",
        )
    return table


def coverage_table(stats: StoreStats, taxonomy: Taxonomy, target_per_class: int) -> Table:
    table = Table(title=f"Coverage (target {target_per_class}/class)")
    table.add_column("institution", style="cyan")
    table.add_column("الجهة", justify="right")
    table.add_column("have", justify="right")
    table.add_column("need", justify="right")
    for institution_id in taxonomy.ids:
        have = stats.per_class.get(institution_id, 0)
        need = max(0, target_per_class - have)
        table.add_row(
            institution_id,
            taxonomy.name_ar(institution_id),
            str(have),
            "[green]0[/green]" if not need else f"[yellow]{need}[/yellow]",
        )
    return table


def examples_table(example_set: ExampleSet, taxonomy: Taxonomy) -> Table:
    table = Table(title=f"Few-shot examples ({len(example_set.examples)})")
    table.add_column("doc", style="cyan", no_wrap=True)
    table.add_column("الجهة", justify="right")
    table.add_column("chars", justify="right")
    table.add_column("")
    for example in example_set.examples:
        flags = []
        if example.is_override:
            flags.append(f"[yellow]صُحح من {example.corrected_from}[/yellow]")
        if example.truncated:
            flags.append("[dim]مقتطع[/dim]")
        table.add_row(
            example.doc_id,
            taxonomy.name_ar(example.label),
            str(len(example.text)),
            " ".join(flags),
        )
    return table
