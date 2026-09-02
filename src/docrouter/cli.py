"""Command line interface.

    docrouter taxonomy                 inspect the institution list
    docrouter generate                 build a mock dataset
    docrouter classify <path>          route one file or a folder
    docrouter train-baseline           fit the offline TF-IDF model
    docrouter evaluate                 score a backend against labels
    docrouter batch submit|collect     bulk routing at half price
"""

from __future__ import annotations

import json
import os
import random
from collections import Counter
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from . import ingest, mockdata
from .classify import BaselineClassifier, LLMClassifier
from .evaluate import evaluate as run_evaluate
from .models import Document
from .taxonomy import load as load_taxonomy

app = typer.Typer(add_completion=False, help="Arabic government document routing.")
batch_app = typer.Typer(help="Bulk routing through the Batches API (50% cost).")
label_app = typer.Typer(help="Human-in-the-loop labeling of real documents.")
app.add_typer(batch_app, name="batch")
app.add_typer(label_app, name="label")

console = Console()

DATA = Path("data")
RUNS = Path("runs")
MODELS = Path("models")


def _require_key() -> None:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        console.print(
            "[red]ANTHROPIC_API_KEY is not set.[/red] "
            "Set it in your shell or copy .env.example to .env and load it."
        )
        raise typer.Exit(1)


def _load_baseline(tax, threshold: float = 0.55) -> BaselineClassifier:
    model_path = MODELS / "baseline.joblib"
    if not model_path.exists():
        console.print(
            f"[red]No trained baseline at {model_path}.[/red] "
            "Run `docrouter train-baseline` first."
        )
        raise typer.Exit(1)
    return BaselineClassifier(taxonomy=tax, review_threshold=threshold).load(model_path)


def _write_predictions(predictions, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for p in predictions:
            fh.write(p.model_dump_json() + "\n")
    return path


@app.command()
def taxonomy() -> None:
    """Show the institution catalogue and validate the config."""
    tax = load_taxonomy()
    table = Table(title=f"Institutions (taxonomy v{tax.version})")
    table.add_column("id", style="cyan", no_wrap=True)
    table.add_column("الجهة", justify="right")
    table.add_column("english")
    for inst in tax.institutions:
        table.add_row(inst.id, inst.name_ar, inst.name_en)
    console.print(table)
    console.print(f"classes: {len(tax.institutions)}   fallback: {tax.fallback_id}")
    console.print(f"declared confusion pairs: {len(tax.confusion_pairs)}")


@app.command()
def generate(
    engine: str = typer.Option(
        "curated",
        help="curated (authored, offline, hard) | template (synthetic, easy) | llm (needs API key)",
    ),
    per_class: int = typer.Option(20, help="documents per institution; template/llm only"),
    seed: int = typer.Option(7),
    hard_only: bool = typer.Option(False, help="curated only: keep just the boundary cases"),
    out: Path = typer.Option(DATA / "generated" / "mock.jsonl"),
    as_files: bool = typer.Option(False, help="also write .txt files under data/generated/files"),
) -> None:
    """Create a labeled mock dataset."""
    if engine == "curated":
        docs = mockdata.generate_curated(seed=seed, hard_only=hard_only)
    elif engine == "llm":
        _require_key()
        console.print(f"[cyan]Generating with Claude — {per_class} docs x classes...[/cyan]")
        docs = mockdata.generate_llm(n_per_class=per_class, seed=seed)
    elif engine == "template":
        docs = mockdata.generate_templates(n_per_class=per_class, seed=seed)
    else:
        console.print(f"[red]unknown engine {engine!r}[/red]")
        raise typer.Exit(1)

    mockdata.save_jsonl(docs, out)
    scanned = sum(1 for d in docs if d.source == "ocr")
    console.print(
        f"[green]Wrote {len(docs)} documents[/green] to {out} "
        f"({scanned} carry simulated OCR noise)"
    )
    if as_files:
        directory = mockdata.save_as_files(docs, DATA / "generated" / "files")
        console.print(f"[green]Also wrote .txt files[/green] under {directory}")


@app.command()
def classify(
    path: Path = typer.Argument(..., help="a file, a folder, or a .jsonl dataset"),
    backend: str = typer.Option("llm", help="llm or baseline"),
    model: str = typer.Option(None, help="override the Claude model"),
    effort: str = typer.Option(None, help="low | medium | high | xhigh | max"),
    threshold: float = typer.Option(0.55, help="below this, hold for human review"),
    ocr: str = typer.Option("claude", help="OCR backend for scanned input: claude or tesseract"),
    limit: int = typer.Option(0, help="stop after N documents (0 = all)"),
    out: Path = typer.Option(RUNS / "predictions.jsonl"),
) -> None:
    """Route documents to institutions."""
    tax = load_taxonomy()

    if path.suffix == ".jsonl":
        docs = mockdata.load_jsonl(path)
    elif path.is_dir():
        if backend == "llm":
            _require_key()
        docs = ingest.load_directory(path, ocr_backend=ocr)
    else:
        if backend == "llm":
            _require_key()
        docs = [ingest.load_document(path, ocr_backend=ocr)]

    if limit:
        docs = docs[:limit]
    if not docs:
        console.print("[yellow]No documents found.[/yellow]")
        raise typer.Exit(1)

    if backend == "llm":
        _require_key()
        clf = LLMClassifier(
            taxonomy=tax, model=model, effort=effort, review_threshold=threshold
        )
    elif backend == "baseline":
        model_path = MODELS / "baseline.joblib"
        if not model_path.exists():
            console.print(
                f"[red]No trained baseline at {model_path}.[/red] "
                "Run `docrouter train-baseline` first."
            )
            raise typer.Exit(1)
        clf = BaselineClassifier(taxonomy=tax, review_threshold=threshold).load(model_path)
    else:
        console.print(f"[red]unknown backend {backend!r}[/red]")
        raise typer.Exit(1)

    with console.status(f"Classifying {len(docs)} document(s) via {clf.name}..."):
        predictions = clf.classify_many(docs)

    table = Table(title="Routing")
    table.add_column("doc", style="cyan", no_wrap=True)
    table.add_column("الجهة", justify="right")
    table.add_column("conf", justify="right")
    table.add_column("")
    for p in predictions[:40]:
        flag = "[yellow]مراجعة[/yellow]" if p.needs_review else ""
        table.add_row(p.doc_id, tax.name_ar(p.institution_id), f"{p.confidence:.2f}", flag)
    console.print(table)
    if len(predictions) > 40:
        console.print(f"... and {len(predictions) - 40} more")

    _write_predictions(predictions, out)
    held = sum(1 for p in predictions if p.needs_review)
    console.print(f"[green]Wrote {len(predictions)} predictions[/green] to {out}")
    console.print(f"held for review: {held} / {len(predictions)}")


@app.command("train-baseline")
def train_baseline(
    dataset: Path = typer.Option(DATA / "generated" / "mock.jsonl"),
    test_ratio: float = typer.Option(0.25),
    seed: int = typer.Option(7),
    out: Path = typer.Option(MODELS / "baseline.joblib"),
) -> None:
    """Fit the offline TF-IDF + SVM baseline and report held-out accuracy."""
    docs = mockdata.load_jsonl(dataset)
    rng = random.Random(seed)
    rng.shuffle(docs)
    split = int(len(docs) * (1 - test_ratio))
    train, test = docs[:split], docs[split:]

    per_class = len(train) / max(len(load_taxonomy().institutions), 1)
    if per_class < 15:
        console.print(
            f"[yellow]Thin training data: ~{per_class:.0f} documents per class.[/yellow] "
            "Expect low confidence and most documents held for review. "
            "For a usable baseline, train on the template corpus "
            "(`docrouter generate --engine template --per-class 40`) and keep "
            "the curated set as a held-out benchmark."
        )

    clf = BaselineClassifier()
    console.print(f"Training on {len(train)} documents ({len(test)} held out)...")
    clf.fit(train)
    clf.save(out)

    report = run_evaluate(test, clf.classify_many(test))
    for line in report.summary_lines(load_taxonomy()):
        console.print(line)
    console.print(f"[green]Saved model[/green] to {out}")


@app.command()
def evaluate(
    dataset: Path = typer.Option(DATA / "generated" / "mock.jsonl"),
    predictions: Path = typer.Option(RUNS / "predictions.jsonl"),
    out: Path = typer.Option(RUNS / "report.json"),
    show_confusion: bool = typer.Option(False),
) -> None:
    """Score predictions against the dataset's labels."""
    docs = mockdata.load_jsonl(dataset)
    from .models import Prediction

    with predictions.open(encoding="utf-8") as fh:
        preds = [Prediction.model_validate_json(line) for line in fh if line.strip()]

    tax = load_taxonomy()
    report = run_evaluate(docs, preds, taxonomy=tax)

    for line in report.summary_lines(tax):
        console.print(line)

    table = Table(title="Per class")
    table.add_column("institution", style="cyan")
    table.add_column("n", justify="right")
    table.add_column("P", justify="right")
    table.add_column("R", justify="right")
    table.add_column("F1", justify="right")
    for m in sorted(report.per_class, key=lambda m: m.f1):
        if m.support:
            table.add_row(
                m.institution_id, str(m.support),
                f"{m.precision:.2f}", f"{m.recall:.2f}", f"{m.f1:.2f}",
            )
    console.print(table)

    if show_confusion:
        for gold, row in sorted(report.confusion.items()):
            wrong = {k: v for k, v in row.items() if k != gold}
            if wrong:
                console.print(f"[yellow]{gold}[/yellow] -> {wrong}")

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    console.print(f"[green]Wrote report[/green] to {out}")


@batch_app.command("submit")
def batch_submit(
    dataset: Path = typer.Argument(DATA / "generated" / "mock.jsonl"),
    model: str = typer.Option(None),
    effort: str = typer.Option(None),
    limit: int = typer.Option(0),
) -> None:
    """Submit a dataset to the Batches API and print the batch id."""
    _require_key()
    docs = mockdata.load_jsonl(dataset) if dataset.suffix == ".jsonl" else ingest.load_directory(dataset)
    if limit:
        docs = docs[:limit]
    clf = LLMClassifier(model=model, effort=effort)
    batch_id = clf.submit_batch(docs)
    console.print(f"[green]Submitted {len(docs)} documents.[/green]")
    console.print(f"batch id: [cyan]{batch_id}[/cyan]")
    console.print(f"collect with: docrouter batch collect {batch_id}")


@batch_app.command("collect")
def batch_collect(
    batch_id: str = typer.Argument(...),
    out: Path = typer.Option(RUNS / "predictions.jsonl"),
) -> None:
    """Check a batch and write its predictions once it has ended."""
    _require_key()
    clf = LLMClassifier()
    status = clf.batch_status(batch_id)
    console.print(f"status: {status.processing_status}")
    if status.processing_status != "ended":
        console.print("[yellow]Not finished yet — try again shortly.[/yellow]")
        raise typer.Exit(0)

    predictions, errors = clf.collect_batch(batch_id)
    _write_predictions(predictions, out)
    console.print(f"[green]Wrote {len(predictions)} predictions[/green] to {out}")
    if errors:
        console.print(f"[yellow]{len(errors)} failed:[/yellow] {json.dumps(errors, ensure_ascii=False)[:400]}")


@app.command("threshold")
def threshold_sweep(
    dataset: Path = typer.Option(DATA / "generated" / "mock.jsonl", help="labeled documents"),
    predictions: Path = typer.Option(RUNS / "predictions.jsonl"),
    target_auto_accuracy: float = typer.Option(
        0.95, help="required accuracy on anything auto-routed"
    ),
    misroute_cost: float = typer.Option(20.0, help="cost of one misrouted document"),
    review_cost: float = typer.Option(1.0, help="cost of one human review"),
    per_class: bool = typer.Option(False, help="also fit a cut-off per institution"),
    validate: bool = typer.Option(True, help="pick on one half, verify on the other"),
    out: Path = typer.Option(RUNS / "threshold.json"),
) -> None:
    """Sweep the auto-route cut-off and recommend one."""
    from .models import Prediction
    from .threshold import (
        calibration, per_class_thresholds, recommend_for_target,
        recommend_min_cost, split_validate, sweep,
    )

    tax = load_taxonomy()
    docs = mockdata.load_jsonl(dataset)
    with predictions.open(encoding="utf-8") as fh:
        preds = [Prediction.model_validate_json(l) for l in fh if l.strip()]

    # --- is the confidence score worth thresholding at all? ---
    cal = calibration(docs, preds)
    console.print("[bold]Calibration[/bold]")
    console.print(
        f"  mean confidence {cal.mean_confidence:.2f} vs accuracy {cal.accuracy:.2f}"
        f"   ECE {cal.ece:.3f}  ->  [cyan]{cal.verdict}[/cyan]"
    )
    if cal.ece >= 0.15:
        console.print(
            "  [yellow]Confidence tracks reality poorly; any cut-off here is "
            "unreliable. Fix calibration before trusting a threshold.[/yellow]"
        )

    bins = Table(title="Reliability")
    bins.add_column("confidence", style="cyan")
    bins.add_column("n", justify="right")
    bins.add_column("predicted", justify="right")
    bins.add_column("actual", justify="right")
    bins.add_column("gap", justify="right")
    for b in cal.bins:
        colour = "green" if abs(b.gap) < 0.1 else "yellow"
        bins.add_row(
            f"{b.low:.1f}-{b.high:.1f}", str(b.n),
            f"{b.mean_confidence:.2f}", f"{b.accuracy:.2f}",
            f"[{colour}]{b.gap:+.2f}[/{colour}]",
        )
    console.print(bins)

    # --- the sweep ---
    points = sweep(docs, preds, misroute_cost=misroute_cost, review_cost=review_cost)
    cheapest = recommend_min_cost(points)
    on_target = recommend_for_target(points, target_auto_accuracy)

    table = Table(title=f"Threshold sweep (n={points[0].auto_routed + points[0].held})")
    table.add_column("t", justify="right", style="cyan")
    table.add_column("auto", justify="right")
    table.add_column("coverage", justify="right")
    table.add_column("auto acc", justify="right")
    table.add_column("misrouted", justify="right")
    table.add_column("held", justify="right")
    table.add_column("cost", justify="right")
    table.add_column("pick", no_wrap=True)
    for p in points:
        marks = []
        if on_target and p.threshold == on_target.threshold:
            marks.append("[green]target[/green]")
        if p.threshold == cheapest.threshold:
            marks.append("[magenta]cheapest[/magenta]")
        table.add_row(
            f"{p.threshold:.2f}", str(p.auto_routed), f"{p.coverage:.0%}",
            f"{p.auto_accuracy:.1%}" if p.defined else "—",
            f"{p.misrouted}", str(p.held), f"{p.expected_cost:.0f}",
            " · ".join(marks),
        )
    console.print(table)

    console.print("[bold]Recommendation[/bold]")
    if on_target:
        console.print(
            f"  target mode : [green]{on_target.threshold:.2f}[/green] — "
            f"{on_target.auto_accuracy:.1%} accurate on {on_target.coverage:.0%} of "
            f"documents, {on_target.misrouted} misrouted, {on_target.held} held"
        )
    else:
        console.print(
            f"  target mode : [red]no threshold reaches {target_auto_accuracy:.0%}[/red] "
            "on this data — the model cannot support that SLA here at any cut-off"
        )
    console.print(
        f"  cost mode   : [magenta]{cheapest.threshold:.2f}[/magenta] — "
        f"expected cost {cheapest.expected_cost:.0f} "
        f"(misroute={misroute_cost:g}, review={review_cost:g})"
    )

    # --- did the choice just fit this data? ---
    validation = None
    if validate:
        validation = split_validate(
            docs, preds, target_auto_accuracy=target_auto_accuracy,
            misroute_cost=misroute_cost, review_cost=review_cost,
        )
        console.print("")
        console.print("[bold]Held-out check[/bold]")
        if validation is None:
            console.print("  [yellow]Not enough labeled data to split.[/yellow]")
        else:
            console.print(
                f"  picked {validation.chosen_threshold:.2f} on {validation.n_train} docs; "
                f"on the {validation.n_test} it never saw: "
                f"{validation.test_auto_accuracy:.1%} accurate at "
                f"{validation.test_coverage:.0%} coverage "
                f"({validation.test_misrouted} misrouted)"
            )
            if validation.optimism > 0.05:
                console.print(
                    f"  [yellow]Optimism {validation.optimism:+.1%} — the sweep "
                    "flattered itself. Treat the headline as an upper bound.[/yellow]"
                )

    per_class_rows = None
    if per_class:
        per_class_rows = per_class_thresholds(
            docs, preds, target_auto_accuracy=target_auto_accuracy, taxonomy=tax
        )
        pc = Table(title=f"Per-class cut-offs (target {target_auto_accuracy:.0%})")
        pc.add_column("institution", style="cyan")
        pc.add_column("n", justify="right")
        pc.add_column("t", justify="right")
        pc.add_column("coverage", justify="right")
        pc.add_column("")
        for row in per_class_rows:
            pc.add_row(
                row.institution_id, str(row.support),
                f"{row.threshold:.2f}" if row.threshold is not None else "[red]none[/red]",
                f"{row.coverage:.0%}" if row.threshold is not None else "—",
                "[yellow]thin[/yellow]" if row.thin else "",
            )
        console.print(pc)
        console.print(
            "  [dim]'thin' means too few documents to set a trustworthy cut-off; "
            "fall back to the global one there.[/dim]"
        )

    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "calibration": cal.model_dump() | {"verdict": cal.verdict},
        "sweep": [p.model_dump() for p in points],
        "recommended_target": on_target.model_dump() if on_target else None,
        "recommended_cost": cheapest.model_dump(),
        "validation": validation.model_dump() if validation else None,
        "per_class": [r.model_dump() for r in per_class_rows] if per_class_rows else None,
        "costs": {"misroute": misroute_cost, "review": review_cost},
    }
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    console.print(f"\n[green]Wrote sweep[/green] to {out}")
    if on_target:
        console.print(
            f"apply it with: docrouter classify <path> --threshold {on_target.threshold:.2f}"
        )


LABELS = DATA / "labels" / "labels.jsonl"
QUEUE = DATA / "labels" / "queue.jsonl"


@label_app.command("prelabel")
def label_prelabel(
    path: Path = typer.Argument(..., help="folder of real documents, or a .jsonl dataset"),
    backend: str = typer.Option(
        "llm", help="llm | baseline | none (build a queue with no predictions)"
    ),
    size: int = typer.Option(50, help="documents in the review batch"),
    random_ratio: float = typer.Option(
        0.2, help="share drawn uniformly at random — keep above 0 for an honest estimate"
    ),
    per_class_cap: int = typer.Option(0, help="max per predicted class (0 = uncapped)"),
    ocr: str = typer.Option("claude", help="OCR backend for scanned input"),
    compare_baseline: bool = typer.Option(
        True, help="also run the offline baseline, to use disagreement as a signal"
    ),
    labels: Path = typer.Option(LABELS),
    out: Path = typer.Option(QUEUE),
) -> None:
    """Ingest real documents, pre-label them, and build a review queue."""
    from .labeling import build_queue
    from .labeling.review import save_queue
    from .labeling.store import LabelStore

    tax = load_taxonomy()
    store = LabelStore(labels)
    done = store.labeled_ids()

    if path.suffix == ".jsonl":
        docs = mockdata.load_jsonl(path)
    elif path.is_dir():
        docs = ingest.load_directory(path, ocr_backend=ocr)
    else:
        docs = [ingest.load_document(path, ocr_backend=ocr)]

    fresh = [d for d in docs if d.doc_id not in done]
    console.print(
        f"ingested {len(docs)} document(s); {len(fresh)} not yet reviewed"
    )
    if not fresh:
        console.print("[yellow]Nothing left to review.[/yellow]")
        raise typer.Exit(0)

    predictions = None
    baseline_predictions = None

    if backend == "llm":
        _require_key()
        clf = LLMClassifier(taxonomy=tax)
        with console.status(f"Pre-labeling {len(fresh)} document(s) with {clf.name}..."):
            predictions = clf.classify_many(fresh)
    elif backend == "baseline":
        predictions = _load_baseline(tax).classify_many(fresh)
    elif backend != "none":
        console.print(f"[red]unknown backend {backend!r}[/red]")
        raise typer.Exit(1)

    if compare_baseline and backend == "llm":
        model_path = MODELS / "baseline.joblib"
        if model_path.exists():
            baseline_predictions = (
                BaselineClassifier(taxonomy=tax).load(model_path).classify_many(fresh)
            )
        else:
            console.print(
                "[yellow]No trained baseline — skipping the disagreement signal.[/yellow]"
            )

    items = build_queue(
        fresh,
        predictions=predictions,
        baseline_predictions=baseline_predictions,
        already_labeled=done,
        taxonomy=tax,
        size=size,
        random_ratio=random_ratio,
        per_class_cap=per_class_cap or None,
    )
    save_queue(items, out)

    lanes = Counter(i.lane for i in items)
    console.print(f"[green]Queued {len(items)} document(s)[/green] to {out}")
    console.print(f"  priority lane: {lanes.get('priority', 0)}   random lane: {lanes.get('random', 0)}")
    console.print("review with: docrouter label review")


@label_app.command("review")
def label_review(
    queue: Path = typer.Option(QUEUE),
    labels: Path = typer.Option(LABELS),
    reviewer: str = typer.Option(..., prompt="اسم المراجع / reviewer name"),
    port: int = typer.Option(8765),
    blind_random: bool = typer.Option(
        True, help="hide the model's guess on random-lane documents (avoids anchoring)"
    ),
    open_browser: bool = typer.Option(True),
) -> None:
    """Open the local review UI. Nothing leaves this machine."""
    from .labeling.review import load_queue, serve
    from .labeling.store import LabelStore

    if not queue.exists():
        console.print(f"[red]No queue at {queue}.[/red] Run `docrouter label prelabel` first.")
        raise typer.Exit(1)

    store = LabelStore(labels)
    done = store.labeled_ids()
    items = [i for i in load_queue(queue) if i.doc_id not in done]
    if not items:
        console.print("[yellow]Every document in this queue is already reviewed.[/yellow]")
        raise typer.Exit(0)

    console.print(f"[green]Serving {len(items)} document(s)[/green] at http://127.0.0.1:{port}/")
    console.print("Loopback only — no document leaves this machine. Ctrl+C to stop.")
    serve(
        items, store, reviewer=reviewer, port=port,
        blind_random=blind_random, open_browser=open_browser,
    )
    console.print("\n[green]Session ended.[/green] Run `docrouter label status` for totals.")


@label_app.command("status")
def label_status(
    labels: Path = typer.Option(LABELS),
    target_per_class: int = typer.Option(25, help="labels per institution you're aiming for"),
) -> None:
    """Progress, agreement rates, and what still needs labeling."""
    from .labeling.store import LabelStore

    tax = load_taxonomy()
    stats = LabelStore(labels).stats()

    if not stats.total_records:
        console.print(f"[yellow]No labels yet at {labels}.[/yellow]")
        raise typer.Exit(0)

    console.print(f"records (incl. corrections): {stats.total_records}")
    console.print(
        f"labeled: {stats.labeled}   skipped: {stats.skipped}   unclear: {stats.unclear}"
    )
    if stats.median_seconds:
        console.print(f"median time per document: {stats.median_seconds:.0f}s")

    console.print("")
    console.print("[bold]Model agreement[/bold]")

    if stats.random.n:
        lo, hi = stats.random.wilson_interval()
        console.print(
            f"  random lane   : {stats.random.agreement:.1%} "
            f"(n={stats.random.n}, 95% CI {lo:.0%}-{hi:.0%})   <- the honest estimate"
        )
        if stats.random.n < 30:
            console.print(
                "  [yellow]Random-lane sample is small; treat the estimate as provisional.[/yellow]"
            )
    else:
        console.print(
            "  random lane   : [dim]no data yet[/dim]"
            "   <- the honest estimate comes from here"
        )

    if stats.priority.n:
        console.print(
            f"  priority lane : {stats.priority.agreement:.1%} (n={stats.priority.n})"
            "   <- hard cases by design; expected to be lower"
        )
    else:
        console.print("  priority lane : [dim]no data yet[/dim]")

    table = Table(title=f"Coverage (target {target_per_class}/class)")
    table.add_column("institution", style="cyan")
    table.add_column("الجهة", justify="right")
    table.add_column("have", justify="right")
    table.add_column("need", justify="right")
    for institution_id in tax.ids:
        have = stats.per_class.get(institution_id, 0)
        need = max(0, target_per_class - have)
        table.add_row(
            institution_id, tax.name_ar(institution_id), str(have),
            "[green]0[/green]" if not need else f"[yellow]{need}[/yellow]",
        )
    console.print(table)

    if stats.reviewers:
        console.print("reviewers: " + ", ".join(f"{k} ({v})" for k, v in stats.reviewers.items()))


@label_app.command("export")
def label_export(
    source: Path = typer.Argument(
        ..., help="folder or .jsonl the documents were ingested from"
    ),
    labels: Path = typer.Option(LABELS),
    out: Path = typer.Option(DATA / "gold" / "gold.jsonl"),
    ocr: str = typer.Option("claude"),
) -> None:
    """Write a gold dataset that `evaluate` and `train-baseline` can consume."""
    from .labeling.store import LabelStore

    gold = LabelStore(labels).gold()
    if not gold:
        console.print("[yellow]No confirmed labels to export yet.[/yellow]")
        raise typer.Exit(0)

    if source.suffix == ".jsonl":
        docs = mockdata.load_jsonl(source)
    else:
        docs = ingest.load_directory(source, ocr_backend=ocr)

    labeled = []
    for doc in docs:
        if doc.doc_id in gold:
            doc.true_label = gold[doc.doc_id]
            labeled.append(doc)

    missing = set(gold) - {d.doc_id for d in labeled}
    mockdata.save_jsonl(labeled, out)
    console.print(f"[green]Exported {len(labeled)} labeled document(s)[/green] to {out}")
    if missing:
        console.print(
            f"[yellow]{len(missing)} label(s) had no matching document in {source}[/yellow]"
        )
    console.print(f"evaluate against it with: docrouter evaluate --dataset {out}")


if __name__ == "__main__":
    app()
