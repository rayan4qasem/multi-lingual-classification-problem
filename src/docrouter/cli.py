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

from . import ingest, mockdata, reporting
from .classify import (
    BaselineClassifier,
    MissingModel,
    UnknownBackend,
    available_backends,
    create_classifier,
)
from .evaluate import evaluate as run_evaluate
from .protocols import BatchClassifier, Classifier
from .taxonomy import load as load_taxonomy

app = typer.Typer(add_completion=False, help="Arabic government document routing.")
batch_app = typer.Typer(help="Bulk routing through the Batches API (50% cost).")
label_app = typer.Typer(help="Human-in-the-loop labeling of real documents.")
prompt_app = typer.Typer(help="Few-shot examples built from confirmed labels.")
app.add_typer(batch_app, name="batch")
app.add_typer(label_app, name="label")
app.add_typer(prompt_app, name="prompt")

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


def _make_classifier(backend: str, **kwargs) -> Classifier:
    """Build a classifier by name, turning registry errors into clean exits."""
    try:
        return create_classifier(backend, **kwargs)
    except UnknownBackend:
        console.print(
            f"[red]Unknown backend {backend!r}.[/red] Available: {', '.join(available_backends())}"
        )
        raise typer.Exit(1) from None
    except MissingModel as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from None


def _make_batch_classifier(backend: str = "llm", **kwargs) -> BatchClassifier:
    """Build a classifier that can also route in bulk.

    Batching is a narrower contract than classification, so it is checked
    rather than assumed — the offline baseline is a perfectly good
    `Classifier` and deliberately not a `BatchClassifier`.
    """
    clf = _make_classifier(backend, **kwargs)
    if not isinstance(clf, BatchClassifier):
        console.print(
            f"[red]Backend {backend!r} does not support batching.[/red] "
            "Use `docrouter classify` instead."
        )
        raise typer.Exit(1)
    return clf


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
    console.print(reporting.institutions_table(tax))
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
        f"[green]Wrote {len(docs)} documents[/green] to {out} ({scanned} carry simulated OCR noise)"
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
    examples: Path = typer.Option(None, help="few-shot example set from `prompt build`"),
    model_path: Path = typer.Option(
        MODELS / "baseline.joblib", help="trained artifact; baseline backend only"
    ),
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

    example_set = None
    if examples:
        from . import fewshot

        example_set = fewshot.load(examples)
        leaked = fewshot.check_leakage(example_set, docs)
        if leaked:
            # Scoring a model on documents sitting in its own prompt inflates
            # the result, so this is worth shouting about.
            console.print(
                f"[yellow]{len(leaked)} document(s) are in both the example "
                "set and this run — their scores are not meaningful. "
                "Exclude them before quoting any accuracy.[/yellow]"
            )
        console.print(f"using {len(example_set.examples)} few-shot example(s) from {examples}")

    if backend == "llm":
        _require_key()

    clf = _make_classifier(
        backend,
        taxonomy=tax,
        model=model,
        effort=effort,
        review_threshold=threshold,
        examples=example_set,
        model_path=model_path,
    )

    with console.status(f"Classifying {len(docs)} document(s) via {clf.name}..."):
        predictions = clf.classify_many(docs)

    console.print(reporting.routing_table(predictions, tax))
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

    console.print(reporting.per_class_table(report))

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
    docs = (
        mockdata.load_jsonl(dataset)
        if dataset.suffix == ".jsonl"
        else ingest.load_directory(dataset)
    )
    if limit:
        docs = docs[:limit]
    clf = _make_batch_classifier("llm", model=model, effort=effort)
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
    clf = _make_batch_classifier("llm")
    status = clf.batch_status(batch_id)
    console.print(f"status: {status.processing_status}")
    if status.processing_status != "ended":
        console.print("[yellow]Not finished yet — try again shortly.[/yellow]")
        raise typer.Exit(0)

    predictions, errors = clf.collect_batch(batch_id)
    _write_predictions(predictions, out)
    console.print(f"[green]Wrote {len(predictions)} predictions[/green] to {out}")
    if errors:
        console.print(
            f"[yellow]{len(errors)} failed:[/yellow] {json.dumps(errors, ensure_ascii=False)[:400]}"
        )


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
        calibration,
        per_class_thresholds,
        recommend_for_target,
        recommend_min_cost,
        split_validate,
        sweep,
    )

    tax = load_taxonomy()
    docs = mockdata.load_jsonl(dataset)
    with predictions.open(encoding="utf-8") as fh:
        preds = [Prediction.model_validate_json(line) for line in fh if line.strip()]

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

    console.print(reporting.reliability_table(cal))

    # --- the sweep ---
    points = sweep(docs, preds, misroute_cost=misroute_cost, review_cost=review_cost)
    cheapest = recommend_min_cost(points)
    on_target = recommend_for_target(points, target_auto_accuracy)

    console.print(reporting.sweep_table(points, on_target, cheapest))

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
            docs,
            preds,
            target_auto_accuracy=target_auto_accuracy,
            misroute_cost=misroute_cost,
            review_cost=review_cost,
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
        console.print(reporting.class_thresholds_table(per_class_rows, target_auto_accuracy))
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
    console.print(f"ingested {len(docs)} document(s); {len(fresh)} not yet reviewed")
    if not fresh:
        console.print("[yellow]Nothing left to review.[/yellow]")
        raise typer.Exit(0)

    predictions = None
    baseline_predictions = None

    if backend != "none":
        if backend == "llm":
            _require_key()
        clf = _make_classifier(backend, taxonomy=tax, model_path=MODELS / "baseline.joblib")
        with console.status(f"Pre-labeling {len(fresh)} document(s) with {clf.name}..."):
            predictions = clf.classify_many(fresh)

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
    console.print(
        f"  priority lane: {lanes.get('priority', 0)}   random lane: {lanes.get('random', 0)}"
    )
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
        items,
        store,
        reviewer=reviewer,
        port=port,
        blind_random=blind_random,
        open_browser=open_browser,
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
    console.print(f"labeled: {stats.labeled}   skipped: {stats.skipped}   unclear: {stats.unclear}")
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
            "  random lane   : [dim]no data yet[/dim]   <- the honest estimate comes from here"
        )

    if stats.priority.n:
        console.print(
            f"  priority lane : {stats.priority.agreement:.1%} (n={stats.priority.n})"
            "   <- hard cases by design; expected to be lower"
        )
    else:
        console.print("  priority lane : [dim]no data yet[/dim]")

    console.print(reporting.coverage_table(stats, tax, target_per_class))

    if stats.reviewers:
        console.print("reviewers: " + ", ".join(f"{k} ({v})" for k, v in stats.reviewers.items()))


@label_app.command("export")
def label_export(
    source: Path = typer.Argument(..., help="folder or .jsonl the documents were ingested from"),
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


EXAMPLES = DATA / "prompt" / "examples.json"


@prompt_app.command("build")
def prompt_build(
    source: Path = typer.Argument(..., help="folder or .jsonl the labeled documents live in"),
    labels: Path = typer.Option(LABELS, help="label store; omit to use dataset labels"),
    predictions: Path = typer.Option(
        None, help="model predictions, so human overrides can be preferred"
    ),
    per_class: int = typer.Option(1, help="guaranteed examples per institution"),
    max_examples: int = typer.Option(20),
    max_chars: int = typer.Option(700, help="characters kept per example"),
    redact: bool = typer.Option(True, help="mask IDs, phones, IBANs, tax numbers, emails"),
    ocr: str = typer.Option("claude"),
    out: Path = typer.Option(EXAMPLES),
) -> None:
    """Select few-shot examples from confirmed labels."""
    from . import fewshot
    from .models import Prediction

    tax = load_taxonomy()

    if source.suffix == ".jsonl":
        docs = mockdata.load_jsonl(source)
    else:
        docs = ingest.load_directory(source, ocr_backend=ocr)

    gold: dict[str, str] = {}
    if labels.exists():
        from .labeling.store import LabelStore

        gold = LabelStore(labels).gold()
    if not gold:
        gold = {d.doc_id: d.true_label for d in docs if d.true_label}
        console.print(
            f"[yellow]No label store at {labels}; using the dataset's own labels.[/yellow]"
        )
    if not gold:
        console.print("[red]No confirmed labels to build examples from.[/red]")
        raise typer.Exit(1)

    model_labels: dict[str, str] = {}
    if predictions and predictions.exists():
        with predictions.open(encoding="utf-8") as fh:
            model_labels = {
                p.doc_id: p.institution_id
                for p in (Prediction.model_validate_json(line) for line in fh if line.strip())
            }

    example_set = fewshot.select_examples(
        docs,
        gold,
        model_labels=model_labels,
        taxonomy=tax,
        per_class=per_class,
        max_examples=max_examples,
        max_chars=max_chars,
        do_redact=redact,
    )
    fewshot.save(example_set, out)

    overrides = sum(1 for e in example_set.examples if e.is_override)
    covered = len(example_set.per_class())
    console.print(f"[green]Selected {len(example_set.examples)} example(s)[/green] to {out}")
    console.print(
        f"  covering {covered}/{len(tax.ids)} institutions; {overrides} are human corrections"
    )
    if not redact:
        console.print(
            "[yellow]Redaction is off — real identifiers will be sent on every "
            "request. Only do this if policy allows it.[/yellow]"
        )
    missing = [i for i in tax.ids if i not in example_set.per_class()]
    if missing:
        console.print(f"  [yellow]no example yet for:[/yellow] {', '.join(missing)}")
    console.print("inspect with: docrouter prompt show")


@prompt_app.command("show")
def prompt_show(
    examples: Path = typer.Option(EXAMPLES),
    full: bool = typer.Option(False, help="print the whole system prompt"),
    count_tokens: bool = typer.Option(False, help="ask the API for an exact token count"),
) -> None:
    """Inspect the example set and what it costs in prompt tokens."""
    from . import fewshot
    from .classify.llm import SYSTEM_PREAMBLE

    tax = load_taxonomy()
    if not examples.exists():
        console.print(f"[red]No example set at {examples}.[/red] Run `docrouter prompt build`.")
        raise typer.Exit(1)

    example_set = fewshot.load(examples)
    base = SYSTEM_PREAMBLE + tax.render_for_prompt()
    block = fewshot.render(example_set, tax)

    console.print(reporting.examples_table(example_set, tax))

    console.print(f"base prompt    : {len(base):,} chars")
    console.print(f"examples block : {len(block):,} chars (+{len(block) / max(len(base), 1):.0%})")

    if count_tokens:
        _require_key()
        import anthropic

        client = anthropic.Anthropic()
        # A one-token user turn, so the delta measured is the system prompt.
        probe: list[anthropic.types.MessageParam] = [{"role": "user", "content": "x"}]
        without = client.messages.count_tokens(
            model=os.environ.get("DOCROUTER_MODEL", "claude-opus-5"),
            system=base,
            messages=probe,
        ).input_tokens
        with_examples = client.messages.count_tokens(
            model=os.environ.get("DOCROUTER_MODEL", "claude-opus-5"),
            system=base + block,
            messages=probe,
        ).input_tokens
        console.print(
            f"tokens         : {without:,} -> {with_examples:,} (+{with_examples - without:,})"
        )
        console.print(
            "[dim]Cached after the first call, so this is paid once per cache "
            "window, not per document.[/dim]"
        )
    else:
        console.print("[dim]Pass --count-tokens for an exact count (needs an API key).[/dim]")

    if full:
        console.print("")
        console.print(base + block)
    elif block:
        console.print("")
        console.print(block[:1200] + ("\n…" if len(block) > 1200 else ""))


if __name__ == "__main__":
    app()
