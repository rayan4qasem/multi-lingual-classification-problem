"""The SOLID contracts themselves.

These are the tests that would fail if a refactor quietly broke
substitutability or re-introduced a dispatch chain.
"""

from __future__ import annotations

import pytest

from docrouter import mockdata, taxonomy
from docrouter.classify import (
    BaselineClassifier,
    ClassifierRegistry,
    LLMClassifier,
    MissingModel,
    UnknownBackend,
    available_backends,
    create_classifier,
)
from docrouter.protocols import BatchClassifier, Classifier


class StubClient:
    """Stands in for anthropic.Anthropic so no key is needed."""


def _trained_baseline():
    docs = mockdata.generate_templates(n_per_class=6, seed=2, ocr_noise_ratio=0.0)
    clf = BaselineClassifier()
    clf.fit(docs)
    return clf, docs


# ---------- Liskov: the backends really are substitutable ----------


def test_both_backends_satisfy_the_classifier_protocol():
    baseline, _ = _trained_baseline()
    assert isinstance(baseline, Classifier)
    assert isinstance(LLMClassifier(client=StubClient()), Classifier)


def test_backends_are_interchangeable_at_the_call_site():
    baseline, docs = _trained_baseline()
    sample = docs[:4]
    tax = taxonomy.load()

    def route(classifier: Classifier):
        """A caller that knows only the protocol."""
        predictions = classifier.classify_many(sample)
        assert len(predictions) == len(sample)
        for prediction in predictions:
            assert prediction.institution_id in tax.ids
            assert 0.0 <= prediction.confidence <= 1.0
            assert prediction.backend == classifier.name
        return predictions

    route(baseline)  # the LLM path is covered by its own stubbed tests


def test_classify_and_classify_many_agree():
    baseline, docs = _trained_baseline()
    one = baseline.classify(docs[0])
    many = baseline.classify_many([docs[0]])[0]
    assert one.model_dump() == many.model_dump()


# ---------- Interface segregation ----------


def test_only_the_llm_backend_implements_the_batch_protocol():
    baseline, _ = _trained_baseline()
    llm = LLMClassifier(client=StubClient())

    assert isinstance(llm, BatchClassifier)
    # The baseline has no notion of an async batch and is not forced to fake
    # one just to satisfy a single fat interface.
    assert not isinstance(baseline, BatchClassifier)
    assert isinstance(baseline, Classifier)


# ---------- Open/closed: the classifier registry ----------


def test_registry_lists_the_bundled_backends():
    assert set(available_backends()) == {"llm", "baseline"}


def test_unknown_backend_raises_a_typed_error():
    with pytest.raises(UnknownBackend, match="unknown backend"):
        create_classifier("does-not-exist")


def test_missing_model_artifact_raises_a_typed_error(tmp_path):
    with pytest.raises(MissingModel, match="train-baseline"):
        create_classifier("baseline", model_path=tmp_path / "absent.joblib")


def test_a_new_backend_needs_no_change_to_the_factory():
    class ConstantClassifier:
        """A third backend, e.g. a future on-prem fine-tune."""

        name = "constant"

        def classify(self, doc):
            return self.classify_many([doc])[0]

        def classify_many(self, docs):
            from docrouter.models import Prediction

            return [
                Prediction(
                    doc_id=d.doc_id,
                    institution_id="moj_courts",
                    confidence=1.0,
                    backend=self.name,
                )
                for d in docs
            ]

    registry = ClassifierRegistry()
    registry.register("constant", lambda **kw: ConstantClassifier())

    classifier = registry.create("constant", taxonomy=taxonomy.load(), anything="ignored")
    assert isinstance(classifier, Classifier)
    assert classifier.classify_many(mockdata.generate_curated(seed=1)[:2])[0].backend == "constant"


def test_factories_tolerate_arguments_they_do_not_need():
    # Callers should not have to know which backend they are building.
    clf = create_classifier("llm", client=StubClient(), model_path="ignored", nonsense=123)
    assert isinstance(clf, Classifier)


# ---------- Single responsibility: rendering is separable ----------


def test_reporting_renders_without_a_console_or_cli():
    from rich.table import Table

    from docrouter import reporting
    from docrouter.evaluate import evaluate

    baseline, docs = _trained_baseline()
    tax = taxonomy.load()
    predictions = baseline.classify_many(docs[:5])

    assert isinstance(reporting.institutions_table(tax), Table)
    assert isinstance(reporting.routing_table(predictions, tax), Table)
    assert isinstance(reporting.per_class_table(evaluate(docs[:5], predictions)), Table)


def test_routing_table_respects_its_limit():
    baseline, docs = _trained_baseline()
    from docrouter import reporting

    table = reporting.routing_table(baseline.classify_many(docs[:20]), taxonomy.load(), limit=5)
    assert table.row_count == 5


# ---------- regressions found by the end-to-end sweep ----------


def test_every_ingesting_command_accepts_a_model_path():
    """`label prelabel` hardcoded models/baseline.joblib while `classify`
    took --model-path, so a baseline trained anywhere else was unreachable
    from the labeling loop."""
    import inspect

    from docrouter import cli

    for command in (cli.classify, cli.label_prelabel):
        params = inspect.signature(command).parameters
        assert "model_path" in params, f"{command.__name__} lacks --model-path"


def test_cli_ingestion_is_funnelled_through_one_helper():
    """Ingestion errors are translated in _load_documents; call sites that
    bypass it would reintroduce raw tracebacks."""
    import inspect

    from docrouter import cli

    source = inspect.getsource(cli)
    body = source.split("def _load_documents", 1)[1]
    after_helper = body.split("def _write_predictions", 1)[1]
    assert "ingest.load_directory" not in after_helper
    assert "ingest.load_document" not in after_helper
