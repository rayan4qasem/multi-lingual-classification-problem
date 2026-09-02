"""TF-IDF + linear SVM baseline.

Its job is not to beat the LLM. It exists so that every LLM accuracy number
has something to be measured against, it runs offline with no API cost, and
it is the natural starting point if the deployment ever has to move on-prem.

Character n-grams do most of the work here: Arabic is morphologically rich,
and 2-5 char grams survive the prefixes and clitics that split word-level
features across a dozen surface forms.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline, make_union
from sklearn.svm import LinearSVC

from .. import normalize
from ..models import Alternative, Document, Prediction
from ..taxonomy import Taxonomy, load as load_taxonomy


def _build_pipeline() -> Pipeline:
    features = make_union(
        TfidfVectorizer(
            analyzer="char_wb", ngram_range=(2, 5), min_df=2, max_features=200_000,
            sublinear_tf=True,
        ),
        TfidfVectorizer(
            analyzer="word", ngram_range=(1, 2), min_df=2, max_features=100_000,
            sublinear_tf=True,
        ),
    )
    # LinearSVC has no predict_proba; calibration gives us confidences on the
    # same 0-1 scale the LLM backend reports, so the two are comparable.
    return Pipeline(
        [
            ("features", features),
            ("clf", CalibratedClassifierCV(LinearSVC(C=1.0), cv=3, method="sigmoid")),
        ]
    )


class BaselineClassifier:
    def __init__(
        self,
        taxonomy: Taxonomy | None = None,
        review_threshold: float = 0.55,
    ):
        self.taxonomy = taxonomy or load_taxonomy()
        self.review_threshold = review_threshold
        self.pipeline: Pipeline | None = None

    @property
    def name(self) -> str:
        return "baseline:tfidf-svm"

    def fit(self, docs: list[Document]) -> None:
        labels = [d.true_label for d in docs]
        if any(l is None for l in labels):
            raise ValueError("every training document needs a true_label")
        texts = [normalize.aggressive(d.text) for d in docs]
        self.pipeline = _build_pipeline()
        self.pipeline.fit(texts, labels)

    def classify(self, doc: Document) -> Prediction:
        return self.classify_many([doc])[0]

    def classify_many(self, docs) -> list[Prediction]:
        if self.pipeline is None:
            raise RuntimeError("baseline is untrained — call fit() or load() first")
        docs = list(docs)
        texts = [normalize.aggressive(d.text) for d in docs]
        probabilities = self.pipeline.predict_proba(texts)
        classes = self.pipeline.classes_

        predictions = []
        for doc, row in zip(docs, probabilities):
            order = np.argsort(row)[::-1]
            top = order[0]
            predictions.append(
                Prediction(
                    doc_id=doc.doc_id,
                    institution_id=str(classes[top]),
                    confidence=float(row[top]),
                    rationale_ar="تصنيف إحصائي دون تعليل",
                    alternatives=[
                        Alternative(
                            institution_id=str(classes[i]), confidence=float(row[i])
                        )
                        for i in order[1:3]
                    ],
                    needs_review=float(row[top]) < self.review_threshold,
                    backend=self.name,
                )
            )
        return predictions

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.pipeline, path)

    def load(self, path: str | Path) -> "BaselineClassifier":
        self.pipeline = joblib.load(Path(path))
        return self
