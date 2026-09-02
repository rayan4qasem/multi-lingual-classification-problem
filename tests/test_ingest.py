"""Ingestion, extractor registry, and OCR backend selection. All offline."""

from __future__ import annotations

from pathlib import Path

import pytest

from docrouter import ingest
from docrouter.ingest import (
    ClaudeVisionOcr,
    Extraction,
    ExtractorRegistry,
    ImageExtractor,
    OcrRegistry,
    OCRUnavailable,
    PdfExtractor,
    PlainTextExtractor,
    TesseractOcr,
    UnsupportedDocument,
    default_registry,
)
from docrouter.protocols import OcrBackend, TextExtractor

ARABIC = "بلاغ عن سرقة مركبة من أمام المسكن بحي الروضة، أرجو تسجيل البلاغ."


class RecordingOcr:
    """A stand-in OCR backend that records what it was asked to transcribe."""

    def __init__(self, text: str = "نص مستخرج ضوئياً من الصفحة"):
        self.text = text
        self.calls: list[int] = []

    @property
    def name(self) -> str:
        return "recording"

    def transcribe(self, images):
        self.calls.append(len(images))
        return self.text


def _write_pdf(path: Path, pages: list[str]) -> Path:
    """Build a real PDF with a genuine text layer."""
    import fitz

    doc = fitz.open()
    for text in pages:
        page = doc.new_page()
        if text:
            page.insert_text((72, 72), text, fontsize=12)
        doc_font = None  # noqa: F841  (kept explicit: no font embedding needed)
    doc.save(path)
    doc.close()
    return path


# ---------- protocol conformance ----------


def test_bundled_backends_satisfy_the_ocr_protocol():
    assert isinstance(ClaudeVisionOcr(client=object()), OcrBackend)
    assert isinstance(TesseractOcr(), OcrBackend)
    assert isinstance(RecordingOcr(), OcrBackend)


def test_bundled_extractors_satisfy_the_extractor_protocol():
    for extractor in (PdfExtractor(), ImageExtractor(), PlainTextExtractor()):
        assert isinstance(extractor, TextExtractor)


# ---------- registry: the open/closed boundary ----------


def test_registry_dispatches_by_suffix():
    registry = default_registry()
    assert isinstance(registry.get(".txt"), PlainTextExtractor)
    assert isinstance(registry.get(".pdf"), PdfExtractor)
    assert registry.supports(".DOCX") is True


def test_unsupported_suffix_raises_a_typed_error():
    with pytest.raises(UnsupportedDocument) as exc:
        default_registry().get(".xyz")
    assert ".xyz" in str(exc.value)


def test_a_new_format_needs_no_change_to_ingest(tmp_path):
    # The whole point of the registry: extend without editing the module.
    class RtfExtractor:
        @property
        def suffixes(self):
            return frozenset({".rtf"})

        def extract(self, path, ocr=None):
            return Extraction(f"استُخرج من {path.name}", "digital", 1)

    registry = default_registry()
    registry.register(RtfExtractor())

    target = tmp_path / "letter.rtf"
    target.write_text("ignored", encoding="utf-8")

    doc = ingest.load_document(target, ocr_backend=None, registry=registry)
    assert "letter.rtf" in doc.text
    assert doc.source == "digital"


def test_registering_over_a_suffix_replaces_the_extractor():
    class Loud:
        @property
        def suffixes(self):
            return frozenset({".txt"})

        def extract(self, path, ocr=None):
            return Extraction("REPLACED", "digital", 1)

    registry = default_registry()
    registry.register(Loud())
    assert not isinstance(registry.get(".txt"), PlainTextExtractor)


def test_empty_registry_supports_nothing():
    assert ExtractorRegistry().suffixes == frozenset()


# ---------- OCR registry ----------


def test_ocr_registry_creates_known_backends():
    assert ingest.OCR_REGISTRY.create("tesseract").name == "tesseract"
    assert ingest.OCR_REGISTRY.create("claude", client=object()).name == "claude"
    assert ingest.OCR_REGISTRY.create("local", client=object()).name == "local"


def test_ocr_registry_rejects_unknown_names():
    with pytest.raises(ValueError, match="unknown OCR backend"):
        ingest.OCR_REGISTRY.create("nope")


def test_custom_ocr_backend_can_be_registered():
    registry = OcrRegistry()
    registry.register("recording", RecordingOcr)
    assert registry.create("recording").name == "recording"
    assert "recording" in registry.names


def test_resolve_ocr_accepts_instance_name_or_none():
    backend = RecordingOcr()
    assert ingest.resolve_ocr(backend) is backend
    assert ingest.resolve_ocr(None) is None
    assert ingest.resolve_ocr("tesseract").name == "tesseract"


def test_run_ocr_without_a_backend_is_a_typed_error():
    with pytest.raises(OCRUnavailable):
        ingest.run_ocr([b"x"], backend=None)


# ---------- plain text ----------


def test_plain_text_is_read_and_normalized(tmp_path):
    target = tmp_path / "doc.txt"
    target.write_text("الْحَمْدُ ـــ للهِ\n\n\n\nنص", encoding="utf-8")
    doc = ingest.load_document(target, ocr_backend=None)
    assert doc.source == "digital"
    assert "ـ" not in doc.text  # tatweel stripped by normalize.light
    assert doc.doc_id == "doc"
    assert doc.path == str(target)


# ---------- PDF: the per-page OCR decision ----------
#
# The default policy needs a real Arabic text layer, which PyMuPDF's built-in
# fonts cannot produce portably (no Arabic in the base-14 set, and relying on
# a system font would break on the Linux CI runners). So the policy itself is
# unit-tested directly, and the per-page routing is exercised through an
# injected policy - which is exactly what `is_usable` exists for.


def test_default_policy_accepts_a_real_arabic_text_layer():
    assert ingest.arabic_text_layer_is_usable(ARABIC) is True


@pytest.mark.parametrize(
    "text, reason",
    [
        ("", "empty"),
        ("قصير", "below the character floor"),
        ("Lorem ipsum dolor sit amet consectetur adipiscing elit sed do", "not Arabic"),
    ],
)
def test_default_policy_rejects_unusable_text_layers(text, reason):
    assert ingest.arabic_text_layer_is_usable(text) is False, reason


def test_digital_pdf_needs_no_ocr(tmp_path):
    latin_ok = ingest.PdfExtractor(is_usable=lambda t: len(t.strip()) >= 20)
    registry = ExtractorRegistry([latin_ok, PlainTextExtractor()])
    path = _write_pdf(tmp_path / "digital.pdf", ["Ministry of Justice case file 4471"])

    ocr = RecordingOcr()
    doc = ingest.load_document(path, ocr_backend=ocr, registry=registry)
    assert doc.source == "digital"
    assert ocr.calls == []  # never invoked
    assert doc.page_count == 1


def test_pdf_without_a_text_layer_is_sent_to_ocr(tmp_path):
    path = _write_pdf(tmp_path / "scanned.pdf", [""])
    ocr = RecordingOcr()
    doc = ingest.load_document(path, ocr_backend=ocr)
    assert doc.source == "ocr"
    assert ocr.calls == [1]
    assert "مستخرج" in doc.text


def test_mixed_pdf_ocrs_only_the_pages_that_need_it(tmp_path):
    # A digital cover sheet plus two scanned attachments - the realistic case,
    # and the reason the decision is per page rather than per file.
    latin_ok = ingest.PdfExtractor(is_usable=lambda t: len(t.strip()) >= 20)
    registry = ExtractorRegistry([latin_ok])
    path = _write_pdf(tmp_path / "mixed.pdf", ["Ministry of Justice case file 4471", "", ""])

    ocr = RecordingOcr()
    doc = ingest.load_document(path, ocr_backend=ocr, registry=registry)
    assert doc.source == "ocr"
    assert ocr.calls == [2], "only the two blank pages should be rasterized"
    assert doc.page_count == 3
    assert "Ministry of Justice" in doc.text  # the digital page survived


def test_latin_only_pdf_is_treated_as_needing_ocr(tmp_path):
    # Under the default policy a Latin text layer is a bad scan export.
    path = _write_pdf(tmp_path / "latin.pdf", ["Lorem ipsum dolor sit amet " * 4])
    ocr = RecordingOcr()
    doc = ingest.load_document(path, ocr_backend=ocr)
    assert doc.source == "ocr"
    assert ocr.calls == [1]


def test_scanned_pdf_without_an_ocr_backend_fails_loudly(tmp_path):
    path = _write_pdf(tmp_path / "scanned.pdf", [""])
    with pytest.raises(OCRUnavailable, match="without a usable text layer"):
        ingest.load_document(path, ocr_backend=None)


def test_image_without_an_ocr_backend_fails_loudly(tmp_path):
    target = tmp_path / "scan.png"
    target.write_bytes(b"\x89PNG\r\n\x1a\n")
    with pytest.raises(OCRUnavailable):
        ingest.load_document(target, ocr_backend=None)


# ---------- directories ----------


def test_directory_ingest_is_recursive_sorted_and_filtered(tmp_path):
    (tmp_path / "nested").mkdir()
    (tmp_path / "b.txt").write_text(ARABIC, encoding="utf-8")
    (tmp_path / "a.txt").write_text(ARABIC, encoding="utf-8")
    (tmp_path / "nested" / "c.md").write_text(ARABIC, encoding="utf-8")
    (tmp_path / "ignore.zip").write_bytes(b"PK")

    docs = ingest.load_directory(tmp_path, ocr_backend=None)
    assert [d.doc_id for d in docs] == ["a", "b", "c"]


def test_empty_directory_yields_nothing(tmp_path):
    assert ingest.load_directory(tmp_path, ocr_backend=None) == []


# ---------- Claude backend wiring (no network) ----------


class _FakeBlock:
    type = "text"

    def __init__(self, text):
        self.text = text


class _FakeResponse:
    def __init__(self, text):
        self.content = [_FakeBlock(text)]


class _FakeMessages:
    def __init__(self):
        self.last_kwargs = None

    def create(self, **kwargs):
        self.last_kwargs = kwargs
        return _FakeResponse("نص من كلود")


class _FakeClient:
    def __init__(self):
        self.messages = _FakeMessages()


def test_claude_backend_sends_images_and_an_instruction():
    client = _FakeClient()
    backend = ClaudeVisionOcr(client=client, model="claude-opus-5")
    out = backend.transcribe([b"one", b"two"])

    assert out == "نص من كلود"
    content = client.messages.last_kwargs["messages"][0]["content"]
    assert sum(1 for b in content if b["type"] == "image") == 2
    assert content[-1]["type"] == "text"
    assert "انسخ" in content[-1]["text"]
    assert client.messages.last_kwargs["model"] == "claude-opus-5"


def test_claude_backend_short_circuits_on_no_images():
    client = _FakeClient()
    assert ClaudeVisionOcr(client=client).transcribe([]) == ""
    assert client.messages.last_kwargs is None


def test_claude_backend_does_not_construct_a_client_until_used():
    # Constructing an anthropic client without a key raises; this must not
    # happen just because a registry was built.
    backend = ClaudeVisionOcr()
    assert backend._client is None


# ---------- regressions found by the end-to-end sweep ----------


class _UnauthenticatedMessages:
    """Mirrors the real SDK: constructs fine, rejects at request time."""

    def create(self, **kwargs):
        raise TypeError("Could not resolve authentication method.")


class _UnauthenticatedClient:
    def __init__(self):
        self.messages = _UnauthenticatedMessages()


def test_missing_credentials_surface_as_ocr_unavailable():
    """Found end to end: a PDF without an Arabic text layer and no API key
    crashed with a raw anthropic TypeError.

    The seam matters. `anthropic.Anthropic()` succeeds without a key and only
    raises when the request is made, so a test that mocks the constructor
    passes while the real path stays broken - which is what happened the
    first time this was fixed.
    """
    backend = ClaudeVisionOcr(client=_UnauthenticatedClient())
    with pytest.raises(OCRUnavailable, match="ANTHROPIC_API_KEY"):
        backend.transcribe([b"page"])


def test_ocr_unavailable_names_the_local_alternative():
    backend = ClaudeVisionOcr(client=_UnauthenticatedClient())
    with pytest.raises(OCRUnavailable, match="tesseract"):
        backend.transcribe([b"page"])


def test_non_auth_failures_are_not_disguised_as_ocr_unavailable():
    """A network failure is a different problem and must not be reported as
    'set your API key'."""

    class _Flaky:
        class messages:
            @staticmethod
            def create(**kwargs):
                raise ConnectionError("network down")

    with pytest.raises(ConnectionError):
        ClaudeVisionOcr(client=_Flaky()).transcribe([b"page"])


@pytest.mark.parametrize("name", ["absent.pdf", "absent.txt", "absent.docx", "absent.png"])
def test_missing_files_raise_filenotfound_for_every_format(tmp_path, name):
    """PyMuPDF raises its own error type for a missing file, which used to
    escape as a traceback while a missing .txt reported cleanly."""
    with pytest.raises(FileNotFoundError, match="no such file"):
        ingest.load_document(tmp_path / name, ocr_backend=None)


# ---------- the gateway vision backend ----------


def test_gateway_ocr_sends_images_as_data_urls():
    """The local-only OCR path: page images go to the self-hosted gateway,
    not off the network."""

    class Recorder:
        def __init__(self):
            self.sent = None

        def chat(self, model, messages, **kw):
            self.sent = messages
            return "نص مستخرج"

    client = Recorder()
    backend = ingest.GatewayVisionOcr(model="vision-model", client=client)
    assert backend.transcribe([b"page-one", b"page-two"]) == "نص مستخرج"

    content = client.sent[0]["content"]
    images = [c for c in content if c["type"] == "image_url"]
    assert len(images) == 2
    assert images[0]["image_url"]["url"].startswith("data:image/png;base64,")
    assert content[-1]["type"] == "text" and "انسخ" in content[-1]["text"]


def test_gateway_ocr_short_circuits_on_no_images():
    class Never:
        def chat(self, *a, **kw):
            raise AssertionError("should not be called")

    assert ingest.GatewayVisionOcr(client=Never()).transcribe([]) == ""


def test_a_text_only_model_fails_naming_the_local_alternative():
    from docrouter.classify.openai_compat import GatewayError

    class TextOnly:
        def chat(self, *a, **kw):
            raise GatewayError("/chat/completions returned 400: model does not support images")

    with pytest.raises(OCRUnavailable, match="tesseract"):
        ingest.GatewayVisionOcr(model="gpt-oss", client=TextOnly()).transcribe([b"x"])
