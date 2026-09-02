"""Arabic text normalization.

Two strengths, deliberately:

`light` is what the LLM sees — it strips only noise that carries no meaning
(diacritics, tatweel, presentation forms, zero-width marks). Anything that
could change a word stays, because Claude reads real Arabic better than it
reads mangled Arabic.

`aggressive` is what the TF-IDF baseline sees — it also folds alef/ya/hamza
variants and Arabic-Indic digits, because bag-of-words models need
"إجازة"/"اجازة" to collide into one feature.
"""

from __future__ import annotations

import re
import unicodedata

# Harakat, tanween, shadda, sukun, superscript alef, and the Quranic marks.
_TASHKEEL = re.compile(r"[ؐ-ًؚ-ٰٟۖ-ۭ]")
_TATWEEL = re.compile(r"ـ+")
# Zero-width joiner/non-joiner, BOM, RTL/LTR marks and embeddings.
_INVISIBLE = re.compile(r"[​-‏‪-‮⁦-⁩﻿]")
_WHITESPACE = re.compile(r"[ \t ]+")
_BLANK_LINES = re.compile(r"\n{3,}")

_ARABIC_INDIC = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789")
_FOLD = str.maketrans(
    {
        "أ": "ا", "إ": "ا", "آ": "ا", "ٱ": "ا",
        "ى": "ي", "ئ": "ي",
        "ؤ": "و",
        "ة": "ه",
        "ک": "ك", "ی": "ي",
    }
)


def light(text: str) -> str:
    """Normalization safe enough to send to an LLM or show to a human."""
    if not text:
        return ""
    # NFKC turns Arabic presentation forms (ﻻ, ﷲ) back into ordinary letters,
    # which is exactly what OCR output tends to be full of.
    text = unicodedata.normalize("NFKC", text)
    text = _INVISIBLE.sub("", text)
    text = _TASHKEEL.sub("", text)
    text = _TATWEEL.sub("", text)
    text = _WHITESPACE.sub(" ", text)
    text = _BLANK_LINES.sub("\n\n", text)
    return "\n".join(line.strip() for line in text.split("\n")).strip()


def aggressive(text: str) -> str:
    """Feature-space normalization for bag-of-words models. Lossy on purpose."""
    text = light(text).translate(_ARABIC_INDIC).translate(_FOLD)
    # Keep Arabic letters, ASCII alphanumerics and separators; drop the rest.
    text = re.sub(r"[^ء-ي0-9a-zA-Z\s]", " ", text)
    return _WHITESPACE.sub(" ", text).strip()


def looks_arabic(text: str, threshold: float = 0.2) -> bool:
    """True when Arabic letters make up at least `threshold` of the letters.

    Used to decide whether a PDF's embedded text layer is real Arabic or the
    garbage some scanners emit, in which case we OCR instead.
    """
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return False
    arabic = sum(1 for c in letters if "؀" <= c <= "ۿ")
    return arabic / len(letters) >= threshold
