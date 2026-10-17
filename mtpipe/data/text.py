"""Plain-Python text normalisation, applied inside Spark through a pandas UDF."""

import re
import unicodedata

# Control characters, zero-width characters, bidi marks and the BOM. Whitespace
# (including \r and U+2028) is turned into spaces before these are dropped, so
# removing them never glues two words together.
_INVISIBLE = re.compile(r"[\x00-\x1f\x7f-\x9f​-‏‪-‮⁠-⁯﻿]")
_SPACE = re.compile(r"\s+")


def normalize(text):
    """NFKC-normalise, drop invisible characters and collapse whitespace.

    NFKC folds compatibility forms (full-width letters, ligatures, non-breaking
    spaces) into their canonical equivalents, so the same sentence typed two ways
    deduplicates and filters identically.
    """
    if text is None:
        return None
    text = unicodedata.normalize("NFKC", text)
    text = _INVISIBLE.sub("", _SPACE.sub(" ", text))
    return _SPACE.sub(" ", text).strip()
