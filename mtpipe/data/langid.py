"""Language identification as a vectorised Spark UDF, using the offline py3langid model."""

import pandas as pd
from pyspark.sql.functions import pandas_udf

# One identifier per Python worker process, built on first use: loading the model
# per batch would dominate the runtime.
_identifiers = {}


def _identifier(candidates):
    if candidates not in _identifiers:
        from py3langid.langid import MODEL_DIR, MODEL_FILE, LanguageIdentifier

        identifier = LanguageIdentifier.from_model_file(MODEL_DIR / MODEL_FILE, norm_probs=True)
        if candidates:
            identifier.set_languages(list(candidates))
        _identifiers[candidates] = identifier
    return _identifiers[candidates]


def identify_language(candidates=None):
    """Return a UDF mapping a text column to a struct<lang, prob>."""
    candidates = tuple(candidates) if candidates else None

    @pandas_udf("lang string, prob double")
    def identify(texts: pd.Series) -> pd.DataFrame:
        identifier = _identifier(candidates)
        results = [identifier.classify(t) if t else (None, 0.0) for t in texts]
        return pd.DataFrame(results, columns=["lang", "prob"])

    return identify
