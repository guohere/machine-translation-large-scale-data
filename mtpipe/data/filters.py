"""Row-level filters, written as Spark column expressions.

Each function returns a column holding the name of the first check a row fails,
or null if it passes. Recording the reason rather than dropping rows straight
away gives per-filter counts from a single aggregation, and lets the pipeline
save examples of each rejection for inspection.
"""

from functools import reduce

from pyspark.sql import functions as F


def _first_failing(checks):
    """Fold (condition, reason) pairs into one CASE WHEN expression; order sets priority."""
    first_cond, first_reason = checks[0]
    expr = F.when(first_cond, F.lit(first_reason))
    expr = reduce(lambda acc, check: acc.when(check[0], F.lit(check[1])), checks[1:], expr)
    return expr.otherwise(F.lit(None).cast("string"))


def _words(col):
    # Input is normalised to single spaces, so splitting on " " counts words.
    return F.size(F.split(col, " "))


def _letter_ratio(col):
    return F.length(F.regexp_replace(col, r"[^\p{L}]", "")) / F.length(col)


def heuristic_reason(cfg, src="src", tgt="tgt"):
    """Cheap checks that run in the JVM: emptiness, length, letter share, ratio, copies."""
    f = cfg.filters
    sides = [F.col(src)] + ([F.col(tgt)] if cfg.bitext else [])
    checks = [(c.isNull() | (F.length(c) == 0), "empty") for c in sides]
    checks += [(_words(c) < f.min_words, "too_short") for c in sides]
    checks += [(_words(c) > f.max_words, "too_long") for c in sides]
    checks += [(_letter_ratio(c) < f.min_letter_ratio, "few_letters") for c in sides]
    if cfg.bitext:
        s, t = F.length(F.col(src)), F.length(F.col(tgt))
        checks.append((F.greatest(s, t) / F.least(s, t) > f.max_length_ratio, "length_ratio"))
        # An untranslated copy teaches the model to echo its input.
        checks.append((F.lower(F.col(src)) == F.lower(F.col(tgt)), "copy"))
    return _first_failing(checks)


def language_reason(cfg, src_lang="src_lid", tgt_lang="tgt_lid"):
    """Checks on the struct<lang, prob> columns produced by langid.identify_language."""
    min_prob = cfg.langid.min_prob
    checks = [(F.col(f"{src_lang}.lang") != cfg.src_lang, "wrong_lang_src")]
    if cfg.bitext:
        checks.append((F.col(f"{tgt_lang}.lang") != cfg.tgt_lang, "wrong_lang_tgt"))
    checks.append((F.col(f"{src_lang}.prob") < min_prob, "unsure_lang_src"))
    if cfg.bitext:
        checks.append((F.col(f"{tgt_lang}.prob") < min_prob, "unsure_lang_tgt"))
    return _first_failing(checks)
