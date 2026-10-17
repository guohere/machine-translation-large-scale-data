import numpy as np

from mtpipe.data.config import DedupConfig
from mtpipe.data.dedup import band_keys, exact_dedup, near_dedup


def test_exact_dedup_on_source_is_case_insensitive_and_deterministic(spark):
    rows = [("Hello world again.", "B"), ("hello world again.", "A"), ("Something else.", "C")]
    df = spark.createDataFrame(rows, ["src", "tgt"])
    kept = sorted(tuple(r) for r in exact_dedup(df, ["src"], ["src", "tgt"]).collect())
    # The smallest row of the duplicate group survives: "Hello..." sorts before "hello...".
    assert kept == [("Hello world again.", "B"), ("Something else.", "C")]


def test_exact_dedup_on_pair_keeps_different_targets(spark):
    rows = [("Same source.", "Ziel eins."), ("Same source.", "Ziel zwei."), ("Same source.", "Ziel eins.")]
    df = spark.createDataFrame(rows, ["src", "tgt"])
    assert exact_dedup(df, ["src", "tgt"], ["src", "tgt"]).count() == 2


def test_near_dedup_drops_close_variants_only(spark):
    rows = [
        ("The European Parliament adopted the resolution on Tuesday.",),
        ("The European Parliament adopted the resolution on Tuesday!",),  # punctuation only
        ("The European parliament adopted the resolution on tuesday",),  # case and punctuation
        ("The European Parliament rejected the proposal on Friday.",),  # genuinely different
        ("Ok.",),  # too short to shingle: passes through untouched
    ]
    df = spark.createDataFrame(rows, ["src"])
    kept = {r["src"] for r in near_dedup(df, "src", DedupConfig()).collect()}
    assert len(kept) == 3
    assert "The European Parliament rejected the proposal on Friday." in kept
    assert "Ok." in kept


def keys(text, bands=10, rows=10):
    rng = np.random.default_rng(0)
    a = rng.integers(1, 1 << 31, bands * rows, dtype=np.uint64)
    b = rng.integers(0, 1 << 61, bands * rows, dtype=np.uint64)
    return band_keys(text, 5, bands, rows, a, b)


def test_band_keys_ignore_case_and_punctuation():
    assert keys("Inflation fell, again!") == keys("inflation fell again.")


def test_band_keys_separate_similar_from_different():
    base = "The central bank kept interest rates unchanged for the third month in a row."
    close = "The central bank kept interest rates unchanged for the third month in a row, it said."
    other = "Heavy rain caused flooding in several northern towns over the weekend."
    shared = lambda x, y: sum(p == q for p, q in zip(keys(x), keys(y), strict=True))  # noqa: E731
    assert shared(base, close) > 0
    assert shared(base, other) == 0


def test_band_keys_too_short():
    assert keys("Ok.") is None
