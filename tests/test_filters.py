from pyspark.sql import functions as F

from mtpipe.data.config import load_config
from mtpipe.data.filters import heuristic_reason, language_reason
from mtpipe.data.langid import identify_language

GOOD = ("The committee approved the new budget.", "Der Ausschuss hat den neuen Haushalt gebilligt.")


def reasons(spark, cfg, rows, columns=("src", "tgt")):
    df = spark.createDataFrame(rows, list(columns))
    return [r["reason"] for r in df.select(heuristic_reason(cfg).alias("reason")).collect()]


def test_heuristic_filters_bitext(spark):
    cfg = load_config()
    rows = [
        GOOD,
        ("", "Etwas Text hier."),
        ("Too short", "Zu kurz ist das nicht."),
        (" ".join(["word"] * 151), "Ein Satz mit genug Wörtern."),
        ("12 345 678 910 1112", "12 345 678 910 1112."),
        ("This is a reasonably long English sentence about politics.", "Ja, das ist es."),
        ("Brexit negotiations resumed today.", "brexit negotiations resumed today."),
    ]
    assert reasons(spark, cfg, rows) == [
        None,
        "empty",
        "too_short",
        "too_long",
        "few_letters",
        "length_ratio",
        "copy",
    ]


def test_heuristic_filters_monolingual(spark):
    cfg = load_config()
    cfg.input.tgt_col = None
    rows = [(GOOD[0],), ("Hi there",), ("A short but valid sentence.",)]
    assert reasons(spark, cfg, rows, columns=("src",)) == [None, "too_short", None]


def test_language_filter(spark):
    cfg = load_config()
    rows = [
        GOOD,
        GOOD[::-1],  # sides swapped
        (GOOD[0], "Le comité a approuvé le nouveau budget."),
    ]
    identify = identify_language(["en", "de", "fr"])
    df = (
        spark.createDataFrame(rows, ["src", "tgt"])
        .withColumn("src_lid", identify(F.col("src")))
        .withColumn("tgt_lid", identify(F.col("tgt")))
    )
    got = [r["reason"] for r in df.select(language_reason(cfg).alias("reason")).collect()]
    assert got == [None, "wrong_lang_src", "wrong_lang_tgt"]
