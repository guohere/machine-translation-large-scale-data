"""Clean a parallel (or monolingual) corpus with PySpark.

    read -> normalise -> heuristic filters -> language ID -> exact dedup -> near dedup -> parquet

Writes, under the output directory:
  data/                  cleaned rows as parquet
  stats.json             rows in/out, rejections by reason and wall time per stage
  rejected/<reason>.jsonl  a sample of rejected rows for each reason

Usage:
  python -m mtpipe.data.clean --config configs/news_commentary_en_de.yaml
"""

import argparse
import json
import shutil
import time
from pathlib import Path

import pandas as pd
from pyspark.sql import SparkSession, Window
from pyspark.sql import functions as F
from pyspark.sql.functions import pandas_udf
from pyspark.sql.types import StringType

from mtpipe.data.config import load_config
from mtpipe.data.dedup import exact_dedup, near_dedup
from mtpipe.data.filters import heuristic_reason, language_reason
from mtpipe.data.langid import identify_language
from mtpipe.data.text import normalize


@pandas_udf(StringType())
def normalize_udf(texts: pd.Series) -> pd.Series:
    return texts.map(normalize)


def build_spark(master, extra_conf):
    builder = (
        SparkSession.builder.appName("mt-clean")
        .master(master)
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        .config("spark.ui.showConsoleProgress", "false")
    )
    for key, value in extra_conf.items():
        builder = builder.config(key, str(value))
    return builder.getOrCreate()


def read_input(spark, cfg):
    inp = cfg.input
    if inp.format == "tsv":
        # quote="" turns off CSV quoting: raw corpora contain unbalanced quotes.
        schema = ", ".join(f"`{c}` string" for c in inp.columns)
        df = spark.read.csv(inp.path, sep="\t", quote="", escape="", schema=schema, mode="PERMISSIVE")
    elif inp.format == "text":
        df = spark.read.text(inp.path).withColumnRenamed("value", inp.src_col)
    elif inp.format == "parquet":
        df = spark.read.parquet(inp.path)
    else:
        df = spark.read.json(inp.path)
    columns = [F.col(inp.src_col).alias("src")]
    if cfg.bitext:
        columns.append(F.col(inp.tgt_col).alias("tgt"))
    df = df.select(*columns)
    return df.repartition(inp.repartition) if inp.repartition else df


class StageLog:
    """Collects per-stage counts and timings for stats.json."""

    def __init__(self):
        self.stages = []
        self._start = time.perf_counter()

    def add(self, name, rows_in, rows_out, rejected=None):
        now = time.perf_counter()
        self.stages.append(
            {
                "stage": name,
                "rows_in": rows_in,
                "rows_out": rows_out,
                "removed": rows_in - rows_out,
                "rejected_by_reason": rejected or {},
                "seconds": round(now - self._start, 2),
            }
        )
        self._start = now


def apply_reasons(df, reason, log, stage, samples, sample_dir):
    """Tag rows with a rejection reason, record counts and samples, return the survivors."""
    tagged = df.withColumn("reason", reason).persist()
    counts = {row["reason"]: row["count"] for row in tagged.groupBy("reason").count().collect()}
    kept_rows = counts.pop(None, 0)
    log.add(stage, kept_rows + sum(counts.values()), kept_rows, dict(sorted(counts.items())))
    if samples and counts:
        save_samples(tagged.filter(F.col("reason").isNotNull()), samples, sample_dir)
    return tagged.filter(F.col("reason").isNull()).drop("reason"), kept_rows


def save_samples(rejected, n, sample_dir):
    # Ordered by a content hash, so the sample is stable from run to run.
    window = Window.partitionBy("reason").orderBy(F.xxhash64("src"))
    rows = rejected.withColumn("_rank", F.row_number().over(window)).filter(F.col("_rank") <= n).drop("_rank")
    by_reason = {}
    for row in rows.collect():
        record = row.asDict(recursive=True)
        by_reason.setdefault(record.pop("reason"), []).append(record)
    sample_dir.mkdir(parents=True, exist_ok=True)
    for reason, records in by_reason.items():
        with open(sample_dir / f"{reason}.jsonl", "w") as f:
            f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in records)


def run(cfg, spark):
    out = Path(cfg.output)
    if out.exists():
        shutil.rmtree(out)
    sample_dir = out / "rejected"
    log = StageLog()

    df = read_input(spark, cfg)
    normalized = df.select(*[normalize_udf(F.col(c)).alias(c) for c in df.columns])

    kept, rows = apply_reasons(
        normalized, heuristic_reason(cfg), log, "heuristic_filters", cfg.rejected_samples, sample_dir
    )

    if cfg.langid.enabled:
        identify = identify_language(cfg.langid.candidates)
        with_lid = kept.withColumn("src_lid", identify(F.col("src")))
        if cfg.bitext:
            with_lid = with_lid.withColumn("tgt_lid", identify(F.col("tgt")))
        kept, rows = apply_reasons(
            with_lid, language_reason(cfg), log, "language_id", cfg.rejected_samples, sample_dir
        )
        kept = kept.withColumn("src_lang_prob", F.col("src_lid.prob")).drop("src_lid")
        if cfg.bitext:
            kept = kept.withColumn("tgt_lang_prob", F.col("tgt_lid.prob")).drop("tgt_lid")

    keys = ["src"] if cfg.dedup.exact_on == "source" or not cfg.bitext else ["src", "tgt"]
    kept = exact_dedup(kept, keys, kept.columns).persist()
    rows_after = kept.count()
    log.add("exact_dedup", rows, rows_after)
    rows = rows_after

    if cfg.dedup.near:
        kept = near_dedup(kept, "src", cfg.dedup).persist()
        rows_after = kept.count()
        log.add("near_dedup", rows, rows_after)
        rows = rows_after

    kept.write.parquet(str(out / "data"))
    log.add("write", rows, rows)

    stats = {
        "input_rows": log.stages[0]["rows_in"],
        "output_rows": rows,
        "kept_fraction": round(rows / max(log.stages[0]["rows_in"], 1), 4),
        "total_seconds": round(sum(s["seconds"] for s in log.stages), 2),
        "stages": log.stages,
        "config": cfg.to_dict(),
    }
    (out / "stats.json").write_text(json.dumps(stats, indent=2) + "\n")
    return stats


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", help="YAML config; see configs/")
    parser.add_argument("--input", help="override input.path")
    parser.add_argument("--output", help="override output")
    parser.add_argument("--master", default="local[*]", help="Spark master, e.g. local[8] or a cluster URL")
    args = parser.parse_args(argv)

    cfg = load_config(args.config, **{"input.path": args.input, "output": args.output})
    if not cfg.input.path:
        parser.error("no input path: set input.path in the config or pass --input")
    spark = build_spark(args.master, cfg.spark)
    try:
        stats = run(cfg, spark)
    finally:
        spark.stop()

    print(f"{stats['input_rows']:,} rows in -> {stats['output_rows']:,} out ({stats['kept_fraction']:.1%} kept)")
    for stage in stats["stages"]:
        print(f"  {stage['stage']:<18} -{stage['removed']:>10,}  {stage['seconds']:>7.1f}s  {stage['rejected_by_reason']}")


if __name__ == "__main__":
    main()
