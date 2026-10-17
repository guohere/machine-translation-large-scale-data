"""Exact and near-duplicate removal.

Duplicates waste training compute and over-weight boilerplate, and duplicates
that also occur in a test set inflate evaluation scores.
"""

import hashlib
import zlib

import numpy as np
import pandas as pd
from pyspark.sql import Window
from pyspark.sql import functions as F
from pyspark.sql.functions import pandas_udf
from pyspark.sql.types import ArrayType, LongType

_MERSENNE = np.uint64((1 << 61) - 1)


def exact_dedup(df, key_cols, value_cols):
    """Keep one row per case-insensitive key.

    Among rows sharing a key, the lexicographically smallest row is kept, so
    the output does not depend on partitioning (unlike dropDuplicates).
    """
    key = F.xxhash64(*[F.lower(F.col(c)) for c in key_cols])
    best = F.min(F.struct(*[F.col(c) for c in value_cols])).alias("_row")
    return df.groupBy(key.alias("_key")).agg(best).select("_row.*")


def band_keys(text, ngram, bands, rows_per_band, a, b):
    """MinHash signature of a text's character n-grams, folded into one 64-bit key per band.

    Letters and digits only, lowercased, so punctuation and case changes do not
    count as differences. Returns None when the text is shorter than one n-gram.
    """
    chars = "".join(ch for ch in text.lower() if ch.isalnum())
    shingles = {chars[i : i + ngram] for i in range(len(chars) - ngram + 1)}
    if not shingles:
        return None
    hashes = np.fromiter((zlib.crc32(s.encode()) for s in shingles), dtype=np.uint64, count=len(shingles))
    # Universal hashing h -> (a*h + b) mod p, one row per permutation. a < 2^31 and
    # crc32 values < 2^32 keep a*h + b below 2^64, so uint64 arithmetic never wraps.
    signature = ((a[:, None] * hashes[None, :] + b[:, None]) % _MERSENNE).min(axis=1)
    keys = []
    for band in range(bands):
        chunk = signature[band * rows_per_band : (band + 1) * rows_per_band].tobytes()
        keys.append(int.from_bytes(hashlib.blake2b(chunk, digest_size=8).digest(), "little", signed=True))
    return keys


def near_dedup(df, col, cfg):
    """Drop rows that share a MinHash band with a row of smaller id.

    Two rows land in the same bucket if all `rows_per_band` hashes of any one
    band match. For Jaccard similarity s that happens with probability
    1 - (1 - s^r)^b, a steep S-curve around (1/b)^(1/r): about 0.79 for the
    default b = r = 10. In each bucket the row with the smallest id is kept.

    This replaces Spark ML's MinHashLSH.approxSimilarityJoin, which pairs rows
    when any single hash matches. Moderately similar rows then collide too
    often, and the candidate self-join grows roughly quadratically. Grouping by
    bucket is linear in the number of rows. As with any bucket-based dedup, a
    chain a~b~c can drop c even if it is not similar to a.
    """
    rng = np.random.default_rng(cfg.seed)
    num_perm = cfg.bands * cfg.rows_per_band
    a = rng.integers(1, 1 << 31, num_perm, dtype=np.uint64)
    b = rng.integers(0, 1 << 61, num_perm, dtype=np.uint64)
    args = (cfg.ngram, cfg.bands, cfg.rows_per_band, a, b)

    @pandas_udf(ArrayType(LongType()))
    def signature(texts: pd.Series) -> pd.Series:
        return texts.map(lambda t: band_keys(t, *args))

    # Ids come from a content hash, not monotonically_increasing_id(): that one
    # can change if Spark recomputes the DataFrame, which would silently break
    # the join back onto the rows below.
    with_id = df.withColumn("_id", F.xxhash64(F.col(col)))
    buckets = with_id.select("_id", F.posexplode(signature(F.col(col))).alias("_band", "_key"))
    root = F.min("_id").over(Window.partitionBy("_band", "_key"))
    duplicates = buckets.withColumn("_root", root).filter(F.col("_id") != F.col("_root")).select("_id").distinct()
    return with_id.join(duplicates, "_id", "left_anti").drop("_id")
