"""Cleaning configuration, loaded from YAML (see configs/)."""

from dataclasses import asdict, dataclass, field, fields

import yaml


@dataclass
class InputConfig:
    path: str = ""
    # tsv: tab-separated, columns named by `columns` | text: one sentence per line
    # (monolingual) | parquet | jsonl
    format: str = "tsv"
    columns: list[str] = field(default_factory=lambda: ["src", "tgt"])
    src_col: str = "src"
    tgt_col: str | None = "tgt"  # None for monolingual text
    # Gzipped text arrives as a single partition; repartition so filters run in parallel.
    repartition: int | None = None


@dataclass
class FilterConfig:
    min_words: int = 3
    max_words: int = 150
    # Character-length ratio between the longer and the shorter side (bitext only).
    max_length_ratio: float = 2.5
    # Share of characters that are letters; catches tables, code and number lists.
    min_letter_ratio: float = 0.5


@dataclass
class LangIdConfig:
    enabled: bool = True
    min_prob: float = 0.5
    # Restricting candidates improves accuracy on short sentences; None means all 97 languages.
    candidates: list[str] | None = None


@dataclass
class DedupConfig:
    # "source" keeps one target per source sentence; "pair" only drops identical pairs.
    exact_on: str = "source"
    near: bool = True
    ngram: int = 5  # character n-gram shingles
    # Banded MinHash: rows sharing all hashes of any band are near-duplicates.
    # The similarity threshold is roughly (1/bands)^(1/rows_per_band): 0.79 for 10 x 10.
    bands: int = 10
    rows_per_band: int = 10
    seed: int = 42


@dataclass
class CleanConfig:
    src_lang: str = "en"
    tgt_lang: str | None = "de"
    output: str = "data/clean"
    rejected_samples: int = 20  # rejected rows kept per reason, for manual inspection
    input: InputConfig = field(default_factory=InputConfig)
    filters: FilterConfig = field(default_factory=FilterConfig)
    langid: LangIdConfig = field(default_factory=LangIdConfig)
    dedup: DedupConfig = field(default_factory=DedupConfig)
    spark: dict = field(default_factory=dict)  # extra spark.* settings

    @property
    def bitext(self):
        return self.input.tgt_col is not None

    def to_dict(self):
        return asdict(self)


_SECTIONS = {"input": InputConfig, "filters": FilterConfig, "langid": LangIdConfig, "dedup": DedupConfig}


def load_config(path=None, **overrides):
    raw = yaml.safe_load(open(path)) if path else {}
    raw = raw or {}
    known = {f.name for f in fields(CleanConfig)}
    unknown = set(raw) - known
    if unknown:
        raise ValueError(f"unknown config keys: {sorted(unknown)}")
    sections = {name: cls(**(raw.pop(name, None) or {})) for name, cls in _SECTIONS.items()}
    config = CleanConfig(**raw, **sections)
    for key, value in overrides.items():
        if value is not None:
            section, _, attr = key.rpartition(".")
            setattr(getattr(config, section) if section else config, attr, value)
    if config.dedup.exact_on not in ("source", "pair"):
        raise ValueError("dedup.exact_on must be 'source' or 'pair'")
    if config.input.format not in ("tsv", "text", "parquet", "jsonl"):
        raise ValueError(f"unsupported input format: {config.input.format}")
    if config.input.format == "text":
        config.input.tgt_col = None
    if not config.bitext:
        config.tgt_lang = None
    return config
