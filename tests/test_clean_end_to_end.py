import gzip
import json

import yaml

from mtpipe.data.clean import run
from mtpipe.data.config import load_config

PAIRS = [
    # file order is German, then English, as in News Commentary
    ("Der Ausschuss hat den neuen Haushalt gebilligt.", "The committee approved the new budget."),
    ("Die Inflation ist im dritten Quartal gesunken.", "Inflation fell in the third quarter."),
    ("Die Inflation ist im dritten Quartal gesunken.", "Inflation fell in the third quarter."),  # exact dup
    ("Die Inflation ist im dritten Quartal gesunken!", "Inflation fell in the third quarter!"),  # near dup
    ("Zu kurz.", "Too short."),
    ("Le comité a approuvé le budget.", "The committee approved the budget today."),  # wrong target language
    ("Die Zentralbank hält die Zinsen unverändert.", 'The central bank keeps "rates unchanged.'),  # stray quote
]


def test_pipeline_end_to_end(spark, tmp_path):
    raw = tmp_path / "corpus.tsv.gz"
    with gzip.open(raw, "wt") as f:
        f.writelines(f"{de}\t{en}\n" for de, en in PAIRS)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "output": str(tmp_path / "out"),
                "input": {"path": str(raw), "columns": ["de", "en"], "src_col": "en", "tgt_col": "de"},
                "langid": {"candidates": ["en", "de", "fr"]},
            }
        )
    )
    stats = run(load_config(config_path), spark)

    assert stats["input_rows"] == 7
    by_stage = {s["stage"]: s for s in stats["stages"]}
    assert by_stage["heuristic_filters"]["rejected_by_reason"] == {"too_short": 1}
    assert by_stage["language_id"]["rejected_by_reason"] == {"wrong_lang_tgt": 1}
    assert by_stage["exact_dedup"]["removed"] == 1
    assert by_stage["near_dedup"]["removed"] == 1
    assert stats["output_rows"] == 3

    out = spark.read.parquet(str(tmp_path / "out" / "data"))
    assert set(out.columns) == {"src", "tgt", "src_lang_prob", "tgt_lang_prob"}
    # The unbalanced quote survives: CSV quoting is off for raw corpora.
    assert 'The central bank keeps "rates unchanged.' in {r["src"] for r in out.collect()}

    sample = (tmp_path / "out" / "rejected" / "too_short.jsonl").read_text().splitlines()
    assert json.loads(sample[0]) == {"src": "Too short.", "tgt": "Zu kurz."}
    assert json.loads((tmp_path / "out" / "stats.json").read_text())["output_rows"] == 3
