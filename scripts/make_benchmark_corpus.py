"""Generate a synthetic EN/DE corpus with known noise, for scale tests of the cleaning stage.

The sentences are word salad from small vocabularies, so they are useless for
training. What is known is how many rows each filter should remove: about
`--dup` exact duplicates, `--near` near-duplicates (punctuation or case
changes) and `--noise` broken rows (too short, number lists, untranslated copies).

    python scripts/make_benchmark_corpus.py --rows 1000000 --out data/raw/benchmark.tsv
"""

import argparse
import random

EN = (
    "the government said on tuesday that new rules for banks would come into force next year while critics "
    "warned of higher costs for small firms and families across the country after months of debate in parliament"
).split()
DE = (
    "die regierung sagte am dienstag dass neue regeln für banken im nächsten jahr in kraft treten während "
    "kritiker vor höheren kosten für kleine firmen und familien im ganzen land nach monaten der debatte warnten"
).split()


def sentence(rng, words, length=None):
    text = " ".join(rng.choices(words, k=length or rng.randint(8, 25)))
    return text[0].upper() + text[1:] + "."


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rows", type=int, default=1_000_000)
    parser.add_argument("--dup", type=float, default=0.10)
    parser.add_argument("--near", type=float, default=0.05)
    parser.add_argument("--noise", type=float, default=0.05)
    parser.add_argument("--out", default="data/raw/benchmark.tsv")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    seen = []
    with open(args.out, "w") as f:
        for _ in range(args.rows):
            roll = rng.random()
            if seen and roll < args.dup:
                de, en = rng.choice(seen)
            elif seen and roll < args.dup + args.near:
                de, en = rng.choice(seen)
                en = en.rstrip(".") + rng.choice(["!", "?", " ."])
            elif roll < args.dup + args.near + args.noise:
                de, en = rng.choice(
                    [
                        ("Zu kurz.", "Too short."),
                        ("1 2 3 4 5 6 7", "1 2 3 4 5 6 7"),
                        (sentence(rng, EN),) * 2,
                    ]
                )
            else:
                length = rng.randint(8, 25)  # same word count on both sides, like a real translation
                de, en = sentence(rng, DE, length), sentence(rng, EN, length)
                if len(seen) < 100_000:
                    seen.append((de, en))
            f.write(f"{de}\t{en}\n")
    print(f"wrote {args.rows:,} rows to {args.out}")


if __name__ == "__main__":
    main()
