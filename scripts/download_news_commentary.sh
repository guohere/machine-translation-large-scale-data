#!/usr/bin/env bash
# Download the News Commentary v18.1 German-English bitext from statmt.org.
set -euo pipefail

mkdir -p data/raw
url="https://data.statmt.org/news-commentary/v18.1/training/news-commentary-v18.de-en.tsv.gz"
curl -fL --retry 3 -o data/raw/news-commentary-v18.de-en.tsv.gz "$url"
echo "first lines (check the column order matches configs/news_commentary_en_de.yaml):"
gzip -dc data/raw/news-commentary-v18.de-en.tsv.gz | head -3
