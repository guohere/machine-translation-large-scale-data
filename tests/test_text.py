import pytest

from mtpipe.data.text import normalize


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("  Hello   world \n", "Hello world"),
        ("full width　space", "full width space"),  # NBSP and ideographic space
        ("Ｆｕｌｌｗｉｄｔｈ", "Fullwidth"),  # NFKC folds full-width letters
        ("zero​width﻿", "zerowidth"),
        ("line separator", "line separator"),  # whitespace never glues words
        ("carriage\rreturn", "carriage return"),
        ("tab\tand\x07bell", "tab andbell"),
        ("", ""),
    ],
)
def test_normalize(raw, expected):
    assert normalize(raw) == expected


def test_normalize_none():
    assert normalize(None) is None
