"""Conservative Polars text keys; original display names must be retained separately."""

import polars as pl
from unidecode import unidecode

_RUSSIAN_TO_LATIN = {
    chr(code): unidecode(chr(code)).upper() for code in range(ord("А"), ord("Я") + 1)
}


def clean_text(expr: pl.Expr) -> pl.Expr:
    """NFKC, punctuation, whitespace and case cleanup, with explicit missing tokens."""
    cleaned = (
        expr.cast(pl.String)
        .str.normalize("NFKC")
        .str.replace_all("[’‘ʼ`´]", "'")
        .str.replace_all("[‐‑‒–—−]", "-")
        .str.replace_all(r"\s+", " ")
        .str.strip_chars()
        .str.to_uppercase()
    )
    return cleaned.replace(["", "NAN"], None)


def normalize_identifier(expr: pl.Expr) -> pl.Expr:
    """Normalize document, ticket or card keys without losing leading zeros."""
    return clean_text(expr).str.replace_all(r"\s+", "")


def normalize_name(expr: pl.Expr) -> pl.Expr:
    """Build a comparison key, preserving word positions and original spelling limits.

    Russian uses E for Ё/Э, I for И/Й, Y for Ы, KH for Х, TS for Ц,
    SHCH for Щ, IU for Ю, IA for Я, and drops Ь/Ъ. Apostrophes and
    hyphens are removed within words. Latin variants and typos are not guessed.
    """
    # Preserve existing Ё/Е equivalence; Unidecode otherwise emits IO for Ё.
    normalized = (
        clean_text(expr)
        .str.replace_all("Ё", "Е", literal=True)
        .str.replace_many(_RUSSIAN_TO_LATIN)
        .str.replace_all("['-]", "")
    )
    return normalized.replace("", None)
