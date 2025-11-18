"""
Utilidades para trabajar con ISBN.

Funciones principales:
- clean_isbn: normaliza entradas eliminando separadores y mayusculizando 'X'.
- is_valid_isbn10 / is_valid_isbn13: validación de checksum y formato.
- isbn10_to_isbn13 / try_normalize_isbn: conversión y normalización consistente.
- extract_isbns_from_text: extracción heurística desde texto libre/URLs.

Supuestos:
- Se espera que las entradas contengan posibles separadores; la normalización elimina todo excepto dígitos y 'X'.
- Para igualdad y claves se prefiere ISBN-13 válido. Si existe ISBN-10 válido se convierte a ISBN-13.
"""
from __future__ import annotations

import re
from typing import List, Optional, Tuple

ISBN10_RE = re.compile(r"(?i)\b(\d[\d\-\s]{8,}[\dxX])\b")
ISBN13_RE = re.compile(r"\b(97[89][\d\-\s]{9,}\d)\b")


def clean_isbn(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    s = re.sub(r"[^0-9Xx]", "", str(value))
    return s.upper() if s else None


def _isbn10_checksum(digits9: str) -> str:
    total = 0
    for i, ch in enumerate(digits9, start=1):
        total += (11 - i) * int(ch)
    remainder = total % 11
    check = 11 - remainder
    if check == 10:
        return "X"
    if check == 11:
        return "0"
    return str(check)


def is_valid_isbn10(isbn: Optional[str]) -> bool:
    s = clean_isbn(isbn)
    if not s or len(s) != 10:
        return False
    if not re.fullmatch(r"\d{9}[\dX]", s):
        return False
    expected = _isbn10_checksum(s[:9])
    return s[-1] == expected


def _isbn13_checksum(digits12: str) -> str:
    total = 0
    for i, ch in enumerate(digits12):
        weight = 1 if (i % 2 == 0) else 3
        total += int(ch) * weight
    check = (10 - (total % 10)) % 10
    return str(check)


def is_valid_isbn13(isbn: Optional[str]) -> bool:
    s = clean_isbn(isbn)
    if not s or len(s) != 13 or not s.isdigit():
        return False
    expected = _isbn13_checksum(s[:12])
    return s[-1] == expected


def isbn10_to_isbn13(isbn10: str) -> Optional[str]:
    s = clean_isbn(isbn10)
    if not s or len(s) != 10 or not is_valid_isbn10(s):
        return None
    core = "978" + s[:9]
    return core + _isbn13_checksum(core)


def try_normalize_isbn(isbn: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
    """
    Devuelve (isbn10_normalizado, isbn13_normalizado) si son válidos, en caso contrario None.
    Si entra ISBN-10 válido se convierte a ISBN-13 también.
    """
    s = clean_isbn(isbn)
    if not s:
        return None, None
    if len(s) == 10 and is_valid_isbn10(s):
        isbn13 = isbn10_to_isbn13(s)
        return s, isbn13
    if len(s) == 13 and is_valid_isbn13(s):
        return None, s
    return None, None


def extract_isbns_from_text(text: Optional[str]) -> List[str]:
    if not text:
        return []
    candidates = set()
    for m in ISBN13_RE.finditer(text):
        cleaned = clean_isbn(m.group(1))
        if cleaned and is_valid_isbn13(cleaned):
            candidates.add(cleaned)
    for m in ISBN10_RE.finditer(text):
        cleaned = clean_isbn(m.group(1))
        if cleaned and is_valid_isbn10(cleaned):
            candidates.add(cleaned)
            converted = isbn10_to_isbn13(cleaned)
            if converted:
                candidates.add(converted)
    return sorted(candidates)


__all__ = [
    "clean_isbn",
    "is_valid_isbn10",
    "is_valid_isbn13",
    "isbn10_to_isbn13",
    "try_normalize_isbn",
    "extract_isbns_from_text",
]
