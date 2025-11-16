"""
Utilidades de calidad y normalización para el pipeline de libros.
Incluye validaciones y métricas que alimentan docs/quality_metrics.json
"""
from __future__ import annotations

import json
import re
from datetime import datetime
from typing import Dict, Iterable, List, Mapping, Optional, Tuple

ISO4217 = {
    "USD", "EUR", "GBP", "JPY", "CNY", "AUD", "CAD", "CHF", "MXN", "BRL",
}

BCP47_BASIC = {"en", "en-US", "en-GB", "es", "es-ES", "pt-BR", "fr", "de", "it"}


def normalize_whitespace(s: Optional[str]) -> Optional[str]:
    if s is None:
        return None
    s2 = re.sub(r"\s+", " ", s).strip()
    return s2 if s2 != "" else None


def to_snake(name: str) -> str:
    s = re.sub(r"[^0-9A-Za-z]+", "_", name)
    s = re.sub(r"__+", "_", s)
    return s.strip("_").lower()


def parse_date_to_iso(date_str: Optional[str]) -> Tuple[Optional[str], bool]:
    """
    Devuelve (fecha_ISO_YYYY-MM-DD, es_parcial). Si no se puede parsear, (None, False)
    """
    if not date_str:
        return None, False
    s = str(date_str).strip()
    # Formatos comunes
    for fmt in ("%Y-%m-%d", "%Y-%m", "%Y"):
        try:
            dt = datetime.strptime(s, fmt)
            if fmt == "%Y-%m-%d":
                return dt.strftime("%Y-%m-%d"), False
            if fmt == "%Y-%m":
                return dt.strftime("%Y-%m-01"), True
            if fmt == "%Y":
                return dt.strftime("%Y-01-01"), True
        except ValueError:
            pass
    # Intento libre con heurística
    m = re.match(r"^(\d{4})(?:[-/](\d{1,2})(?:[-/](\d{1,2}))?)?$", s)
    if m:
        y = int(m.group(1))
        mth = int(m.group(2)) if m.group(2) else 1
        d = int(m.group(3)) if m.group(3) else 1
        try:
            dt = datetime(y, mth, d)
            return dt.strftime("%Y-%m-%d"), True
        except ValueError:
            return None, False
    return None, False


def validate_language(code: Optional[str]) -> bool:
    if not code:
        return False
    c = str(code).strip()
    # Normalizar a lower para simple check; permitir subtags con '-'
    if len(c) < 2:
        return False
    # check básico
    return c in BCP47_BASIC or re.fullmatch(r"[a-zA-Z]{2,3}(?:-[a-zA-Z0-9]{2,8})*", c) is not None


def normalize_language(code: Optional[str]) -> Optional[str]:
    if not code:
        return None
    c = str(code).strip()
    # caso común: lower excepto subtags región en mayúsculas
    parts = c.split('-')
    if not parts:
        return None
    parts[0] = parts[0].lower()
    for i in range(1, len(parts)):
        if len(parts[i]) == 2:
            parts[i] = parts[i].upper()
        else:
            parts[i] = parts[i]
    return "-".join(parts)


def validate_currency(code: Optional[str]) -> bool:
    if not code:
        return False
    c = str(code).upper().strip()
    return bool(re.fullmatch(r"[A-Z]{3}", c)) and c in ISO4217


def nulls_by_column(rows: List[Mapping[str, object]], columns: List[str]) -> Dict[str, int]:
    counts = {c: 0 for c in columns}
    for r in rows:
        for c in columns:
            v = r.get(c) if isinstance(r, Mapping) else None
            if v is None or (isinstance(v, str) and v.strip() == ""):
                counts[c] += 1
    return counts


def listify(x: Optional[object]) -> List[str]:
    if x is None:
        return []
    if isinstance(x, list):
        return [str(i).strip() for i in x if str(i).strip()]
    # split común por coma o punto y coma
    parts = re.split(r"[,;]", str(x))
    return [p.strip() for p in parts if p.strip()]


def uniq_preserve(seq: Iterable[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for s in seq:
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out


def compute_quality_metrics(rows: List[Mapping[str, object]]) -> Dict[str, object]:
    total = len(rows)
    cols = [
        "book_id", "titulo", "autor_principal", "isbn13", "fecha_publicacion",
        "idioma", "precio", "moneda",
    ]
    nulos = nulls_by_column(rows, cols) if rows else {c: 0 for c in cols}

    def pct(valid: int, denom: int) -> float:
        return round((valid / denom) * 100.0, 2) if denom else 0.0

    fechas_validas = sum(1 for r in rows if r.get("fecha_publicacion_valida") is True)
    idiomas_validos = sum(1 for r in rows if r.get("idioma_valido") is True)
    monedas_validas = sum(1 for r in rows if r.get("moneda_valida") is True)
    isbn_validos = sum(1 for r in rows if r.get("isbn13_valido") is True)

    completitud = 0.0
    if total:
        for r in rows:
            filled = sum(1 for c in cols if r.get(c) not in (None, ""))
            completitud += filled / len(cols)
        completitud = round((completitud / total) * 100.0, 2)

    # métricas adicionales: duplicados por isbn13 y por book_id
    isbn13_values = [r.get("isbn13") for r in rows if r.get("isbn13") not in (None, "")]
    dup_isbn13 = len(isbn13_values) - len(set(isbn13_values)) if isbn13_values else 0

    book_id_values = [r.get("book_id") for r in rows if r.get("book_id") not in (None, "")]
    dup_book_id = len(book_id_values) - len(set(book_id_values)) if book_id_values else 0

    # filas por fuente si existe 'fuente_ganadora' en filas
    filas_por_fuente = {}
    try:
        for r in rows:
            src = r.get("fuente_ganadora") or r.get("source_name") or "unknown"
            filas_por_fuente[src] = filas_por_fuente.get(src, 0) + 1
    except Exception:
        filas_por_fuente = {}

    return {
        "filas_totales": total,
        "nulos_por_campo": nulos,
        "porcentaje_fechas_validas": pct(fechas_validas, total),
        "porcentaje_idiomas_validos": pct(idiomas_validos, total),
        "porcentaje_monedas_validas": pct(monedas_validas, total),
        "porcentaje_isbn13_validos": pct(isbn_validos, total),
        "completitud_promedio": completitud,
        "duplicados_isbn13": dup_isbn13,
        "duplicados_book_id": dup_book_id,
        "filas_por_fuente": filas_por_fuente,
    }


def write_quality_json(path: str, metrics: Mapping[str, object]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)


__all__ = [
    "normalize_whitespace",
    "to_snake",
    "parse_date_to_iso",
    "validate_language",
    "normalize_language",
    "validate_currency",
    "nulls_by_column",
    "listify",
    "uniq_preserve",
    "compute_quality_metrics",
    "write_quality_json",
]
