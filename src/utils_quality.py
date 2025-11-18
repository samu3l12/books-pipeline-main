"""
Utilidades de calidad y normalización para el pipeline de libros.
Incluye validaciones y métricas que alimentan docs/quality_metrics.json

Comentarios añadidos: supuestos sobre formatos de fecha e idioma, serialización de listas y cómo se calculan las métricas.
"""
from __future__ import annotations

import json
import re
from datetime import datetime
from typing import Dict, Iterable, List, Mapping, Optional, Tuple

# Lista ampliada de ISO-4217 (subset ampliamente usado) para validar monedas
ISO4217 = {
    "USD", "EUR", "GBP", "JPY", "CNY", "AUD", "CAD", "CHF", "MXN", "BRL",
    "SEK", "NOK", "DKK", "INR", "KRW", "ZAR", "SGD", "HKD", "NZD", "TRY",
    "RUB", "AED", "SAR", "COP", "CLP", "ARS", "THB", "IDR", "MYR", "PHP",
    # añadidos comunes
    "PLN", "HUF", "ILS", "VND", "EGP", "TWD", "KWD", "QAR", "BHD", "OMR",
}

# Mapeo de símbolos a ISO-4217 usado para detección rápida
SYMBOL_TO_ISO = {
    "$": "USD",
    "€": "EUR",
    "\u20ac": "EUR",
    "£": "GBP",
    "\u00a3": "GBP",
    "¥": "JPY",
    "\u00a5": "JPY",
    "R$": "BRL",
}

# Conjunto básico de códigos BCP-47 aceptados (ampliable)
BCP47_BASIC = {"en", "en-US", "en-GB", "es", "es-ES", "pt-BR", "fr", "de", "it"}


def normalize_whitespace(s: Optional[str]) -> Optional[str]:
    if s is None:
        return None
    s2 = re.sub(r"\s+", " ", s).strip()
    return s2 if s2 != "" else None


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
    # Normalizar separadores y eliminar ordinales (1st, 2nd) y palabras como 'de'
    s_norm = re.sub(r"(\d)(st|nd|rd|th)", r"\1", s, flags=re.IGNORECASE)
    s_norm = re.sub(r"\s+de\s+", " ", s_norm, flags=re.IGNORECASE)
    s_norm = s_norm.replace(',', ' ').strip()

    # Intento libre con heurística numérica
    m = re.match(r"^(\d{4})(?:[-/](\d{1,2})(?:[-/](\d{1,2}))?)?$", s_norm)
    if m:
        y = int(m.group(1))
        mth = int(m.group(2)) if m.group(2) else 1
        d = int(m.group(3)) if m.group(3) else 1
        try:
            dt = datetime(y, mth, d)
            return dt.strftime("%Y-%m-%d"), True
        except ValueError:
            return None, False

    # Meses en inglés y español (soporte básico para otros idiomas comunes)
    months = {
        'jan':1,'january':1,'ene':1,'enero':1,
        'feb':2,'february':2,'febrero':2,
        'mar':3,'march':3,'marzo':3,
        'apr':4,'april':4,'abril':4,
        'may':5,'may':5,'mayo':5,
        'jun':6,'june':6,'junio':6,
        'jul':7,'july':7,'julio':7,
        'aug':8,'august':8,'agosto':8,
        'sep':9,'september':9,'septiembre':9,'set':9,'setiembre':9,
        'oct':10,'october':10,'octubre':10,
        'nov':11,'november':11,'noviembre':11,
        'dec':12,'december':12,'diciembre':12
    }
    # buscar patrones como 'July 2013', 'Jul 2013', '2013 July', '1 January 2013', '1 de enero de 2013'
    # patrón: (dia)? mes nombre año
    m2 = re.search(r"(?:(\d{1,2})\s+)?([A-Za-z]+)\s+(\d{4})", s_norm)
    if m2:
        day = int(m2.group(1)) if m2.group(1) else 1
        mon = m2.group(2).lower()
        yr = int(m2.group(3))
        mon_key = mon[:3]
        mval = months.get(mon_key) or months.get(mon)
        if mval:
            try:
                dt = datetime(yr, int(mval), int(day))
                # si se especificó día, no es parcial
                parcial = False if m2.group(1) else True
                return dt.strftime("%Y-%m-%d"), parcial
            except Exception:
                return None, False
    # patrón invertido: '2013 July' o '2013 Julio'
    m3 = re.search(r"^(\d{4})\s+([A-Za-z]+)$", s_norm)
    if m3:
        yr = int(m3.group(1))
        mon = m3.group(2).lower()
        mval = months.get(mon[:3]) or months.get(mon)
        if mval:
            try:
                dt = datetime(yr, int(mval), 1)
                return dt.strftime("%Y-%m-%d"), True
            except Exception:
                return None, False

    return None, False


def validate_language(code: Optional[str]) -> bool:
    if not code:
        return False
    c = str(code).strip()
    if len(c) < 2:
        return False
    return c in BCP47_BASIC or re.fullmatch(r"[a-zA-Z]{2,3}(?:-[a-zA-Z0-9]{2,8})*", c) is not None


def normalize_language(code: Optional[str]) -> Optional[str]:
    if not code:
        return None
    c = str(code).strip()
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


def normalize_currency(code: Optional[str]) -> Optional[str]:
    """
    Normaliza una representación de moneda a su código ISO-4217 de tres letras.
    - Si la entrada ya es un código ISO válido (p. ej. 'EUR' o 'eur'), lo devuelve en mayúsculas.
    - Si la entrada es un símbolo ('$','€','£', 'R$'...), lo mapea al código ISO correspondiente.
    - Si la entrada contiene ruido (ej. 'EUR ' o 'USD
      ' o '€12.34'), intenta extraer el código/símbolo y mapearlo.
    - Devuelve None si no se puede normalizar.

    Esta función centraliza la lógica de normalización de moneda usada por enrich/integrate.
    """
    if not code:
        return None
    s = str(code).strip()
    if s == "":
        return None
    # Si ya es un código ISO (normalizado)
    up = s.upper()
    # limpiar caracteres no alfabéticos (ej: 'usd.' -> 'USD')
    letters = ''.join([c for c in up if c.isalpha()])
    if letters and letters in ISO4217:
        return letters
    # símbolo directo
    if s in SYMBOL_TO_ISO:
        return SYMBOL_TO_ISO[s]
    # comprobar primer caracter simbólico (p. ej. '€12.34')
    for sym, iso in SYMBOL_TO_ISO.items():
        if sym in s:
            return iso
    # intentar extraer tres-letras en el string
    m = re.search(r"([A-Z]{3})", up)
    if m:
        cand = m.group(1)
        if cand in ISO4217:
            return cand
    return None


def nulls_by_column(rows: List[Mapping[str, object]], columns: List[str]) -> Dict[str, int]:
    counts = {c: 0 for c in columns}
    for r in (rows or []):
        if not isinstance(r, Mapping):
            for c in columns:
                counts[c] += 1
            continue
        for c in columns:
            v = r.get(c)
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


def get_quality_thresholds() -> Dict[str, float]:
    """Devuelve umbrales recomendados para validar la rúbrica (valores en % excepto duplicados).
    Estos pueden usarse por scripts de validación para determinar 'Excelente'."""
    return {
        "porcentaje_isbn13_validos": 90.0,
        "porcentaje_fechas_validas": 95.0,
        "porcentaje_idiomas_validos": 95.0,
        "porcentaje_monedas_validas": 90.0,
        "completitud_promedio": 90.0,
        "duplicados_isbn13": 0,
        "duplicados_book_id": 0,
    }


__all__ = [
    "normalize_whitespace",
    "parse_date_to_iso",
    "validate_language",
    "normalize_language",
    "validate_currency",
    "nulls_by_column",
    "listify",
    "uniq_preserve",
    "compute_quality_metrics",
    "write_quality_json",
    "get_quality_thresholds",
    # nueva exportación
    "normalize_currency",
]
