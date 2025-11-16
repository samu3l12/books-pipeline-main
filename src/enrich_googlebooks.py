"""
Enriquecimiento con Google Books API.
Lee landing/goodreads_books.json y genera landing/googlebooks_books.csv
Campos: gb_id, title, subtitle, authors, publisher, pub_date, language, categories, isbn13, isbn10, price_amount, price_currency
"""
from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional
import re
import time

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from dotenv import load_dotenv

from utils_isbn import is_valid_isbn13, try_normalize_isbn, extract_isbns_from_text
from utils_quality import normalize_language

GOOGLE_API = "https://www.googleapis.com/books/v1/volumes"

SYMBOL_TO_ISO = {
    "$": "USD",
    "\u20ac": "EUR",
    "\u00a3": "GBP",
    "\u00a5": "JPY",
    "R$": "BRL",
}


@dataclass
class GBRow:
    gb_id: Optional[str]
    title: Optional[str]
    subtitle: Optional[str]
    authors: Optional[str]  # lista serializada separada por ';'
    publisher: Optional[str]
    pub_date: Optional[str]
    language: Optional[str]
    categories: Optional[str]  # lista serializada separada por ';'
    isbn13: Optional[str]
    isbn10: Optional[str]
    price_amount: Optional[float]
    price_currency: Optional[str]
    paginas: Optional[int]
    formato: Optional[str]


def _session(timeout: int = 15) -> requests.Session:
    s = requests.Session()
    retries = Retry(total=4, backoff_factor=0.5, status_forcelist=(429, 500, 502, 503, 504), allowed_methods=("GET",))
    adapter = HTTPAdapter(max_retries=retries)
    s.mount("http://", adapter)
    s.mount("https://", adapter)
    s.headers.update({
        "User-Agent": "books-pipeline/1.0 (+https://example.local)",
        "Accept": "application/json",
    })
    return s


def _search_google_books(api_key: Optional[str], isbn13: Optional[str], title: Optional[str], author: Optional[str], publisher: Optional[str] = None, subject: Optional[str] = None, lccn: Optional[str] = None, oclc: Optional[str] = None, raw_query: Optional[str] = None) -> Optional[Dict]:
    """Construye la query `q` usando los prefijos soportados por la API.
    Prioridad: si se recibe un ISBN-13 válido se usa `isbn:...` exclusivamente.
    Si no hay ISBN válido, se concatena cualquier campo disponible con sus prefijos
    (intitle:, inauthor:, inpublisher:, subject:, lccn:, oclc:) para enviar una búsqueda lo más completa posible.
    KEYWORDS: SEARCH_QUERY_BUILD, GB_API_CALL
    """
    params: Dict[str, str] = {"maxResults": "1"}
    country = os.getenv("GOOGLE_BOOKS_COUNTRY")
    if country:
        params["country"] = country
    if api_key:
        params["key"] = api_key
    # si se pasa isbn13 válido, usar búsqueda por ISBN (más precisa)
    if isbn13 and is_valid_isbn13(isbn13):
        # EXTRACT_ISBN: búsqueda por ISBN
        params["q"] = f"isbn:{isbn13}"
    else:
        # Estrategia: buscar por título completo SIN prefijos para maximizar coincidencias
        # (replicar comportamiento del navegador que devuelve resultados con query simple)
        # Opcionalmente añadir autor con prefijo inauthor: para afinar
        # KEYWORDS: TITLE_SEARCH, AUTHOR_FILTER
        q_parts: List[str] = []

        if title:
            # Búsqueda simple por título (sin prefijo intitle:) para obtener más resultados
            q_parts.append(title.replace(' ', '+'))

        # Si hay autor, añadirlo con prefijo para mejorar la precisión
        if author:
            q_parts.append(f"inauthor:{author.replace(' ', '+')}")

        # Campos opcionales adicionales (si existen en el JSON)
        if publisher:
            q_parts.append(f"inpublisher:{publisher.replace(' ', '+')}")
        if subject:
            # subject puede ser lista serializada; tomar primer elemento si hay ';'
            subj = subject.split(";")[0] if isinstance(subject, str) and ";" in subject else subject
            q_parts.append(f"subject:{subj.replace(' ', '+')}")
        if lccn:
            q_parts.append(f"lccn:{lccn.replace(' ', '+')}")
        if oclc:
            q_parts.append(f"oclc:{oclc.replace(' ', '+')}")

        # Si no hay partes construidas, intentar raw_query o variables de entorno
        if not q_parts:
            if raw_query:
                # RAW_QUERY fallback
                params["q"] = raw_query.replace(' ', '+')
            else:
                # Fallback: leer variables globales de entorno (compatibilidad)
                pub_env = os.getenv("GB_PUBLISHER")
                subj_env = os.getenv("GB_SUBJECT")
                lccn_env = os.getenv("GB_LCCN")
                oclc_env = os.getenv("GB_OCLC")
                if pub_env:
                    q_parts.append(f"inpublisher:{pub_env.replace(' ', '+')}")
                if subj_env:
                    q_parts.append(f"subject:{subj_env.replace(' ', '+')}")
                if lccn_env:
                    q_parts.append(f"lccn:{lccn_env.replace(' ', '+')}")
                if oclc_env:
                    q_parts.append(f"oclc:{oclc_env.replace(' ', '+')}")
                if not q_parts:
                    return None

        params["q"] = "+".join(q_parts)
    try:
        s = _session()
        resp = s.get(GOOGLE_API, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        items = data.get("items") or []
        return items[0] if items else None
    except Exception:
        return None


def _map_gb_item(item: Dict) -> GBRow:
    # MAPEO: convierte un 'item' de Google Books a la fila GBRow usada en el CSV
    # KEYWORDS: MAP_ITEM, PARSE_VOLUMEINFO
    vol = (item or {}).get("volumeInfo", {})
    sale = (item or {}).get("saleInfo", {})
    ids = {i.get("type"): i.get("identifier") for i in vol.get("industryIdentifiers", []) if isinstance(i, dict)}
    isbn10 = ids.get("ISBN_10")
    isbn13 = ids.get("ISBN_13")
    i10, i13 = try_normalize_isbn(isbn13 or isbn10)
    isbn10 = i10 or isbn10
    isbn13 = i13 or isbn13

    price = None
    currency = None
    retail = sale.get("retailPrice")
    if isinstance(retail, dict):
        price = retail.get("amount")
        currency = retail.get("currencyCode")
        # normalizar tipos
        # conservar representaci\u00f3n original para detectar s\u00edmbolos si los hubiera
        orig_price = price
        try:
            price = float(price) if price is not None else None
        except Exception:
            # intentar extraer n\u00famero si viene como string con s\u00edmbolo
            try:
                cleaned = re.sub(r"[^0-9.,-]", "", str(orig_price))
                cleaned = cleaned.replace(',', '.')
                price = float(cleaned) if cleaned not in (None, "") else None
            except Exception:
                price = None
        if currency:
            currency = str(currency).upper().strip()
            # si currency es un s\u00edmbolo no ISO, mapear
            if not re.fullmatch(r"[A-Z]{3}", currency):
                currency = SYMBOL_TO_ISO.get(currency, None)
        else:
            # intentar detectar s\u00edmbolo en la representaci\u00f3n original si existe
            if isinstance(orig_price, str):
                m = re.search(r"([\u20ac$\u00a3\u00a5])", orig_price)
                if m:
                    currency = SYMBOL_TO_ISO.get(m.group(1))

    authors = vol.get("authors") or []
    cats = vol.get("categories") or []
    page_count = vol.get("pageCount")
    try:
        page_count = int(page_count) if page_count is not None else None
    except Exception:
        page_count = None
    formato = vol.get("printType") or vol.get("format")
    formato = str(formato).strip() if formato else None

    return GBRow(
        gb_id=item.get("id"),
        title=vol.get("title"),
        subtitle=vol.get("subtitle"),
        authors=";".join([a for a in authors if a]) if authors else None,
        publisher=vol.get("publisher"),
        pub_date=vol.get("publishedDate"),
        language=normalize_language(vol.get("language")),
        categories=";".join([c for c in cats if c]) if cats else None,
        isbn13=isbn13,
        isbn10=isbn10,
        price_amount=price,
        price_currency=currency,
        paginas=page_count,
        formato=formato,
    )


def enrich(input_json: str, output_csv: str, use_api: bool = True) -> int:
    # Cargar variables de entorno y exigir clave API: la política del proyecto requiere
    # usar siempre la Google Books API para enriquecimiento; no se generan datos sintéticos.
    load_dotenv()
    api_key = os.getenv("GOOGLE_BOOKS_API_KEY")
    if not api_key:
        raise RuntimeError("GOOGLE_BOOKS_API_KEY no encontrada. Añadirla en .env o en variables de entorno para proceder.")

    with open(input_json, "r", encoding="utf-8") as f:
        payload = json.load(f)
    recs = payload.get("records", [])

    rows: List[GBRow] = []
    for r in recs:
        isbn13 = r.get("isbn13")
        title = r.get("title")
        author = r.get("author")
        book_url = r.get("book_url") or ""
        # EXTRAER_ISBN: intentar extraer ISBN desde el registro (title/book_url) si no existe isbn13
        if not isbn13:
            candidates = extract_isbns_from_text((title or "") + " " + (book_url or ""))
            for cand in candidates:
                i10, i13 = try_normalize_isbn(cand)
                if i13:
                    isbn13 = i13
                    break
        # extraer otros campos potenciales del JSON y pasarlos a la API
        publisher = r.get("publisher") or r.get("editorial")
        subject = r.get("categories") or r.get("categoria")
        lccn = r.get("lccn")
        oclc = r.get("oclc")
        raw_query = " ".join([s for s in [title, author] if s]) or None
        # SEARCH_CALL: Forzar llamada a la API (siempre); pausa ética aplicada más abajo
        item = _search_google_books(api_key, isbn13, title, author, publisher=publisher, subject=subject, lccn=lccn, oclc=oclc, raw_query=raw_query)
        if item:
            # MAPEAR_ITEM: mapear y guardar
            row = _map_gb_item(item)
            rows.append(row)
        else:
            # No crear registros sintéticos; registrar para trazabilidad y continuar
            print(f"[enrich] No match Google Books -> title='{title}' author='{author}' isbn13='{isbn13}'")
        # RATE_LIMIT_PAUSE: Respetar pausa ética entre llamadas
        time.sleep(0.6)

    out_p = Path(output_csv)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "gb_id","title","subtitle","authors","publisher","pub_date","language","categories","isbn13","isbn10","price_amount","price_currency","paginas","formato"
    ]
    with out_p.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            row = {k: getattr(r, k) for k in fieldnames}
            # asegurar valores serializables
            if row.get("paginas") is None:
                row["paginas"] = ""
            if row.get("formato") is None:
                row["formato"] = ""
            # forzar isbn como str vacio si None para evitar coerciones posteriores
            if row.get("isbn10") is None:
                row["isbn10"] = ""
            if row.get("isbn13") is None:
                row["isbn13"] = ""
            writer.writerow(row)
    return len(rows)


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Enriquecimiento Google Books -> CSV")
    parser.add_argument("--input", default=str(Path(__file__).resolve().parents[1] / "landing" / "goodreads_books.json"))
    parser.add_argument("--output", default=str(Path(__file__).resolve().parents[1] / "landing" / "googlebooks_books.csv"))
    args = parser.parse_args()

    n = enrich(args.input, args.output)
    print(f"Escritas {n} filas en {args.output}")


if __name__ == "__main__":
    main()
