"""
Enriquecimiento con Google Books API.
Lee landing/goodreads_books.json y genera landing/googlebooks_books.csv
Campos: gb_id, title, subtitle, authors, publisher, pub_date, language, categories, isbn13, isbn10, price_amount, price_currency

Comentarios añadidos: supuestos sobre la API, encoding del parámetro q ( '+' en vez de espacios ), política de pausas y trazabilidad de candidatos.
"""
from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import re
import time

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from dotenv import load_dotenv

from utils_isbn import is_valid_isbn13, try_normalize_isbn, extract_isbns_from_text
from utils_quality import normalize_language, normalize_currency

# importar helper de logging movido a work/ — import robusto
try:
    from work.utils_logging import log_request_csv, ensure_work_dirs
    # crear dirs de trabajo si la implementación existe
    try:
        ensure_work_dirs()
    except Exception:
        pass
except Exception:
    # fallback no-op para permitir ejecución sin work/utils_logging
    def log_request_csv(*args, **kwargs):
        return None
    def ensure_work_dirs(*args, **kwargs):
        return None

GOOGLE_API = "https://www.googleapis.com/books/v1/volumes"

# Mapeo de símbolos a ISO-4217 usado localmente
SYMBOL_TO_ISO = {
    "$": "USD",
    "€": "EUR",
    "£": "GBP",
    "¥": "JPY",
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


def _levenshtein(a: str, b: str) -> int:
    """Distancia de Levenshtein (implementación eficiente en memoria).
    KEYWORD: LEVENSHTEIN

    Nota: se usa para medir similitud de títulos cuando no hay ISBN. Se combina
    con token overlap para robustez frente a reordenamientos.
    """
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    la, lb = len(a), len(b)
    if la < lb:
        # swap para usar menos memoria
        a, b = b, a
        la, lb = lb, la
    previous = list(range(lb + 1))
    for i, ca in enumerate(a, start=1):
        current = [i] + [0] * lb
        for j, cb in enumerate(b, start=1):
            add = previous[j] + 1
            delete = current[j - 1] + 1
            change = previous[j - 1] + (0 if ca == cb else 1)
            current[j] = min(add, delete, change)
        previous = current
    return previous[lb]


def _token_overlap(a: Optional[str], b: Optional[str]) -> float:
    """Calcula fracción de tokens compartidos entre a y b (0..1).
    KEYWORD: TOKEN_OVERLAP
    """
    if not a or not b:
        return 0.0
    ta = set([t for t in re.split(r"\W+", a.lower()) if t])
    tb = set([t for t in re.split(r"\W+", b.lower()) if t])
    if not ta or not tb:
        return 0.0
    inter = ta.intersection(tb)
    return len(inter) / max(len(ta), len(tb))


def _normalize_for_cmp(s: Optional[str]) -> str:
    if not s:
        return ""
    # quitar espacios extra, lower y quitar caracteres no alfanum para comparar
    return re.sub(r"\s+", " ", re.sub(r"[^0-9A-Za-z\s]", "", s)).strip().lower()


def _select_best_item(items: List[Dict], title: Optional[str], author: Optional[str], isbn13: Optional[str]) -> Tuple[Optional[Dict], List[Dict]]:
    """Selecciona el mejor item entre varios candidatos usando scoring y devuelve
    (mejor_item, lista_candidatos_con_score).
    Mejoras: si empate en puntuación, preferir item con ISBN-13; si sigue empate, preferir fecha de publicación más reciente.
    KEYWORD: SELECT_BEST_MATCH, MATCH_SCORING
    """
    if not items:
        return None, []
    tnorm = _normalize_for_cmp(title)
    anorm = _normalize_for_cmp(author)
    best = None
    best_score = -1.0
    candidates: List[Dict] = []

    def _extract_price_from_sale(sale: Dict) -> Tuple[Optional[float], Optional[str], Optional[str]]:
        # extrae precio de saleInfo: retailPrice.amount o offers[].listPrice.amountInMicros
        # devuelve (amount, currency, country)
        try:
            if not sale or not isinstance(sale, dict):
                return None, None, None
            # retailPrice.amount (float) preferente
            retail = sale.get('retailPrice')
            country = sale.get('country') or sale.get('saleability')
            if isinstance(retail, dict):
                amt = retail.get('amount')
                cur = retail.get('currencyCode')
                if amt is not None:
                    try:
                        return float(amt), (str(cur).upper().strip() if cur else None), (str(country).upper().strip() if country else None)
                    except Exception:
                        pass
            # offers[].listPrice.amountInMicros
            offers = sale.get('offers') or []
            for offer in offers:
                lp = offer.get('listPrice') or {}
                amt_micros = lp.get('amountInMicros')
                cur = lp.get('currencyCode')
                if amt_micros is not None:
                    try:
                        return float(amt_micros) / 1_000_000.0, (str(cur).upper().strip() if cur else None), (str(country).upper().strip() if country else None)
                    except Exception:
                        pass
            # intentar extraer de 'buyLink' domain si existe
            buy = sale.get('buyLink') or ''
            if isinstance(buy, str) and buy:
                # extraer TLD simple
                m = re.search(r"https?://(?:www\.)?[^/]+\.(?P<tld>[a-z]{2,3})(?:/|$)", buy)
                if m:
                    tld = m.group('tld').upper()
                    # mapear tld a moneda candidada (heurística simple)
                    tld_map = {'ES':'EUR','US':'USD','GB':'GBP','UK':'GBP','DE':'EUR','FR':'EUR','IT':'EUR'}
                    if tld in tld_map:
                        return None, tld_map[tld], tld
        except Exception:
            return None, None, None
        return None, None, None
# preguntar que es eso de estrer isbn13 desde enlaces
    for it in items:
        vol = it.get("volumeInfo", {})
        sale = it.get("saleInfo", {})
        ids = {i.get("type"): i.get("identifier") for i in vol.get("industryIdentifiers", []) if isinstance(i, dict)}
        it_isbn13 = ids.get("ISBN_13") or ids.get("ISBN-13")
        # intentar extraer isbn13 desde enlaces si no está en industryIdentifiers
        if not it_isbn13:
            for link_field in (it.get('selfLink'), (vol.get('infoLink') or ''), (vol.get('canonicalVolumeLink') or '')):
                if link_field and isinstance(link_field, str):
                    m = re.search(r"(97[89][0-9]{10})", link_field)
                    if m:
                        it_isbn13 = m.group(1)
                        break
        score = 0.0
        # ISBN exact match
        if isbn13 and it_isbn13:
            if _normalize_for_cmp(isbn13) == _normalize_for_cmp(it_isbn13):
                score += 100.0
        # título: combinar Levenshtein y token overlap
        it_title = vol.get("title") or ""
        n_it_title = _normalize_for_cmp(it_title)
        if tnorm and n_it_title:
            lev = _levenshtein(tnorm, n_it_title)
            maxlen = max(len(tnorm), len(n_it_title)) or 1
            title_ratio = 1.0 - (lev / maxlen)
            # token overlap añade robustez frente a reordenamientos/stopwords
            tok_ov = _token_overlap(tnorm, n_it_title)
            title_score = max(0.0, title_ratio) * 40.0 + tok_ov * 20.0
            score += title_score
        # autor overlap
        it_authors = vol.get("authors") or []
        it_authors_str = ";".join(it_authors) if isinstance(it_authors, list) else (str(it_authors) if it_authors else "")
        author_overlap = _token_overlap(anorm, _normalize_for_cmp(it_authors_str)) if anorm else 0.0
        score += author_overlap * 40.0

        # extraer precio si existe y añadir como tie-breaker info
        price_amt, price_cur, price_country = _extract_price_from_sale(sale)
        # normalizar moneda si aparece
        if price_cur:
            norm_cur = normalize_currency(price_cur)
            if norm_cur:
                price_cur = norm_cur

        # small tie-breakers: guardar metadata para desempate
        pub_date = vol.get("publishedDate")
        candidates.append({
            "gb_id": it.get("id"),
            "score": float(score),
            "isbn13": it_isbn13,
            "title": vol.get("title"),
            "authors": it_authors_str,
            "publishedDate": pub_date,
            "price_amount": price_amt,
            "price_currency": price_cur,
            "price_country": price_country,
        })

        if score > best_score:
            best_score = score
            best = it
        elif score == best_score and best is not None:
            # desempate: preferir item con isbn13
            best_ids = {i.get("type"): i.get("identifier") for i in (best.get("volumeInfo", {}) or {}).get("industryIdentifiers", []) if isinstance(i, dict)}
            best_isbn13 = best_ids.get("ISBN_13") or best_ids.get("ISBN-13")
            if it_isbn13 and not best_isbn13:
                best = it
                best_score = score
            else:
                # segundo desempate: preferir fecha de publicación más reciente (por año)
                def _year(d):
                    if not d or not isinstance(d, str):
                        return -1
                    m = re.match(r"^(\d{4})", d)
                    return int(m.group(1)) if m else -1
                cur_year = _year(pub_date)
                best_year = _year((best.get("volumeInfo", {}) or {}).get("publishedDate"))
                if cur_year > best_year:
                    best = it
                    best_score = score

    return best, candidates


def _search_google_books(api_key: Optional[str], isbn13: Optional[str], title: Optional[str], author: Optional[str], publisher: Optional[str] = None, subject: Optional[str] = None, lccn: Optional[str] = None, oclc: Optional[str] = None, raw_query: Optional[str] = None) -> Optional[Dict]:
    """Construye la query `q` usando los prefijos soportados por la API.
    Prioridad: si se recibe un ISBN-13 válido se usa `isbn:...` exclusivamente.
    Si no hay ISBN válido, se concatena cualquier campo disponible con sus prefijos
    (intitle:, inauthor:, inpublisher:, subject:, lccn:, oclc:) para enviar una búsqueda lo más completa posible.
    KEYWORDS: SEARCH_QUERY_BUILD, GB_API_CALL, QUERY_PLUS_SIGN, SELECT_BEST_MATCH
    Nota: se reemplazan espacios por '+' en cada segmento para asegurar encoding consistente en q parameter.
    """
    params: Dict[str, str] = {"maxResults": "5"}  # solicitar varios candidatos para scoring
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
        q_parts: List[str] = []

        if title:
            q_parts.append(title.replace(' ', '+'))

        if author:
            q_parts.append(f"inauthor:{author.replace(' ', '+')}")

        if publisher:
            q_parts.append(f"inpublisher:{publisher.replace(' ', '+')}")
        if subject:
            subj = subject.split(";")[0] if isinstance(subject, str) and ";" in subject else subject
            q_parts.append(f"subject:{subj.replace(' ', '+')}")
        if lccn:
            q_parts.append(f"lccn:{lccn.replace(' ', '+')}")
        if oclc:
            q_parts.append(f"oclc:{oclc.replace(' ', '+')}")

        # Si no hay partes construidas, intentar raw_query o variables de entorno
        if not q_parts:
            if raw_query:
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
        # registrar request: construir url completa con params para trazabilidad
        from urllib.parse import urlencode
        url = GOOGLE_API + "?" + urlencode(params)
        start = time.time()
        resp = s.get(GOOGLE_API, params=params, timeout=15)
        duration = time.time() - start
        # intentar registrar request
        try:
            log_request_csv("google_books", "search", url, resp.status_code if hasattr(resp, 'status_code') else None, None, duration, s.headers.get('User-Agent'), None)
        except Exception:
            pass
        resp.raise_for_status()
        data = resp.json()
        items = data.get("items") or []
        # Si hay varios items, elegir el mejor usando scoring y devolver candidatos para trazabilidad
        if len(items) >= 1:
            best, candidates = _select_best_item(items, title, author, isbn13)
            # normalizar currency en candidates (si aparece como símbolo)
            for c in candidates:
                if 'price_currency' in c and c.get('price_currency') and not re.fullmatch(r"[A-Z]{3}", str(c.get('price_currency'))):
                    cur = c.get('price_currency')
                    if isinstance(cur, str) and cur.strip() != '':
                        mapped = SYMBOL_TO_ISO.get(cur.strip(), None)
                        if mapped:
                            c['price_currency'] = mapped
                # si no tenemos currency pero tenemos price_country (ISO TLD), mapear
                if ('price_currency' not in c or c.get('price_currency') in (None, '')) and c.get('price_country'):
                    tld_map = {'ES':'EUR','US':'USD','GB':'GBP','UK':'GBP','DE':'EUR','FR':'EUR','IT':'EUR'}
                    pc = c.get('price_country')
                    if pc and pc in tld_map:
                        c['price_currency'] = tld_map[pc]

            return {"best": best, "candidates": candidates}
        return None
    except Exception as e:
        # registrar fallo
        try:
            log_request_csv("google_books", "search_error", GOOGLE_API, getattr(e, 'response', None).status_code if getattr(e, 'response', None) is not None else None, None, None, None, None, str(e))
        except Exception:
            pass
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
    # usar siempre la Google Books API para enriquecimiento;
    load_dotenv()
    api_key = os.getenv("GOOGLE_BOOKS_API_KEY")
    if not api_key:
        raise RuntimeError("GOOGLE_BOOKS_API_KEY no encontrada. Añadirla en .env o en variables de entorno para proceder.")
    # Leer pausa entre llamadas desde .env (segundos). Por defecto 0.6s.
    try:
        rate_limit = float(os.getenv("GOOGLE_BOOKS_RATE_LIMIT", "0.6"))
        if rate_limit < 0:
            rate_limit = 0.6
    except Exception:
        rate_limit = 0.6

    with open(input_json, "r", encoding="utf-8") as f:
        payload = json.load(f)
    recs = payload.get("records", [])

    rows: List[GBRow] = []
    candidates_out: List[Dict] = []
    csv_row_counter = 0
    for idx, r in enumerate(recs):
        isbn13 = r.get("isbn13")
        title = r.get("title")
        author = r.get("author")
        book_url = r.get("book_url") or ""
        rec_id = r.get("id") or book_url or f"rec_{idx}"
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
        # Realizar llamada a la API solicitando hasta 5 resultados (manejo ya implementado en _search_google_books)
        result = _search_google_books(api_key, isbn13, title, author, publisher=publisher, subject=subject, lccn=lccn, oclc=oclc, raw_query=raw_query)
        best_item = None
        cand_list = []
        best_score = None
        if result:
            best_item = result.get("best")
            cand_list = result.get("candidates") or []
            # MAPEAR_ITEM: mapear y guardar
            if best_item:
                row = _map_gb_item(best_item)
                # normalizar currency en row si necesario
                if row.price_currency and not re.fullmatch(r"[A-Z]{3}", str(row.price_currency)):
                    mapped = SYMBOL_TO_ISO.get(str(row.price_currency).strip(), None)
                    if mapped:
                        row.price_currency = mapped
                rows.append(row)
                csv_row_counter += 1
                # best_score: obtener de cand_list si existe
                best_score = None
                try:
                    best_score = max((float(c.get("score")) for c in cand_list if c.get("score") is not None), default=None)
                except Exception:
                    best_score = None
        else:
            # No crear registros sintéticos; registrar para trazabilidad y continuar
            print(f"[enrich] No match Google Books -> title='{title}' author='{author}' isbn13='{isbn13}'")
        # Registrar candidatos con mapeo al CSV (si existe) y al record original
        candidates_out.append({
            "rec_id": rec_id,
            "input_index": idx,
            "csv_row_number": csv_row_counter if best_item else None,
            "title": title,
            "author": author,
            "isbn13_input": isbn13,
            "candidates": cand_list,
            "best_score": best_score,
        })
        # RATE_LIMIT_PAUSE: Respetar pausa ética entre llamadas
        try:
            time.sleep(rate_limit)
        except Exception:
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
            # normalizar price_currency (ISO-4217) y price_amount
            if row.get("price_currency"):
                nc = normalize_currency(row.get("price_currency"))
                if nc:
                    row["price_currency"] = nc
            # formatear price_amount con '.' como separador decimal si existe
            if row.get("price_amount") is None or row.get("price_amount") == "":
                row["price_amount"] = ""
            else:
                try:
                    # asegurar que escrimos con punto decimal y no con comas
                    row["price_amount"] = str(float(row["price_amount"]))
                except Exception:
                    # conservar original como string si no puede convertirse
                    row["price_amount"] = str(row.get("price_amount"))
            writer.writerow(row)

    # Escribir fichero auxiliar de candidatos para trazabilidad
    cand_path = out_p.parent / "googlebooks_candidates.json"
    with cand_path.open("w", encoding="utf-8") as f:
        json.dump(candidates_out, f, ensure_ascii=False, indent=2)

    return len(rows)


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Enriquecimiento Google Books -> CSV")
    parser.add_argument("--input", default=str(Path(__file__).resolve().parents[1] / "landing" / "goodreads_books.json"))
    parser.add_argument("--output", default=str(Path(__file__).resolve().parents[1] / "landing" / "googlebooks_books.csv"))
    args = parser.parse_args()

    n = enrich(args.input, args.output)
    cand_path = Path(args.output).parent / "googlebooks_candidates.json"
    print(f"Escritas {n} filas en {args.output}")
    print(f"Candidatos escritos en {cand_path}")


if __name__ == "__main__":
    main()
