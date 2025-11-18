"""
Scraper de Goodreads para obtener una muestra de libros desde una búsqueda pública.
Guarda landing/goodreads_books.json con registros y metadatos.

Comentarios añadidos: explicar supuestos, selectores clave y comportamiento de pausa/reintentos.
"""
from __future__ import annotations
import json
import random
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import bs4
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from dotenv import load_dotenv

from utils_isbn import extract_isbns_from_text, try_normalize_isbn
# importar helper de logging movido a work/
try:
    # preferir importar desde el paquete work presente en el repo
    import sys
    from pathlib import Path as _P
    repo_root = _P(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from work.utils_logging import log_request_csv, ensure_work_dirs
    ensure_work_dirs()  # crear dirs si es necesario
except Exception:
    # si no está disponible, definir un no-op para no romper flujo (se recomienda tener work/ en sys.path)
    def log_request_csv(*args, **kwargs):
        return None
    def ensure_work_dirs(*args, **kwargs):
        return None


BASE_URL = "https://www.goodreads.com/search"
# Rotación simple de User-Agents comunes
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36",
]
DEFAULT_UA = USER_AGENTS[0] + " (compatible; BooksPipeline/1.0)"


@dataclass
class GoodreadsRecord:
    title: Optional[str]
    author: Optional[str]
    rating: Optional[float]
    ratings_count: Optional[int]
    book_url: Optional[str]
    isbn10: Optional[str]
    isbn13: Optional[str]


def _build_session(timeout: int = 15) -> requests.Session:
    """Session con reintentos y backoff exponencial para peticiones robustas.

    Supuestos clave:
    - Se respeta la política de pausas (pausa aplicada en el bucle principal).
    - Si se usa proveedor de scraping (SCRAPER_API_URL), se delega la petición a ese proveedor.
    KEYWORDS: SCRAPER_SESSION, RATE_LIMIT_PAUSE
    """
    s = requests.Session()
    retries = Retry(
        total=5,
        backoff_factor=0.6,  # ~0.6, 1.2, 2.4, ...
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET",),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retries, pool_connections=10, pool_maxsize=10)
    s.mount("http://", adapter)
    s.mount("https://", adapter)
    s.headers.update({
        "User-Agent": DEFAULT_UA,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9,es;q=0.8",
        "Cache-Control": "no-cache",
        "Pragma": "no-cache",
    })
    s.timeout = timeout  # tipo: ignore[attr-defined]
    return s


def _scraper_api_wrap(target_url: str) -> Tuple[str, Dict[str, str]]:
    """Wrapper deshabilitado: no se utiliza proveedor externo de scraping en este proyecto.

    Mantener la función por compatibilidad pero siempre devuelve la URL destino sin modificar
    y sin parámetros adicionales. Esto simplifica el flujo y evita dependencias externas.
    KEYWORDS: SCRAPER_API_DISABLED
    """
    # No se usa proveedor externo en esta versión: devolver la URL tal cual
    return target_url, {}


def fetch_search_html(session: requests.Session, query: str, page: int = 1, timeout: int = 15) -> str:
    params = {"q": query, "page": str(page)}
    # Construye URL destino de Goodreads
    from urllib.parse import urlencode

    target = f"{BASE_URL}?{urlencode(params)}"
    # Si hay proveedor de scraping, reescribe
    url, extra = _scraper_api_wrap(target)

    # Cabeceras con UA rotatorio y Referer
    headers = {
        "User-Agent": random.choice(USER_AGENTS) + " (compatible; BooksPipeline/1.0)",
        "Referer": "https://www.google.com/",
    }
    # Logging: medir tiempo y registrar request
    start = time.time()
    try:
        resp = session.get(url, params=extra if extra else None, headers=headers, timeout=timeout)
        duration = time.time() - start
        # registrar request en work/logs/requests
        try:
            log_request_csv("goodreads", "scrape", url, resp.status_code, None, duration, headers.get("User-Agent"), None)
        except Exception:
            pass
        resp.raise_for_status()
        return resp.text
    except Exception as e:
        duration = time.time() - start
        try:
            log_request_csv("goodreads", "scrape", url, getattr(e, 'response', None).status_code if getattr(e, 'response', None) is not None else None, None, duration, headers.get("User-Agent"), str(e))
        except Exception:
            pass
        raise


def parse_search_page(html: str) -> List[GoodreadsRecord]:
    soup = bs4.BeautifulSoup(html, "lxml")
    rows = soup.select("table.tableList tr")
    out: List[GoodreadsRecord] = []
    for tr in rows:
        # Título
        title_tag = tr.select_one("a.bookTitle span")
        title = title_tag.get_text(strip=True) if title_tag else None
        # Autor principal
        author_tag = tr.select_one("a.authorName span")
        author = author_tag.get_text(strip=True) if author_tag else None
        # Rating y recuento
        mini = tr.select_one("span.minirating")
        rating = None
        ratings_count = None
        if mini:
            # usar get_text() y normalizar saltos de línea para evitar advertencias estáticas
            txt = mini.get_text() if mini else ""
            txt = txt.replace('\n', ' ').strip()
            try:
                parts = txt.split(" avg rating", 1)
                if parts:
                    # soporta "4.12" o "avg rating 4.12" según variación
                    token = parts[0].split()[-1]
                    rating = float(token)
            except Exception:
                rating = None
            try:
                if "ratings" in txt:
                    num_str = txt.split("ratings")[0].split("—")[-1].strip().replace(",", "")
                    ratings_count = int(num_str)
            except Exception:
                ratings_count = None
        # URL libro
        a_title = tr.select_one("a.bookTitle")
        book_url = f"https://www.goodreads.com{a_title.get('href')}" if a_title and a_title.get("href") else None
        # Heurística de ISBN
        isbn10 = None
        isbn13 = None
        # usar get_text con keyword `separator` y strip manual para evitar advertencias de tipos
        isbns = extract_isbns_from_text(tr.get_text() + " " + (book_url or ""))
        for cand in isbns:
            i10, i13 = try_normalize_isbn(cand)
            isbn10 = isbn10 or i10
            isbn13 = isbn13 or i13
            if isbn13:
                break
        out.append(GoodreadsRecord(title, author, rating, ratings_count, book_url, isbn10, isbn13))
    return out


def scrape_goodreads(query: str, max_records: int = 15, min_pause_s: float = 0.8, max_pages: int = 3, timeout: int = 15) -> Dict[str, object]:
    load_dotenv()
    session = _build_session(timeout=timeout)

    records: List[GoodreadsRecord] = []
    page = 1
    errors = 0
    while len(records) < max_records and page <= max_pages:
        try:
            html = fetch_search_html(session, query, page=page, timeout=timeout)
        except Exception as e:
            errors += 1
            if errors >= 2:  # falla suave tras 2 intentos
                break
            # backoff aleatorio adicional
            time.sleep(min_pause_s + random.random())
            continue
        page_recs = parse_search_page(html)
        records.extend(page_recs)
        # Pausa ética aleatoria entre 0.8–1.6s aprox
        time.sleep(min_pause_s + random.random() * 0.8)
        page += 1
    # Truncar y filtrar
    deduped: List[GoodreadsRecord] = []
    seen = set()
    for r in records:
        key = ((r.title or "").strip().lower(), (r.author or "").strip().lower())
        if key in seen:
            continue
        seen.add(key)
        if r.title and r.author:
            deduped.append(r)
        if len(deduped) >= max_records:
            break

    meta = {
        "source": "goodreads",
        "base_url": BASE_URL,
        "query": query,
        "user_agent": session.headers.get("User-Agent", DEFAULT_UA),
        "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "record_count": len(deduped),
        "notes": {
            "selectors": {
                "title": "a.bookTitle span",
                "author": "a.authorName span",
                "minirating": "span.minirating",
                "book_url": "a.bookTitle",
            },
            "pauses_seconds": f">={min_pause_s:.1f}",
            "retry_backoff": "Retry(total=5,status=[429,5xx])",
            "scraper_api": False,
        },
    }

    return {
        "metadata": meta,
        "records": [asdict(r) for r in deduped],
    }


# KEYWORDS: SCRAPE_SECTION, ISBN_HEURISTICS, DEDUP_TITLE_AUTHOR

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Scraper Goodreads -> JSON")
    parser.add_argument("--query", default="data science")
    parser.add_argument("--max-records", type=int, default=15)
    parser.add_argument("--max-pages", type=int, default=3)
    parser.add_argument("--timeout", type=int, default=15)
    parser.add_argument("--output", default=str(Path(__file__).resolve().parents[1] / "landing" / "goodreads_books.json"))
    args = parser.parse_args()

    data = scrape_goodreads(args.query, max_records=args.max_records, max_pages=args.max_pages, timeout=args.timeout)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    # proteger si records es None
    print(f"Escrito {len(data.get('records') or [])} registros en {out_path}")


if __name__ == "__main__":
    main()
