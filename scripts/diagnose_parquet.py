"""Diagnóstico rápido de parquets y landing files.
Imprime un JSON con métricas y ejemplos que responden a:
 - qué registros de Goodreads sin ISBN fueron rellenados en `dim_book.parquet` con ISBN de Google Books
 - si existen ISBN en `dim_book.parquet` que no aparecen en ninguna de las fuentes de landing (posible sintético)
 - ejemplos y conteos para investigar gb_only / matches

Uso: python scripts/diagnose_parquet.py

Nota: script solo hace lectura de archivos en `landing/` y `standard/` y escribe en stdout.
"""

import json
import re
import unicodedata
import sys
from pathlib import Path

import pandas as pd

BASE = Path(__file__).resolve().parents[1]
LANDING = BASE / "landing"
STANDARD = BASE / "standard"


def norm(s):
    if s is None:
        return None
    s = str(s)
    s = ''.join(c for c in unicodedata.normalize('NFKD', s) if not unicodedata.combining(c))
    s = re.sub(r'[^0-9A-Za-z\s]', ' ', s)
    s = re.sub(r'\s+', ' ', s).strip().lower()
    return s or None


def safe_read_goodreads(path: Path):
    if not path.exists():
        return pd.DataFrame()
    with open(path, 'r', encoding='utf-8') as f:
        j = json.load(f)
    return pd.DataFrame(j.get('records', []))


def main():
    out = {}
    try:
        gr_path = LANDING / 'goodreads_books.json'
        gb_path = LANDING / 'googlebooks_books.csv'
        dim_path = STANDARD / 'dim_book.parquet'
        detail_path = STANDARD / 'book_source_detail.parquet'

        gr = safe_read_goodreads(gr_path)
        try:
            gb = pd.read_csv(gb_path, dtype={'isbn13': 'string', 'isbn10': 'string'}) if gb_path.exists() else pd.DataFrame()
        except Exception:
            gb = pd.DataFrame()

        try:
            dim = pd.read_parquet(dim_path) if dim_path.exists() else pd.DataFrame()
        except Exception as e:
            print(json.dumps({'error': f'no se pudo leer {dim_path}: {e}'}), flush=True)
            sys.exit(1)

        # normalizaciones para matching
        if 'title' in gr.columns:
            gr['titulo'] = gr['title']
        if 'author' in gr.columns:
            gr['autor_principal'] = gr['author']
        gr['_t'] = gr.get('titulo', pd.Series([None] * len(gr))).apply(norm)
        gr['_a'] = gr.get('autor_principal', pd.Series([None] * len(gr))).apply(lambda x: norm(x.split(';')[0]) if isinstance(x, str) else norm(x))

        if not gb.empty:
            gb['_t'] = gb.get('title', pd.Series([None] * len(gb))).apply(norm)
            gb['_a'] = gb.get('authors', pd.Series([None] * len(gb))).apply(lambda s: norm(s.split(';')[0]) if isinstance(s, str) and s != '' else None)

        dim['_t'] = dim.get('titulo', pd.Series([None] * len(dim))).apply(norm)
        dim['_a'] = dim.get('autor_principal', pd.Series([None] * len(dim))).apply(lambda s: norm(s.split(';')[0]) if isinstance(s, str) and s != '' else None)

        # metrics: how many gr had isbn originally
        gr_total = len(gr)
        gr_with_isbn_orig = int(gr['isbn13'].notna().sum()) if 'isbn13' in gr.columns else 0
        gr_no_isbn_orig = gr_total - gr_with_isbn_orig

        assigned = 0
        ambiguous = 0
        examples_assigned = []

        gr_no_isbn_df = gr[gr.get('isbn13').isna()] if 'isbn13' in gr.columns else gr.copy()

        for i, row in gr_no_isbn_df.iterrows():
            t = row.get('_t')
            a = row.get('_a')
            if t is None and a is None:
                continue
            if t is not None and a is not None:
                candidates = dim[(dim['_t'] == t) & (dim['_a'] == a)]
            elif t is not None:
                candidates = dim[dim['_t'] == t]
            elif a is not None:
                candidates = dim[dim['_a'] == a]
            else:
                candidates = dim.iloc[0:0]

            if len(candidates) == 1:
                r = candidates.iloc[0]
                if pd.notna(r.get('isbn13')) and str(r.get('isbn13')).strip() != '':
                    assigned += 1
                    if len(examples_assigned) < 5:
                        examples_assigned.append({
                            'gr_index': int(i) + 1,
                            'titulo': row.get('titulo'),
                            'matched_dim_idx': int(r.name),
                            'isbn13': r.get('isbn13'),
                            'provenance': r.get('provenance'),
                        })
            elif len(candidates) > 1:
                ambiguous += 1

        # synthetic isbn check
        dim_isbns = set([str(x) for x in dim['isbn13'].dropna().unique().tolist()]) if 'isbn13' in dim.columns else set()
        gr_isbns = set([str(x) for x in gr['isbn13'].dropna().unique().tolist()]) if 'isbn13' in gr.columns else set()
        gb_isbns = set([str(x) for x in gb['isbn13'].dropna().unique().tolist()]) if (not gb.empty and 'isbn13' in gb.columns) else set()
        synthetic = [x for x in dim_isbns if x not in gr_isbns and x not in gb_isbns]

        # provenance checks
        prov_gb_count = 0
        prov_examples = []
        for idx, r in dim.iterrows():
            p = r.get('provenance')
            try:
                d = json.loads(p) if p and isinstance(p, str) else (p if isinstance(p, dict) else {})
            except Exception:
                d = {}
            if isinstance(d, dict) and d.get('isbn13') == 'google_books':
                prov_gb_count += 1
                if len(prov_examples) < 5:
                    prov_examples.append({'dim_idx': int(idx), 'titulo': r.get('titulo'), 'isbn13': r.get('isbn13')})

        out.update({
            'gr_total': gr_total,
            'gr_with_isbn_orig': int(gr_with_isbn_orig),
            'gr_no_isbn_orig': int(gr_no_isbn_orig),
            'assigned_from_dim_count': int(assigned),
            'ambiguous_matches': int(ambiguous),
            'assigned_examples': examples_assigned,
            'dim_isbns_total': len(dim_isbns),
            'synthetic_isbns': synthetic,
            'prov_gb_count': int(prov_gb_count),
            'prov_examples': prov_examples,
        })

    except Exception as e:
        out = {'error': str(e)}

    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()

