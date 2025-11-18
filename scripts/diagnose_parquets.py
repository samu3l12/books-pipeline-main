"""
Script de diagnóstico reutilizable para este repo.
Lee `standard/dim_book.parquet` y `standard/book_source_detail.parquet` (solo lectura),
y opcionalmente los ficheros de `landing/` para comparar.
Imprime un JSON con: shapes, columnas, %nulos, filas más completas, conteos de ISBN,
muestreo de filas con ISBN, gb_only según matching por _csv_row/isbn13/_match_key, y
muestras de registros no emparejados.

Uso:
  python scripts/diagnose_parquets.py

No modifica archivos del repo.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path
import traceback

try:
    import pandas as pd
except Exception as e:
    print(json.dumps({"error": "pandas no disponible: %s" % str(e)}))
    sys.exit(1)

ROOT = Path(__file__).resolve().parents[1]
STANDARD = ROOT / 'standard'
LANDING = ROOT / 'landing'

def safe_read_parquet(p: Path):
    try:
        if not p.exists():
            return None, f"missing: {p}"
        df = pd.read_parquet(p)
        return df, None
    except Exception as e:
        return None, str(e)


def safe_read_json(p: Path):
    try:
        if not p.exists():
            return None, f"missing: {p}"
        import json as _json
        with open(p, 'r', encoding='utf-8') as f:
            data = _json.load(f)
        return data, None
    except Exception as e:
        return None, str(e)


def safe_read_csv(p: Path):
    try:
        if not p.exists():
            return None, f"missing: {p}"
        df = pd.read_csv(p, dtype={"isbn13": "string", "isbn10": "string"})
        return df, None
    except Exception as e:
        return None, str(e)


def norm_title(s: object):
    try:
        if s is None or (isinstance(s, float) and pd.isna(s)):
            return None
        import unicodedata, re
        st = str(s)
        # keep main part before ':' or '-'
        for sep in (":", " - ", " — ", "—", "-"):
            if sep in st:
                st = st.split(sep, 1)[0]
                break
        st = re.sub(r"\s+", " ", st).strip()
        st = ''.join(c for c in unicodedata.normalize('NFKD', st) if not unicodedata.combining(c))
        st = re.sub(r"[^0-9A-Za-z\s]", ' ', st).strip().lower()
        return st if st != '' else None
    except Exception:
        return None


def produce_report():
    out = {"meta": {"root": str(ROOT)}, "parquets": {}, "landing": {}, "matching": {}}

    # Read parquets
    dim, err = safe_read_parquet(STANDARD / 'dim_book.parquet')
    out['parquets']['dim'] = {"path": str(STANDARD / 'dim_book.parquet')}
    if err:
        out['parquets']['dim']['error'] = err
    else:
        out['parquets']['dim'].update({
            "shape": list(dim.shape),
            "columns": list(dim.columns),
            "null_pct_per_col": (dim.isna().mean()*100).round(2).to_dict(),
            "top_nonnull_rows": [{"idx": int(i), "nonnull_count": int(v)} for i,v in list(dim.notna().sum(axis=1).sort_values(ascending=False).head(10).items())],
            "isbn_count": int(dim['isbn13'].notna().sum()) if 'isbn13' in dim.columns else 0,
            "isbn_sample": dim[dim['isbn13'].notna()].head(10).to_dict(orient='records') if 'isbn13' in dim.columns else [],
        })

    detail, err = safe_read_parquet(STANDARD / 'book_source_detail.parquet')
    out['parquets']['detail'] = {"path": str(STANDARD / 'book_source_detail.parquet')}
    if err:
        out['parquets']['detail']['error'] = err
    else:
        out['parquets']['detail'].update({
            "shape": list(detail.shape),
            "columns": list(detail.columns),
            "null_pct_per_col": (detail.isna().mean()*100).round(2).to_dict(),
            "source_counts": detail.get('source_name', pd.Series()).fillna('null').value_counts(dropna=False).to_dict(),
            "sample_head": detail.head(10).to_dict(orient='records')
        })

    # Read landing files
    gr_json, e = safe_read_json(LANDING / 'goodreads_books.json')
    if e:
        out['landing']['goodreads'] = {"error": e}
    else:
        records = gr_json.get('records', []) if isinstance(gr_json, dict) else gr_json
        try:
            gr = pd.DataFrame(records)
        except Exception:
            gr = pd.DataFrame()
        out['landing']['goodreads'] = {"path": str(LANDING / 'goodreads_books.json'), "rows": len(gr), "columns": list(gr.columns)}

    gb, e = safe_read_csv(LANDING / 'googlebooks_books.csv')
    if e:
        out['landing']['googlebooks'] = {"error": e}
    else:
        out['landing']['googlebooks'] = {"path": str(LANDING / 'googlebooks_books.csv'), "rows": len(gb), "columns": list(gb.columns)}

    # Matching diagnostics only if we have landing dfs
    try:
        if isinstance(gr, pd.DataFrame) and not gr.empty and isinstance(gb, pd.DataFrame) and not gb.empty:
            # prepare keys
            if 'title' in gr.columns and 'titulo' not in gr.columns:
                gr['titulo'] = gr['title']
            if 'author' in gr.columns and 'autor_principal' not in gr.columns:
                gr['autor_principal'] = gr['author']
            for c in ('isbn13','isbn10'):
                if c in gr.columns:
                    gr[c] = gr[c].apply(lambda v: str(v).strip() if pd.notna(v) and str(v).strip()!='' else None)
            gr['_match_title'] = gr.get('titulo', pd.Series([None]*len(gr))).apply(lambda s: norm_title(s) if pd.notna(s) else None)
            gr['_match_author'] = gr.get('autor_principal', pd.Series([None]*len(gr))).apply(lambda s: (str(s).strip().lower() if pd.notna(s) else None))
            gr['_match_key'] = gr.apply(lambda r: (f"{r.get('_match_title')}|{r.get('_match_author')}" if r.get('_match_title') or r.get('_match_author') else None), axis=1)

            gb['_csv_row'] = gb.get('_csv_row', pd.Series(range(1, len(gb)+1)))
            if 'title' in gb.columns and 'titulo' not in gb.columns:
                gb['titulo'] = gb['title']
            if 'authors' in gb.columns and 'autores' not in gb.columns:
                gb['autores'] = gb['authors']
            gb['_match_title'] = gb.get('titulo', pd.Series([None]*len(gb))).apply(lambda s: norm_title(s) if pd.notna(s) else None)
            gb['_match_author'] = gb.get('autor_principal', pd.Series([None]*len(gb))).apply(lambda s: (str(s).strip().lower() if pd.notna(s) else None)) if 'autor_principal' in gb.columns else gb.get('_match_author', pd.Series([None]*len(gb)))
            gb['_match_key'] = gb.apply(lambda r: (f"{r.get('_match_title') or ''}|{r.get('_match_author') or ''}" if (r.get('_match_title') or r.get('_match_author')) else None), axis=1)

            # matched by isbn from gr with isbn
            gr_with_isbn = gr[gr.get('isbn13').notna()] if 'isbn13' in gr.columns else gr.iloc[0:0]
            gr_no_isbn = gr[gr.get('isbn13').isna()] if 'isbn13' in gr.columns else gr
            matched_rows = set()
            matched_isbns = set()
            # by isbn
            if not gr_with_isbn.empty and 'isbn13' in gb.columns:
                merged_isbn = pd.merge(gr_with_isbn.add_suffix('_gr'), gb.add_suffix('_gb'), left_on='isbn13_gr', right_on='isbn13_gb', how='left')
                if '_csv_row_gb' in merged_isbn.columns:
                    vals = merged_isbn['_csv_row_gb'].dropna().unique().tolist()
                    matched_rows.update([str(int(v)) if (isinstance(v,(int,float)) or (isinstance(v,str) and v.isdigit())) else str(v) for v in vals if v is not None])
                if 'isbn13_gb' in merged_isbn.columns:
                    matched_isbns.update([v for v in merged_isbn['isbn13_gb'].dropna().unique().tolist() if v is not None])

            # by key for those without isbn
            key_to_gb = {}
            if '_match_key' in gb.columns:
                for gi, grow in gb.iterrows():
                    k = grow.get('_match_key')
                    if k and pd.notna(k):
                        key_to_gb.setdefault(k, []).append(gi)
            for idx, row in gr_no_isbn.iterrows():
                k = row.get('_match_key')
                if k and k in key_to_gb:
                    for gi in key_to_gb[k]:
                        v = gb.loc[gi].get('_csv_row')
                        if pd.notna(v):
                            matched_rows.add(str(int(v)) if (isinstance(v,(int,float)) or (isinstance(v,str) and str(v).isdigit())) else str(v))

            gb_only = gb[~gb['_csv_row'].astype(str).isin(matched_rows)]

            out['matching'].update({
                'gr_total': int(len(gr)),
                'gr_with_isbn': int(len(gr_with_isbn)),
                'gr_no_isbn': int(len(gr_no_isbn)),
                'gb_total': int(len(gb)),
                'matched_gb_rows_count': len(matched_rows),
                'gb_only_count': int(len(gb_only)),
                'gb_only_sample': gb_only.head(10)[['titulo','_match_title','_match_author','isbn13','_csv_row']].to_dict(orient='records'),
                'unmatched_gr_sample': [],
            })
            # find unmatched gr rows
            unmatched = []
            for idx,row in gr.iterrows():
                m = False
                if 'isbn13' in gr.columns and pd.notna(row.get('isbn13')):
                    if 'isbn13' in gb.columns and row.get('isbn13') in gb['isbn13'].dropna().tolist():
                        m = True
                if not m:
                    k = row.get('_match_key')
                    if k and k in key_to_gb:
                        m = True
                if not m:
                    unmatched.append({'idx': int(idx)+1, 'titulo': row.get('titulo'), 'autor': row.get('autor_principal'), 'key': row.get('_match_key')})
            out['matching']['unmatched_gr_sample'] = unmatched[:10]

    except Exception as e:
        out['matching']['error'] = str(e)
        out['matching']['traceback'] = traceback.format_exc()

    return out


def sanitize_for_json(obj):
    """Convierte objetos numpy/pandas a tipos JSON-serializables (recursivo)."""
    try:
        import numpy as _np
        import pandas as _pd
    except Exception:
        _np = None
        _pd = None
    # None, bool, int, float, str
    if obj is None:
        return None
    if isinstance(obj, (bool, int, str)):
        return obj
    # floats: NaN/inf -> None
    if isinstance(obj, float):
        if obj != obj or obj == float('inf') or obj == float('-inf'):
            return None
        return obj
    # numpy scalars
    try:
        if _np is not None and isinstance(obj, (_np.integer, _np.floating, _np.bool_)):
            v = obj.item()
            if isinstance(v, float) and (v != v or v == float('inf') or v == float('-inf')):
                return None
            return v
    except Exception:
        pass
    # pandas Timestamp / datetime
    try:
        import datetime
        if (_pd is not None and isinstance(obj, _pd.Timestamp)) or isinstance(obj, datetime.datetime):
            try:
                return obj.isoformat()
            except Exception:
                return str(obj)
    except Exception:
        pass
    # list/tuple
    if isinstance(obj, (list, tuple)):
        out = []
        for v in obj:
            out.append(sanitize_for_json(v))
        return out
    # dict
    if isinstance(obj, dict):
        res = {}
        for k, v in obj.items():
            try:
                key = str(k)
            except Exception:
                key = repr(k)
            res[key] = sanitize_for_json(v)
        return res
    # pandas Series/DataFrame
    try:
        if _pd is not None and isinstance(obj, _pd.Series):
            return sanitize_for_json(obj.to_dict())
        if _pd is not None and isinstance(obj, _pd.DataFrame):
            return sanitize_for_json(obj.to_dict(orient='records'))
    except Exception:
        pass
    # numpy array
    try:
        if _np is not None and isinstance(obj, _np.ndarray):
            return sanitize_for_json(obj.tolist())
    except Exception:
        pass
    # fallback: string representation
    try:
        return str(obj)
    except Exception:
        return None


if __name__ == '__main__':
    try:
        report = produce_report()
        # sanitize report for JSON
        report_clean = sanitize_for_json(report)
        print(json.dumps(report_clean, ensure_ascii=False, indent=2))
    except Exception as e:
        print(json.dumps({"error": str(e), "traceback": traceback.format_exc()}))
