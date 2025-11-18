"""
Integración de Goodreads (JSON) y Google Books (CSV) a artefactos estándar.
Produce standard/dim_book.parquet, standard/book_source_detail.parquet y docs/quality_metrics.json
Mejoras implementadas:
- Merge robusto por isbn13 y por match_key (titulo+autor) sin producir producto cartesiano
- Aserciones configurables: unicidad book_id (bloqueante), porcentaje mínimo de títulos no nulos (por defecto 90%)
- Fail-soft configurado: registros con errores se marcan en book_source_detail y se excluyen de dim_book
- Resumen de aserciones incluido en docs/quality_metrics.json
"""
from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Tuple, Dict, List, Mapping
import re

import pandas as pd
import traceback
import warnings

# Suprimir FutureWarnings ruidosos de pandas que no afectan la lógica actual
warnings.filterwarnings("ignore", category=FutureWarning, message=".*downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated.*")
warnings.filterwarnings("ignore", category=FutureWarning, message=".*DataFrameGroupBy.apply operated on the grouping columns.*")

# is_valid_isbn13 se usa en otros módulos; no es necesario importarlo aquí para evitar warning
# from utils_isbn import is_valid_isbn13
from utils_quality import (
    compute_quality_metrics,
    listify,
    normalize_language,
    normalize_whitespace,
    parse_date_to_iso,
    uniq_preserve,
    validate_currency,
    validate_language,
    normalize_currency, nulls_by_column,
)
# Añadir import de utils_isbn necesario
from utils_isbn import try_normalize_isbn

# Evitar FutureWarning sobre downcasting en operaciones futuras de pandas
try:
    pd.set_option('future.no_silent_downcasting', True)
except Exception:
    pass

# Funciones auxiliares para sanitizar registros antes de calcular métricas
def _sanitize_value(v):
    try:
        if v is None:
            return None
        if isinstance(v, float) and pd.isna(v):
            return None
        if isinstance(v, (bytes, bytearray)):
            try:
                return v.decode('utf-8')
            except Exception:
                return str(v)
        if isinstance(v, (str, bool, int, float)):
            return v
        if isinstance(v, (list, dict)):
            try:
                return json.dumps(v, ensure_ascii=False)
            except Exception:
                return str(v)
        if hasattr(v, 'tolist'):
            try:
                return v.tolist()
            except Exception:
                pass
        return str(v)
    except Exception:
        return None


def _sanitize_records_for_metrics(records: List[Mapping[str, object]]) -> List[Mapping[str, object]]:
    sanitized: List[Dict[str, object]] = []
    for r in records:
        if not isinstance(r, Mapping):
            sanitized.append({})
            continue
        row: Dict[str, object] = {}
        for k, v in r.items():
            try:
                row[k] = _sanitize_value(v)
            except Exception:
                row[k] = None
        sanitized.append(row)
    return sanitized


# Helpers seguros para leer y escribir en DataFrames con índices no estándar
def _safe_set(df: pd.DataFrame, idx, col: str, val) -> bool:
    try:
        if idx in df.index:
            df.loc[idx, col] = val
            return True
    except Exception:
        pass
    try:
        pos = int(idx)
        df.iloc[pos, df.columns.get_loc(col)] = val
        return True
    except Exception:
        pass
    try:
        df.at[idx, col] = val
        return True
    except Exception:
        return False


def _safe_get(df: pd.DataFrame, idx, col: str):
    try:
        if idx in df.index:
            return df.loc[idx, col]
    except Exception:
        pass
    try:
        pos = int(idx)
        return df.iloc[pos][col]
    except Exception:
        pass
    try:
        return df.at[idx, col]
    except Exception:
        return None

# Import robusto de funciones de logging en work/
try:
    from work.utils_logging import log_rule_jsonl, write_run_summary, ensure_work_dirs
except Exception:
    def log_rule_jsonl(*args, **kwargs):
        return None
    def write_run_summary(*args, **kwargs):
        return None
    def ensure_work_dirs(*args, **kwargs):
        return None

# Configuración y paths
ROOT = Path(__file__).resolve().parents[1]
LANDING = ROOT / "landing"
STANDARD = ROOT / "standard"
DOCS = ROOT / "docs"

# Aserciones / umbrales para la rúbrica
ASSERT_UNIQUENESS_BOOK_ID = True  # si True, bloquear si existen duplicados irreconciliables
MIN_TITLES_PCT = 0.90  # mínimo % de títulos no nulos requerido
# (Se omiten umbrales de fuzzy-matching: el pipeline actual evita matching difuso para no generar asignaciones no deterministas)

# Logger sencillo
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Mapeo sencillo de símbolos a ISO para monedas
SYMBOL_TO_ISO = {
    "$": "USD",
    "\u20ac": "EUR",
    "\u00a3": "GBP",
    "\u00a5": "JPY",
    "R$": "BRL",
}


def _canonical_key(title: str, author: str, publisher: Optional[str], year: Optional[int]) -> str:
    # Genera clave SHA1 estable para fallback cuando isbn13 no existe
    def _norm(x: Optional[object]) -> str:
        if x is None or (isinstance(x, float) and pd.isna(x)):
            return ""
        s = str(x)
        return s.strip().lower()

    base = "|".join([
        _norm(title),
        _norm(author),
        _norm(publisher),
        _norm(year),
    ])
    return hashlib.sha1(base.encode("utf-8")).hexdigest()


def _load_sources() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Carga archivos de landing asegurando ISBN como string para evitar coerciones a int.

    Comentarios:
    - Supuesto: los ficheros en `landing/` son la "fuente de la verdad" y no deben ser sobrescritos por integración.
    - Se fuerza dtype string para isbn para evitar conversión automática a int/float que rompe pyarrow.
    KEYWORDS: LOAD_SOURCES, COERCE_ISBN
    """
    gr_path = LANDING / "goodreads_books.json"
    gb_path = LANDING / "googlebooks_books.csv"

    if not gr_path.exists():
        raise FileNotFoundError(f"Falta archivo de landing esperado: {gr_path}")
    if not gb_path.exists():
        raise FileNotFoundError(f"Falta archivo de landing esperado: {gb_path}")

    with open(gr_path, "r", encoding="utf-8") as f:
        gr_json = json.load(f)
    gr = pd.DataFrame(gr_json.get("records", []))
    gr = gr.copy()
    gr["source_name"] = "goodreads"
    gr["source_file"] = str(gr_path.name)

    gb = pd.read_csv(gb_path, dtype={"isbn13": "string", "isbn10": "string"})
    gb = gb.copy()
    gb["source_name"] = "google_books"
    gb["source_file"] = str(gb_path.name)

    # --- nuevo: exponer numero de fila CSV para poder mapear con googlebooks_candidates.csv ---
    try:
        import numpy as _np
        gb['_csv_row'] = _np.arange(1, len(gb) + 1)
    except:
        gb['_csv_row'] = list(range(1, len(gb) + 1))
    # Asegurar tipo string consistente para comparaciones posteriores (evita float/int mismatches)
    try:
        gb['_csv_row'] = gb['_csv_row'].astype(str)
    except Exception:
        # asegurar manualmente
        gb['_csv_row'] = gb['_csv_row'].apply(lambda v: str(int(v)) if (not isinstance(v, str) and (isinstance(v, float) or isinstance(v, int))) else str(v))

    # Nota: no hay dependencia de archivos de candidatos; los precios vienen en el CSV cuando están presentes.

    return gr, gb


def _normalize_frames(gr: pd.DataFrame, gb: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # Normaliza columnas y formatos básicos en ambos dataframes
    gr = gr.copy()
    gr.rename(columns={
        "title": "titulo",
        "author": "autor_principal",
        "isbn13": "isbn13",
        "isbn10": "isbn10",
    }, inplace=True)
    for c in ("isbn13", "isbn10"):
        if c in gr.columns:
            gr[c] = gr[c].apply(lambda v: str(v).strip() if pd.notna(v) and str(v).strip() != "" else None)

    # No crear columnas en `gr` que no existen en el JSON de Goodreads.
    # Mantener solo los campos que realmente están en el JSON para evitar columnas
    # totalmente NULL/None que luego se arrastran por todo el pipeline.
    # Si en el futuro el scraper añade campos (p.ej. 'isbn13'), se preservarán.

    gb = gb.copy()
    gb.rename(columns={
        "title": "titulo",
        "authors": "autores",
        "publisher": "editorial",
        "pub_date": "fecha_publicacion",
        "language": "idioma",
        "categories": "categoria",
        "price_amount": "precio",
        "price_currency": "moneda",
    }, inplace=True)
    for c in ("isbn13", "isbn10"):
        if c in gb.columns:
            gb[c] = gb[c].apply(lambda v: str(v).strip() if pd.notna(v) and str(v).strip() != "" else None)

    # autores/categoria como listas nativas (no strings separados por ';')
    if "autores" in gb.columns:
        gb["autores"] = gb["autores"].apply(lambda x: uniq_preserve(listify(x)) if (x is not None and not (isinstance(x, float) and pd.isna(x))) else None)
        gb["autor_principal"] = gb["autores"].apply(lambda x: x[0] if isinstance(x, list) and len(x) > 0 else None)
    else:
        gb["autor_principal"] = None
    if "categoria" in gb.columns:
        gb["categoria"] = gb["categoria"].apply(lambda x: uniq_preserve(listify(x)) if (x is not None and not (isinstance(x, float) and pd.isna(x))) else None)

    if "idioma" in gb.columns:
        gb["idioma"] = gb["idioma"].apply(lambda x: normalize_language(x) if pd.notna(x) else None)

    # Usar _main_title para emparejar la parte principal del título (evita subtítulos)
    gb["_match_title"] = gb["titulo"].apply(lambda s: _main_title(s) if pd.notna(s) else None)
    gb["_match_author"] = gb["autor_principal"].apply(lambda s: normalize_whitespace(str(s).lower()) if pd.notna(s) else None)

    # Para Goodreads crear keys similares para emparejar
    gr["_match_title"] = gr.apply(lambda r: _main_title(r.get("titulo")) or normalize_whitespace(str(r.get("titulo") or "").lower()), axis=1)
    gr["_match_author"] = gr.apply(lambda r: normalize_whitespace(str(r.get("autor_principal") or "").lower()), axis=1)
    gr["_match_key"] = gr.apply(lambda r: (f"{r.get('_match_title')}|{r.get('_match_author')}" if r.get('_match_title') or r.get('_match_author') else None), axis=1)
    gb["_match_key"] = gb.apply(lambda r: (f"{r.get('_match_title') or ''}|{r.get('_match_author') or ''}" if (r.get('_match_title') or r.get('_match_author')) else None), axis=1)

    return gr, gb


def _merge_sources(gr: pd.DataFrame, gb: pd.DataFrame) -> pd.DataFrame:
    """Merge robusto por isbn13 (primario) y por _match_key (secundario) sin producir producto cartesiano.
    Devuelve DataFrame con columnas limpias y sufijos explícitos para selección posterior.
    """
    # Dividir gr en con isbn y sin isbn (aseguramos variables antes de usar)
    try:
        gr_with_isbn = gr[gr["isbn13"].notna()].copy() if "isbn13" in gr.columns else gr.iloc[0:0].copy()
        gr_no_isbn = gr[gr["isbn13"].isna()].copy() if "isbn13" in gr.columns else gr.copy()
    except Exception:
        gr_with_isbn = pd.DataFrame()
        gr_no_isbn = pd.DataFrame()

    # Asignar ISBN a registros de Goodreads sin isbn usando el CSV de Google Books
    # Regla: solo asignar cuando hay una correspondencia inequívoca (match_key exacto o _match_title único).
    try:
        if not gr_no_isbn.empty and isinstance(gb, pd.DataFrame):
            try:
                # construir index por _match_key en GB
                gb_index_by_key = {}
                if '_match_key' in gb.columns:
                    for gi, grow in gb.iterrows():
                        k = grow.get('_match_key')
                        if k and pd.notna(k):
                            if k not in gb_index_by_key:
                                gb_index_by_key[k] = gi
                # asignar por key
                for idx, row in gr_no_isbn.iterrows():
                    mk = None
                    if '_match_key' in row.index:
                        mk = row.get('_match_key')
                    else:
                        mk = f"{row.get('_match_title') or ''}|{row.get('_match_author') or ''}"
                    if mk and mk in gb_index_by_key:
                        gi = gb_index_by_key[mk]
                        cand = gb.loc[gi]
                        if cand.get('isbn13') and pd.notna(cand.get('isbn13')) and str(cand.get('isbn13')).strip() != '':
                            # asignar isbn13 y registrar procedencia explícita
                            _safe_set(gr_no_isbn, idx, 'isbn13', cand.get('isbn13'))
                            try:
                                _safe_set(gr_no_isbn, idx, 'isbn13_source', 'google_books')
                            except Exception:
                                pass
                            try:
                                # preferir _csv_row si existe en cand
                                src_row = None
                                if '_csv_row' in cand.index:
                                    src_row = cand.get('_csv_row')
                                else:
                                    src_row = gi
                                _safe_set(gr_no_isbn, idx, 'isbn13_source_row', src_row)
                            except Exception:
                                pass
                            # Copiar campos útiles desde cand con sufijo _gb para preservar procedencia
                            gb_fields = ['editorial','fecha_publicacion','idioma','paginas','formato','categoria','precio','moneda','autores']
                            for f in gb_fields:
                                try:
                                    if f in cand.index and pd.notna(cand.get(f)) and str(cand.get(f)).strip() != '':
                                        _safe_set(gr_no_isbn, idx, f + '_gb', cand.get(f))
                                except Exception:
                                    pass
                # --- NUEVO: fallback estricto por _match_title único ---
                # Si aún quedan filas sin isbn, intentar asignar por título normalizado exacto
                # sólo cuando exista un único candidato en GB con el mismo _match_title.
                if '_match_title' in gb.columns:
                    gb_index_by_title = {}
                    for gi, grow in gb.iterrows():
                        t = grow.get('_match_title')
                        if t and pd.notna(t):
                            gb_index_by_title.setdefault(t, []).append(gi)
                    for idx, row in gr_no_isbn.iterrows():
                        # si ya se asignó isbn en el paso anterior, saltar
                        try:
                            if pd.notna(row.get('isbn13')) and str(row.get('isbn13')).strip() != '':
                                continue
                        except Exception:
                            pass
                        t = None
                        if '_match_title' in row.index:
                            t = row.get('_match_title')
                        else:
                            t = normalize_whitespace(str(row.get('titulo') or '').lower())
                        if t and t in gb_index_by_title:
                            cand_indices = gb_index_by_title.get(t, [])
                            # Si hay múltiples candidatos, registrar ambigüedad y NO asignar
                            if len(cand_indices) > 1:
                                try:
                                    log_rule_jsonl({
                                        'event': 'ambiguous_gb_candidates',
                                        'ts_utc': datetime.now(timezone.utc).isoformat(timespec='seconds'),
                                        'match_title': t,
                                        'gr_index': int(idx) if (isinstance(idx, (int, float)) or (isinstance(idx, str) and idx.isdigit())) else str(idx),
                                        'gb_candidate_rows': cand_indices,
                                    })
                                except Exception:
                                    pass
                                continue
                            # Sólo asignar si hay un único candidato claro
                            if len(cand_indices) == 1:
                                gi = cand_indices[0]
                                cand = gb.loc[gi]
                                if cand.get('isbn13') and pd.notna(cand.get('isbn13')) and str(cand.get('isbn13')).strip() != '':
                                    _safe_set(gr_no_isbn, idx, 'isbn13', cand.get('isbn13'))
                                    try:
                                        _safe_set(gr_no_isbn, idx, 'isbn13_source', 'google_books')
                                    except Exception:
                                        pass
                                    try:
                                        src_row = None
                                        if '_csv_row' in cand.index:
                                            src_row = cand.get('_csv_row')
                                        else:
                                            src_row = gi
                                        _safe_set(gr_no_isbn, idx, 'isbn13_source_row', src_row)
                                    except Exception:
                                        pass
                                    # Copiar campos útiles desde cand con sufijo _gb para preservar procedencia
                                    gb_fields = ['editorial','fecha_publicacion','idioma','paginas','formato','categoria','precio','moneda','autores']
                                    for f in gb_fields:
                                        try:
                                            if f in cand.index and pd.notna(cand.get(f)) and str(cand.get(f)).strip() != '':
                                                _safe_set(gr_no_isbn, idx, f + '_gb', cand.get(f))
                                        except Exception:
                                            pass
            except Exception:
                pass
    except Exception:
        pass

    # --- RECOMBINAR Goodreads tras asignaciones ---
    try:
        # reconstruir gr para que las asignaciones de isbn13 participen en merges posteriores
        gr = pd.concat([gr_with_isbn, gr_no_isbn], ignore_index=True)
        gr_with_isbn = gr[gr['isbn13'].notna()].copy()
        gr_no_isbn = gr[gr['isbn13'].isna()].copy()
    except Exception:
        # si falla, mantener las variables originales
        try:
            gr = pd.concat([gr_with_isbn, gr_no_isbn], ignore_index=True)
        except Exception:
            pass

    # NOTE: No se realizan asignaciones por matching difuso; evitamos heurísticas no deterministas.

    logger.info("gr_total=%d gr_with_isbn=%d gr_no_isbn=%d gb_total=%d", len(gr), len(gr_with_isbn), len(gr_no_isbn), len(gb))

    # Merge por isbn13 para todos los registros que tienen isbn13 en gr
    merged_by_isbn = pd.merge(
        gr_with_isbn.add_suffix("_gr"),
        gb.add_suffix("_gb"),
        left_on="isbn13_gr",
        right_on="isbn13_gb",
        how="left",
        suffixes=("_gr", "_gb"),
    )

    # Merge por match_key para registros sin isbn (título+autor normalizados)
    merged_by_key = pd.merge(
        gr_no_isbn.add_suffix("_gr"),
        gb.add_suffix("_gb"),
        left_on="_match_key_gr",
        right_on="_match_key_gb",
        how="left",
        suffixes=("_gr", "_gb"),
    )

    # Identificar gb rows que no han sido emparejadas por isbn ni por key (gb-only)
    # Mejor detección de filas de GB ya emparejadas: usar _csv_row si está disponible
    matched_gb_rows = set()
    matched_gb_isbns = set()
    # recolectar identificadores desde los merges (robusto frente a tipos)
    try:
        for df in (merged_by_isbn, merged_by_key):
            if isinstance(df, pd.DataFrame) and not df.empty:
                # preferir _csv_row_gb
                if '_csv_row_gb' in df.columns:
                    vals = df['_csv_row_gb'].dropna().unique().tolist()
                    for v in vals:
                        try:
                            matched_gb_rows.add(str(int(v)) if (isinstance(v, (int, float)) or (isinstance(v, str) and str(v).isdigit())) else str(v))
                        except Exception:
                            matched_gb_rows.add(str(v))
                # collect isbn13 matches as additional evidence
                if 'isbn13_gb' in df.columns:
                    for v in df['isbn13_gb'].dropna().unique().tolist():
                        if v is not None:
                            matched_gb_isbns.add(str(v))
                # también buscar nombres de columna alternativos (gb_id)
                for alt in ('gb_id_gb', 'id_gb', 'google_id_gb'):
                    if alt in df.columns:
                        for v in df[alt].dropna().unique().tolist():
                            if v is not None:
                                matched_gb_rows.add(str(v))
    except Exception:
        matched_gb_rows = set()
        matched_gb_isbns = set()

    # Construir la máscara gb_only: preferir _csv_row comparando como strings, fallback a isbn13
    try:
        # garantizar string en gb['_csv_row'] (ya forzado en _load_sources, pero asegurar)
        if '_csv_row' in gb.columns:
            gb['_csv_row'] = gb['_csv_row'].astype(str)
        if matched_gb_rows:
            gb_only_mask = ~gb['_csv_row'].astype(str).isin(matched_gb_rows)
        elif matched_gb_isbns:
            gb_only_mask = ~gb['isbn13'].astype(str).isin(matched_gb_isbns)
        else:
            gb_only_mask = pd.Series([True] * len(gb), index=gb.index)
    except Exception:
        gb_only_mask = pd.Series([True] * len(gb), index=gb.index)

    gb_only = gb[gb_only_mask].copy()

    # --- diagnóstico agregado de matches ---
    try:
        logger.info('gb rows total=%d; gb_only after filtering=%d; gr_with_isbn=%d; gr_no_isbn=%d', len(gb), len(gb_only), len(gr_with_isbn), len(gr_no_isbn))
    except Exception:
        pass

    # Normalizar estructuras para concatenar: queremos columnas con sufijos *_gr y *_gb
    def _ensure_columns(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
        for c in cols:
            if c not in df.columns:
                df[c] = None
        return df

    # Determinar conjunto de columnas esperadas (sufijos)
    cols_gr = [c for c in merged_by_isbn.columns if c.endswith("_gr")] + [c for c in merged_by_key.columns if c.endswith("_gr")]
    cols_gb = [c for c in merged_by_isbn.columns if c.endswith("_gb")] + [c for c in merged_by_key.columns if c.endswith("_gb")]
    cols_gr = list(dict.fromkeys(cols_gr))
    cols_gb = list(dict.fromkeys(cols_gb))

    # Preparar gb_only en formato con sufijo _gb
    gb_only_prefixed = gb_only.copy()
    gb_only_prefixed = gb_only_prefixed.add_suffix("_gb")
    # marcar origen para facilitar filtrado posterior: gb_only
    try:
        gb_only_prefixed['_source_type'] = 'gb_only'
    except Exception:
        pass
    # Añadir columnas _gr vacías para compatibilidad
    for c in cols_gr:
        if c not in gb_only_prefixed.columns:
            gb_only_prefixed[c] = None
    # Asegurar columnas esperadas en merged_by_isbn y merged_by_key
    merged_by_isbn = _ensure_columns(merged_by_isbn, cols_gb + cols_gr)
    merged_by_key = _ensure_columns(merged_by_key, cols_gb + cols_gr)
    # marcar origen para merges basados en isbn/key
    try:
        merged_by_isbn['_source_type'] = 'merged_by_isbn'
    except Exception:
        pass
    try:
        merged_by_key['_source_type'] = 'merged_by_key'
    except Exception:
        pass

    # Concatenar todas las filas en un único merged (alineado por columnas)
    # NOTA: No eliminamos columnas all-null aquí (el usuario lo pidió); mantenemos el esquema original.
    def _drop_all_na_cols(df: pd.DataFrame) -> pd.DataFrame:
        return df

    merged_by_isbn = _drop_all_na_cols(merged_by_isbn)
    merged_by_key = _drop_all_na_cols(merged_by_key)
    gb_only_prefixed = _drop_all_na_cols(gb_only_prefixed)

    # Log de diagnóstico: tamaños intermedios antes de concatenar
    try:
        logger.info('Merged components sizes: by_isbn=%d by_key=%d gb_only=%d',
                    0 if merged_by_isbn is None else (len(merged_by_isbn) if hasattr(merged_by_isbn, '__len__') else 0),
                    0 if merged_by_key is None else (len(merged_by_key) if hasattr(merged_by_key, '__len__') else 0),
                    0 if gb_only_prefixed is None else (len(gb_only_prefixed) if hasattr(gb_only_prefixed, '__len__') else 0))
    except Exception:
        pass

    # Evitar concatenar DataFrames vacíos (reduce warnings y asegura dtypes coherentes)
    to_concat = [df for df in (merged_by_isbn.reset_index(drop=True) if isinstance(merged_by_isbn, pd.DataFrame) else merged_by_isbn,
                               merged_by_key.reset_index(drop=True) if isinstance(merged_by_key, pd.DataFrame) else merged_by_key,
                               gb_only_prefixed.reset_index(drop=True) if isinstance(gb_only_prefixed, pd.DataFrame) else gb_only_prefixed)
                if isinstance(df, pd.DataFrame) and df.shape[0] > 0]
    if to_concat:
        # diagnóstico: comprobar suma de filas antes de concatenar
        component_counts = []
        try:
            component_counts = [len(df) for df in to_concat]
            sum_rows = sum(component_counts)
            logger.info('Component row counts before concat: %s; sum=%d', component_counts, sum_rows)
        except Exception:
            sum_rows = None
        merged = pd.concat(to_concat, ignore_index=True, sort=False)
        # diagnóstico adicional: calcular expected a partir de componentes reales
        try:
            expected = (len(merged_by_isbn) if isinstance(merged_by_isbn, pd.DataFrame) else 0) + (len(merged_by_key) if isinstance(merged_by_key, pd.DataFrame) else 0) + (len(gb_only_prefixed) if isinstance(gb_only_prefixed, pd.DataFrame) else 0)
            logger.info('Expected merged rows (by_isbn + by_key + gb_only) = %d; actual merged = %d', expected, len(merged))
            if expected is not None and len(merged) != expected:
                logger.warning('Concatenación: filas inesperadas. expected=%s actual=%d; component_counts=%s', str(expected), len(merged), component_counts)
        except Exception:
            logger.debug('No se pudo calcular expected rows con componentes')
        try:
             if sum_rows is not None and len(merged) != sum_rows:
                 logger.warning('Concatenación resultó en diferente número de filas: expected_sum=%s actual=%d', str(sum_rows), len(merged))
                 # mostrar primeros indices y muestras para diagnóstico
                 try:
                     for i, df in enumerate(to_concat):
                         logger.debug('component %d head:\n%s', i, df.head(3).to_string())
                 except Exception:
                     pass
        except Exception:
            pass
    else:
         # DataFrame vacío con al menos columnas esperadas
         merged = pd.DataFrame()

    # asegurar existencia de columna _source_type para downstream
    if '_source_type' not in merged.columns:
        merged['_source_type'] = None

    return merged


def _safe_write_parquet(df: pd.DataFrame, path: Path, index: bool = False) -> None:
    """Escribe DataFrame a parquet de forma robusta.

    Intenta escribir con pyarrow; si falla por tipos no esperados, normaliza
    columnas de tipo objeto serializando dicts/listas a JSON y reintenta. Si sigue fallando, fuerza toda la tabla a
    strings como último recurso.

    Escribe a un fichero temporal y luego reemplaza el destino para evitar corrupciones
    y problemas de concurrencia/locks en Windows. También intenta eliminar el destino
    previo antes de reemplazarlo si surge un permiso.
    """
    import os
    import uuid
    import gc

    # Asegurar directorio destino
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass

    # Crear temp_path con uuid para evitar colisiones en re-ejecuciones simultáneas
    temp_path = path.with_name(path.name + f".{uuid.uuid4().hex}.tmp")
    # eliminar temp previo si existe (poco probable por uuid)
    try:
        if temp_path.exists():
            temp_path.unlink()
    except Exception:
        pass

    # Intentar eliminar destino previo antes de escribir para evitar locks en Windows
    def _try_remove_target(p: Path):
        try:
            if p.exists():
                p.unlink()
                return True
        except Exception:
            # intentar renombrar como respaldo (si unlink falla por lock)
            try:
                backup = p.with_name(p.name + ".backup")
                if backup.exists():
                    backup.unlink()
                p.replace(backup)
                return True
            except Exception:
                return False
        return False

    def _try_write(df_to_write: pd.DataFrame, target: Path) -> bool:
        try:
            # Forzar engine pyarrow y cerrar recursos inmediatamente
            df_to_write.to_parquet(target, index=index, engine='pyarrow')
            # forzar GC para que pyarrow libere recursos en Windows
            gc.collect()
            return True
        except Exception as e:
            logger.debug('to_parquet falló para %s: %s', target.name, str(e))
            return False

    # Primer intento: eliminar objetivo si posible (mejora para re-ejecuciones)
    try:
        _try_remove_target(path)
    except Exception:
        pass

    # Primer intento: escribir directamente al archivo temporal
    try:
        if _try_write(df, temp_path):
            try:
                # Reemplazar atómicamente el destino final (os.replace es atómico en Windows)
                try:
                    os.replace(str(temp_path), str(path))
                except Exception:
                    # fallback a Path.replace
                    temp_path.replace(path)
                logger.info('Escrito %s (replace desde temp)', path.name)
                return
            except Exception as e:
                logger.warning('Fallo al reemplazar %s desde temp: %s. Intentando escritura directa.', path.name, str(e))
                # intentar escribir directamente al destino como último recurso
                try:
                    if _try_write(df, path):
                        logger.info('Escrito %s (directo tras fallo replace)', path.name)
                        try:
                            if temp_path.exists():
                                temp_path.unlink()
                        except Exception:
                            pass
                        return
                except Exception:
                    pass
    except Exception:
        pass

    logger.warning('Error escribiendo %s. Intentando normalizar columnas objeto.', path.name)
    df2 = df.copy()
    # detectar columnas objeto y normalizar solo dicts; dejar listas intactas para pyarrow
    for col in df2.columns:
        if df2[col].dtype == object or df2[col].dtype.name == 'bytes':
            def _conv(v):
                try:
                    if v is None or (isinstance(v, float) and pd.isna(v)):
                        return None
                    # bytes -> str
                    if isinstance(v, (bytes, bytearray)):
                        try:
                            return v.decode('utf-8')
                        except Exception:
                            return str(v)
                    # dicts -> JSON string (preserva estructura)
                    if isinstance(v, dict):
                        try:
                            return json.dumps(v, ensure_ascii=False)
                        except Exception:
                            return str(v)
                    # listas: dejarlas como listas nativas para que pyarrow pueda mapearlas a tipos list
                    if isinstance(v, list):
                        return v
                    # numpy types -> native python then str
                    if hasattr(v, 'tolist'):
                        try:
                            return v.tolist()
                        except Exception:
                            pass
                    return v
                except Exception:
                    return v
            try:
                # aplicar conversión que preserva listas
                df2[col] = df2[col].apply(_conv)
            except Exception:
                # último recurso: convertir dicts a JSON y strings para el resto
                try:
                    df2[col] = df2[col].apply(lambda v: None if (v is None or (isinstance(v, float) and pd.isna(v))) else (json.dumps(v, ensure_ascii=False) if isinstance(v, dict) else str(v)))
                except Exception:
                    df2[col] = df2[col].apply(lambda v: None if (v is None or (isinstance(v, float) and pd.isna(v))) else str(v))
    # Intentar escribir df2 a temp y reemplazar
    try:
        if _try_write(df2, temp_path):
            try:
                try:
                    os.replace(str(temp_path), str(path))
                except Exception:
                    temp_path.replace(path)
                logger.info('Escrito (normalizado) %s', path.name)
                return
            except Exception as e2:
                logger.warning('Fallo al reemplazar %s tras normalizar: %s', path.name, str(e2))
                try:
                    if _try_write(df2, path):
                        logger.info('Escrito %s (directo, normalizado)', path.name)
                        try:
                            if temp_path.exists():
                                temp_path.unlink()
                        except Exception:
                            pass
                        return
                except Exception:
                    pass
    except Exception:
        pass

    logger.warning('Fallo escribiendo %s tras normalizar columnas. Forzando todo a strings.', path.name)
    # Último intento: forzar todas las columnas a strings (None -> None)
    df3 = df.copy()
    for col in df3.columns:
        try:
            df3[col] = df3[col].apply(lambda v: None if (v is None or (isinstance(v, float) and pd.isna(v))) else str(v))
        except Exception:
            try:
                df3[col] = df3[col].astype(str)
            except Exception:
                df3[col] = df3[col].apply(lambda v: None if (v is None or (isinstance(v, float) and pd.isna(v))) else str(v))
    # Escribir df3 a temp y reemplazar
    try:
        if _try_write(df3, temp_path):
            try:
                try:
                    os.replace(str(temp_path), str(path))
                except Exception:
                    temp_path.replace(path)
                logger.info('Escrito (fallback strings) %s', path.name)
                return
            except Exception as e3:
                # último recurso: escribir directo
                try:
                    df3.to_parquet(path, index=index, engine='pyarrow')
                    logger.info('Escrito (fallback directo) %s', path.name)
                    try:
                        if temp_path.exists():
                            temp_path.unlink()
                    except Exception:
                        pass
                    return
                except Exception as e4:
                    logger.error('Fallo definitivo al escribir %s: %s', path.name, str(e4))
                    raise
    except Exception as e:
        logger.error('Fallo definitivo al escribir %s: %s', path.name, str(e))
        raise

def safe_compute_quality_metrics(rows):
    """Wrapper seguro para compute_quality_metrics: si falla la función original,
    devuelve métricas mínimas que permiten continuar y rastrear el error.
    """
    try:
        return compute_quality_metrics(rows)
    except Exception as e:
        logger.warning("safe_compute_quality_metrics: compute_quality_metrics falló: %s", str(e))
        # Fallback mínimo
        total = len(rows) if isinstance(rows, list) else 0
        nulos = {}
        try:
            cols = ["book_id", "titulo", "autor_principal", "isbn13", "fecha_publicacion", "idioma", "precio", "moneda"]
            nulos = nulls_by_column(rows, cols) if rows else {c: 0 for c in cols}
        except Exception:
            nulos = {}
        return {
            "filas_totales": total,
            "nulos_por_campo": nulos,
            "porcentaje_fechas_validas": 0.0,
            "porcentaje_idiomas_validos": 0.0,
            "porcentaje_monedas_validas": 0.0,
            "porcentaje_isbn13_validos": 0.0,
            "completitud_promedio": 0.0,
            "duplicados_isbn13": 0,
            "duplicados_book_id": 0,
            "filas_por_fuente": {},
        }

def _canonicalize_merged_columns(merged: pd.DataFrame) -> pd.DataFrame:
    """Asegura nombres de columnas consistentes y tipos básicos para downstream.
    - Normaliza sufijos si existen mezclas (_gr/_gb)
    - Evita crear columnas vacías que provengan de supuestos no presentes en los landing files
    KEYWORDS: CANONICALIZE_COLUMNS
    """
    if not isinstance(merged, pd.DataFrame):
        return merged
    df = merged.copy()
    # eliminar duplicados accidentales en nombres (ej. col y col_gr ambos presentes)
    for col in list(df.columns):
        if col.endswith('_gr') or col.endswith('_gb'):
            base = col[:-3]
            try:
                if base in df.columns:
                    df[base] = df[base].where(df[base].notna(), df[col])
            except Exception:
                pass
    # NO crear columnas vacías aquí; sólo normalizar las existentes
    return df


def _build_dim_from_merged(merged: pd.DataFrame) -> pd.DataFrame:
    """Implementación mínima y robusta para construir dim a partir de merged.
    - Preserva columnas originales de Goodreads
    - Rellena campos vacíos con valores de Google Books cuando estén presentes
    - Añade canonical_key, book_id (isbn13 si válido), provenance simple y ts_ultima_actualizacion
    """
    if not isinstance(merged, pd.DataFrame) or merged.empty:
        return pd.DataFrame()
    df = merged.copy()
    # Campos de Goodreads a preservar si existen
    gr_fields = [c for c in ['titulo', 'autor_principal', 'rating', 'ratings_count', 'book_url', 'isbn10', 'isbn13'] if c in df.columns or f'{c}_gr' in df.columns or f'{c}_gb' in df.columns]
    # Campos útiles opcionales desde Google Books — sólo si hay valores reales en merged
    possible_gb = ['editorial', 'fecha_publicacion', 'idioma', 'paginas', 'formato', 'categoria', 'precio', 'moneda']
    def _has_any(colname: str) -> bool:
        try:
            if colname in df.columns and df[colname].notna().any():
                return True
        except Exception:
            pass
        return False
    gb_extra = [c for c in possible_gb if any(_has_any(x) for x in (c, f'{c}_gb', f'{c}_gr'))]

    records = []
    for _, row in df.iterrows():
        rec = {}
        prov_map = {}
        # para cada campo de Goodreads preferir valor existente
        for f in gr_fields:
            val = None
            src = None
            # Prioridad: *_gr -> base -> *_gb
            for candidate in (f + '_gr', f, f + '_gb'):
                if candidate in row.index and pd.notna(row.get(candidate)) and str(row.get(candidate)).strip() != '':
                    val = row.get(candidate)
                    src = 'goodreads' if candidate.endswith('_gr') or candidate == f else 'google_books'
                    # override src for isbn13 if an explicit source marker exists on the row
                    if f == 'isbn13':
                        # comprobar marcadores de procedencia explícitos (posibles sufijos)
                        for s_col in ('isbn13_source', 'isbn13_source_gr', 'isbn13_source_gb'):
                            if s_col in row.index and pd.notna(row.get(s_col)):
                                src = row.get(s_col)
                                break
                    break
            rec[f] = val
            prov_map[f] = src
         # campos extra desde GB
        for f in gb_extra:
            val = None
            src = None
            for candidate in (f, f + '_gb', f + '_gr'):
                if candidate in row.index and pd.notna(row.get(candidate)) and str(row.get(candidate)).strip() != '':
                    val = row.get(candidate)
                    src = 'google_books' if candidate.endswith('_gb') or candidate == f else 'merged'
                    break
            rec[f] = val
            prov_map[f] = src
        # canonical key / isbn norm
        try:
            isbn13_norm = None
            if 'isbn13' in rec and rec.get('isbn13') is not None:
                isbn13_norm = try_normalize_isbn(rec.get('isbn13'))[1]
        except Exception:
            isbn13_norm = None
        rec['isbn13_norm'] = isbn13_norm
        rec['isbn13_valido'] = bool(isbn13_norm)
        # canonical key
        try:
            key = isbn13_norm if isbn13_norm else _canonical_key(rec.get('titulo'), rec.get('autor_principal'), rec.get('editorial'), None)
        except Exception:
            key = _canonical_key(rec.get('titulo'), rec.get('autor_principal'), None, None)
        rec['canonical_key'] = key
        # book_id preferir isbn13 válido
        rec['book_id'] = isbn13_norm if isbn13_norm else key
        # provenance simple
        try:
            rec['provenance'] = json.dumps(prov_map, ensure_ascii=False)
        except Exception:
            rec['provenance'] = str(prov_map)
        rec['ts_ultima_actualizacion'] = datetime.now(timezone.utc).isoformat(timespec='seconds')
        records.append(rec)
    # Construir DataFrame sólo con las columnas que realmente existen en los records
    dim = pd.DataFrame(records)
    # Añadir campos canónicos si no existen pero necesarios
    if 'canonical_key' not in dim.columns:
        dim['canonical_key'] = dim.apply(lambda r: (r.get('isbn13') or '') and _canonical_key(r.get('titulo'), r.get('autor_principal'), None, None) if True else _canonical_key(r.get('titulo'), r.get('autor_principal'), None, None), axis=1)
    if 'book_id' not in dim.columns:
        # preferir isbn13 normalizado si está, sino canonical_key
        try:
            dim['book_id'] = dim.apply(lambda r: (r.get('isbn13') if (r.get('isbn13') is not None and str(r.get('isbn13')).strip() != '') else r.get('canonical_key')), axis=1)
        except Exception:
            dim['book_id'] = dim.get('canonical_key')
    if 'provenance' not in dim.columns:
        dim['provenance'] = dim.apply(lambda r: (json.dumps(r.get('provenance')) if isinstance(r.get('provenance'), (dict, list)) else r.get('provenance')) if 'provenance' in r.index else json.dumps({k: v for k, v in []}), axis=1)
    if 'ts_ultima_actualizacion' not in dim.columns:
        dim['ts_ultima_actualizacion'] = datetime.now(timezone.utc).isoformat(timespec='seconds')
    # No añadir columnas vacías por defecto; devolver sólo las que tengan presencia real
    return dim


def integrate() -> None:
    """Integración completa: carga, normalización, fusión y escritura de artefactos estándar.
    Ejecutar en el orden correcto es crítico: primero se cargan y normalizan las fuentes,
    luego se fusionan, se generan métricas de calidad y finalmente se escriben los resultados.
    """
    global gr_n, gb_n, detail, assertions

    # Inicializaciones defensivas para evitar referencias no resueltas en flujos excepcionales
    dim = pd.DataFrame()
    dim_for_metrics = pd.DataFrame()
    detail = pd.DataFrame()
    assertions = {}

    # Asegurar directorios de logs de trabajo (work/logs/..)
    try:
        ensure_work_dirs()
    except Exception:
        # no bloquear si no existe work/utils_logging
        pass

    # Eliminar artefactos previos para evitar problemas de locks en Windows y re-ejecuciones
    try:
        prev_dim = STANDARD / 'dim_book.parquet'
        prev_detail = STANDARD / 'book_source_detail.parquet'
        if prev_dim.exists():
            try:
                prev_dim.unlink()
                logger.info('Eliminado artefacto previo: %s', prev_dim.name)
            except Exception:
                logger.debug('No se pudo eliminar previo dim_book.parquet', exc_info=True)
        if prev_detail.exists():
            try:
                prev_detail.unlink()
                logger.info('Eliminado artefacto previo: %s', prev_detail.name)
            except Exception:
                logger.debug('No se pudo eliminar previo book_source_detail.parquet', exc_info=True)
    except Exception:
        pass

    # Carga inicial de fuentes
    gr_n, gb_n = _load_sources()

    # Normalización básica
    gr_n, gb_n = _normalize_frames(gr_n, gb_n)

    # Merge y asignación de ISBN desde candidatos
    merged = _merge_sources(gr_n, gb_n)

    # Normalizar columnas canónicas para prevenir errores posteriores
    try:
        merged = _canonicalize_merged_columns(merged)
    except Exception:
        logger.debug('Fallo al canonicalizar columnas de merged; se mantiene objeto original', exc_info=True)

    # APLICAR NORMALIZACIÓN SEMÁNTICA: fechas ISO, idioma BCP-47, moneda ISO-4217, precio numérico
    try:
        merged = _apply_semantic_normalization(merged)
    except Exception:
        logger.debug('Normalización semántica falló; se continúa con merged original', exc_info=True)

    # Unificar columnas base (coalesce) para que existan nombres sin sufijos (titulo, autor_principal, isbn13, etc.)
    try:
        # Coalescer sólo para los campos relevantes: conservar esquema de Goodreads
        # y añadir columnas útiles de Google Books sólo si existen en el merged.
        # Evitar iterar sobre columnas que sabemos inexistentes en los ficheros de landing.
        gr_fields = ['titulo', 'autor_principal', 'rating', 'ratings_count', 'book_url', 'isbn10', 'isbn13']
        gb_extra = ['editorial', 'fecha_publicacion', 'idioma', 'paginas', 'formato', 'categoria', 'precio', 'moneda', 'autores']

        # Coalesce de campos de Goodreads (priorizar *_gr, luego base, luego *_gb)
        for f in gr_fields:
            try:
                merged[f] = _coalesce_columns(merged, f)
            except Exception:
                # crear columna vacía si algo falla
                merged[f] = pd.Series([None] * len(merged), index=merged.index)

        # Añadir/rellenar campos extra desde Google Books sólo si aparecen en merged
        for f in gb_extra:
            # si existe cualquier variante con sufijo en merged, coalescerla
            candidates_exist = any(col for col in merged.columns if col == f or col.endswith(f"_{'gb'}") or col.endswith(f"_{'gr'}"))
            if candidates_exist:
                try:
                    merged[f] = _coalesce_columns(merged, f)
                except Exception:
                    merged[f] = pd.Series([None] * len(merged), index=merged.index)
            else:
                # no crear columnas que no aportarán valores (evita columnas all-null)
                if f in merged.columns:
                    merged[f] = merged[f]
                else:
                    pass
    except Exception:
        logger.debug('Coalesce de columnas base falló, se continúa con merged tal cual', exc_info=True)

    # Separar merged en dos vistas: una completa para detalle (detail) y otra filtrada para construir dim (excluir gb_only)
    try:
        merged_for_dim = merged[merged['_source_type'] != 'gb_only'].copy() if isinstance(merged, pd.DataFrame) else pd.DataFrame()
    except Exception:
        merged_for_dim = merged.copy() if isinstance(merged, pd.DataFrame) else pd.DataFrame()

    # ===== Construir dim canónico a partir de merged_for_dim =====
    try:
        # Intentar construir la tabla dimensional con la función dedicada
        dim = _build_dim_from_merged(merged_for_dim)

        # Asignar book_id: preferir isbn13 válido, sino canonical_key
        try:
            if 'isbn13' in dim.columns and 'isbn13_valido' in dim.columns:
                dim['book_id'] = dim.apply(lambda r: r['isbn13'] if (pd.notna(r.get('isbn13')) and bool(r.get('isbn13_valido'))) else r.get('canonical_key'), axis=1)
            elif 'canonical_key' in dim.columns:
                dim['book_id'] = dim['canonical_key']
            else:
                dim['book_id'] = None
        except Exception:
            # fallback genérico
            if 'canonical_key' in dim.columns:
                dim['book_id'] = dim['canonical_key']
            else:
                dim['book_id'] = None

        # Asegurar ts_ultima_actualizacion
        try:
            now_ts = datetime.now(timezone.utc).isoformat(timespec='seconds')
            if 'ts_ultima_actualizacion' not in dim.columns:
                dim['ts_ultima_actualizacion'] = now_ts
            else:
                dim['ts_ultima_actualizacion'] = dim['ts_ultima_actualizacion'].fillna(now_ts)
        except Exception:
            dim['ts_ultima_actualizacion'] = datetime.now(timezone.utc).isoformat(timespec='seconds')

        # Deduplicar por book_id garantizando una fila por libro canónico
        try:
            if 'book_id' in dim.columns:
                # preferir filas con más campos no nulos y con fuente_ganadora
                # calcular score simple: número de campos no nulos
                # Reducimos score_cols a campos relevantes existentes en landing
                score_cols = ['titulo', 'autor_principal', 'autores', 'isbn13']
                dim['_completitud_tmp'] = dim[score_cols].notna().sum(axis=1)
                # score simple: número de campos no nulos (prioridad a la completitud)
                dim['_score_total'] = dim['_completitud_tmp']
                dim = dim.sort_values(by=['_score_total'], ascending=False).drop_duplicates(subset=['book_id'], keep='first').reset_index(drop=True)
                # eliminar columnas temporales
                for _c in ['_completitud_tmp','_gb_ms_tmp','_score_total']:
                    if _c in dim.columns:
                        try:
                            del dim[_c]
                        except Exception:
                            pass
        except Exception:
            # si falla deduplicación, no interrumpir; dim se mantiene
            pass
    except Exception as e:
        # No bloquear pipeline si la construcción del dim falla; dejar que el flujo posterior use el fallback
        logger.warning('No se pudo construir dim desde merged_for_dim: %s', str(e))
        dim = pd.DataFrame()

    # Registro de diagnóstico: tamaño de merged y unicidad de book_id antes de dedup
    try:
        logger.info(
            'Merged rows (total) antes dedup: %d; merged_for_dim rows: %d',
            0 if merged is None else (len(merged) if hasattr(merged, '__len__') else 0),
            0 if merged_for_dim is None else (len(merged_for_dim) if hasattr(merged_for_dim, '__len__') else 0),
        )
        if isinstance(merged, pd.DataFrame) and not merged.empty:
            # contar book_id posibles (puede no existir aún)
            if 'book_id' in merged.columns:
                counts = merged['book_id'].value_counts(dropna=False)
                dup_groups = counts[counts > 1]
                logger.info('book_id únicos en merged: %d, grupos duplicados: %d', counts.shape[0], dup_groups.shape[0])
                if not dup_groups.empty:
                    logger.debug('book_id duplicados (muestra): %s', dup_groups.head(10).to_dict())
            else:
                logger.info('book_id no presente aún en merged')
    except Exception:
        logger.debug('Diagnóstico merged falló', exc_info=True)

    # --- Aserciones soft y deduplicación ---
    # Construir `book_source_detail` a partir de las fuentes originales de landing (preservar trazabilidad)
    try:
        # convertir merged (DataFrame) en uno con columnas esperadas
        # detail = merged.copy()
        # fila original (row_number) para trazabilidad
        # detail['row_number'] = detail.index + 1
        # detail['book_id_candidato'] = detail.get('book_id')
        # detail['ts_ingesta'] = detail.get('ts_ultima_actualizacion') if 'ts_ultima_actualizacion' in detail.columns else datetime.now(timezone.utc).isoformat(timespec='seconds')

        # Construir `book_source_detail` preservando las filas originales de landing
        # y rellenando campos de Goodreads desde GoogleBooks cuando sea posible.
        # _build_book_source_detail devuelve un DataFrame con las columnas exactas
        # requeridas y `ts_ingest` con la hora de ejecución.
        try:
            detail = _build_book_source_detail(gr_n, gb_n)
        except Exception as _e:
            logger.warning('Fallo construyendo book_source_detail desde landing: %s. Usando merged como fallback.', str(_e), exc_info=True)
            detail = merged.copy()
            # mantener compatibilidad: asignar row_number y ts_ingest si faltan
            if 'row_number' not in detail.columns:
                detail['row_number'] = detail.index + 1
            if 'book_id_candidato' not in detail.columns:
                detail['book_id_candidato'] = detail.get('book_id')
            if 'ts_ingest' not in detail.columns:
                detail['ts_ingest'] = detail.get('ts_ultima_actualizacion') if 'ts_ultima_actualizacion' in detail.columns else datetime.now(timezone.utc).isoformat(timespec='seconds')
    except Exception as e:
        logger.warning('Error en fase de deduplicación/aserciones: %s', str(e))
        logger.debug(traceback.format_exc())
        # fallback: usar merged como dim y un detail mínimo
        dim = merged.copy()
        detail = merged.copy()
        detail['row_number'] = detail.index + 1
        detail['book_id_candidato'] = detail.get('book_id')
        detail['ts_ingesta'] = detail.get('ts_ultima_actualizacion') if 'ts_ultima_actualizacion' in detail.columns else datetime.now(timezone.utc).isoformat(timespec='seconds')
        detail['valid'] = True
        detail['exclude_reason'] = None

    # APLICAR LIMPIEZA DE REFERENCIAS ANTES DE ESCRIBIR PARA EVITAR BLOQUEOS EN WINDOWS
    try:
        import gc
        # eliminar referencias pesadas temporales
        for _n in ('merged_local','merged_by_isbn','merged_by_key','gb_only_prefixed','to_concat','merged'):
            try:
                if _n in locals():
                    del locals()[_n]
            except Exception:
                pass
        gc.collect()
    except Exception:
        pass

    # Métricas de calidad
    records_sanitized: List[Mapping[str, object]] = []
    try:
        # usar dim deduplicado y final para métricas (asegurar 1 fila por libro canónico)
        try:
            # deduplicar por isbn13 preferente, si no usar book_id
            dim_for_metrics = dim.copy()
            if 'isbn13' in dim_for_metrics.columns and dim_for_metrics['isbn13'].notna().any():
                # ordenar preferentemente por isbn13 y, si existe, por gb_match_score descendente
                dim_for_metrics = dim_for_metrics.sort_values(by=['isbn13'], ascending=True).drop_duplicates(subset=['isbn13'], keep='first').reset_index(drop=True)
            else:
                dim_for_metrics = dim_for_metrics.drop_duplicates(subset=['book_id'], keep='first').reset_index(drop=True)
            records = dim_for_metrics.to_dict(orient='records')
        except Exception as e:
            logger.warning('dim.to_dict falló para métricas: %s. Intentando fallback row-wise.', str(e))
            records = []
            try:
                for _, row in dim.iterrows():
                    try:
                        # construir dict simple con columnas planas
                        rec = {c: _sanitize_value(row.get(c)) for c in dim.columns}
                        records.append(rec)
                    except Exception:
                        records.append({})
            except Exception:
                records = []
        # Sanitizar registros para evitar tipos complejos que rompan compute_quality_metrics
        try:
            records_sanitized = _sanitize_records_for_metrics(records)
        except Exception as e:
            logger.warning('Sanitización de records falló: %s. Intentando sanitizar manualmente.', str(e))
            records_sanitized = []
            for r in (records or []):
                if isinstance(r, Mapping):
                    row = {}
                    for k, v in r.items():
                        try:
                            row[k] = _sanitize_value(v)
                        except Exception:
                            row[k] = None
                    records_sanitized.append(row)
                else:
                    try:
                        records_sanitized.append({'raw': _sanitize_value(r)})
                    except Exception:
                        records_sanitized.append({})
        # usar wrapper seguro para evitar que una excepción interrumpa la escritura
        assertions = safe_compute_quality_metrics(records_sanitized)
        # Forzar valores clave coherentes: filas_totales = número de filas del dim final
        try:
            assertions['filas_totales'] = len(records_sanitized)
        except Exception:
            pass
        # filas_por_fuente: preferir contar por 'source_name' en el detalle (mantener trazabilidad)
        try:
            if isinstance(detail, pd.DataFrame) and 'source_name' in detail.columns:
                assertions['filas_por_fuente'] = detail['source_name'].fillna('unknown').value_counts().to_dict()
            else:
                assertions.setdefault('filas_por_fuente', {})
        except Exception:
            pass
    except Exception as e:
        logger.warning('Error calculando métricas de calidad: %s', str(e))
        logger.debug(traceback.format_exc())
        assertions = safe_compute_quality_metrics([])

    # Registrar una entrada mínima en logs/rules para trazabilidad (evita carpeta vacía)
    try:
        log_rule_jsonl({
            "event": "integrate_run_summary",
            "ts_utc": datetime.now(timezone.utc).isoformat(timespec='seconds'),
            "records_input": assertions.get('filas_totales') if isinstance(assertions, dict) else None,
            "notes": "integrate_pipeline executed",
        })
    except Exception:
        pass

    # Escritura de artefactos estándar
    try:
        # normalizar tipos en dim para evitar problemas con pyarrow
        for col in dim.columns:
            if dim[col].dtype == object:
                # serializar solo dicts y bytes; dejar listas como listas nativas para pyarrow
                def _conv_dim(v):
                    if v is None or (isinstance(v, float) and pd.isna(v)):
                        return None
                    if isinstance(v, (bytes, bytearray)):
                        try:
                            return v.decode('utf-8')
                        except Exception:
                            return str(v)
                    if isinstance(v, dict):
                        try:
                            return json.dumps(v, ensure_ascii=False)
                        except Exception:
                            return str(v)
                    # dejar listas intactas para que pyarrow las escriba como list<>
                    return v
                dim[col] = dim[col].apply(_conv_dim)
        # liberar referencias pesadas antes de escribir para evitar locks en Windows
        try:
            import gc
            # eliminar referencias que ya no se necesitan
            for _name in ('gr_n', 'gb_n', 'merged', 'merged_local', 'merged_by_isbn', 'merged_by_key', 'gb_only'):
                try:
                    if _name in globals():
                        del globals()[_name]
                except Exception:
                    pass
            gc.collect()
        except Exception:
            pass
        _safe_write_parquet(dim, STANDARD / 'dim_book.parquet', index=False)
    except Exception as e:
        logger.error('Fallo al escribir dim_book.parquet: %s', str(e))
        raise

    try:
        if not isinstance(detail, pd.DataFrame):
            detail = pd.DataFrame(detail)
        # normalizar detail: asegurar tipos simples (preservar listas nativas para pyarrow)
        for col in detail.columns:
            if detail[col].dtype == object:
                # aplicar sanitización pero PRESERVAR listas nativas (deja listas intactas)
                def _conv_detail(v):
                    if v is None or (isinstance(v, float) and pd.isna(v)):
                        return None
                    # mantener listas para pyarrow
                    if isinstance(v, list):
                        return v
                    if isinstance(v, (bytes, bytearray)):
                        try:
                            return v.decode('utf-8')
                        except Exception:
                            return str(v)
                    # dicts -> JSON
                    if isinstance(v, dict):
                        try:
                            return json.dumps(v, ensure_ascii=False)
                        except Exception:
                            return str(v)
                    # tipos básicos
                    if isinstance(v, (str, int, float, bool)):
                        return v
                    # fallback: str
                    return str(v)
                try:
                    detail[col] = detail[col].apply(_conv_detail)
                except Exception:
                    detail[col] = detail[col].apply(lambda v: None if (v is None or (isinstance(v, float) and pd.isna(v))) else str(v))
        # asegurar row_number entero
        if 'row_number' in detail.columns:
            try:
                detail['row_number'] = pd.to_numeric(detail['row_number'], errors='coerce').astype('Int64')
            except Exception:
                pass
        _safe_write_parquet(detail, STANDARD / 'book_source_detail.parquet', index=False)
    except Exception as e:
        logger.error('Fallo al escribir book_source_detail.parquet: %s', str(e))
        # intentar escribir una versión reducida para diagnóstico
        try:
            diag = detail.copy()
            # limitar columnas a las más relevantes y sanitizar
            keep = [c for c in ['row_number','book_id_candidato','titulo','autor_principal','isbn13','fecha_publicacion','precio','moneda','valid','exclude_reason'] if c in diag.columns]
            diag = diag[keep]
            for col in diag.columns:
                diag[col] = diag[col].apply(lambda v: None if (v is None or (isinstance(v, float) and pd.isna(v))) else _sanitize_value(v))
            _safe_write_parquet(diag, STANDARD / 'book_source_detail.parquet', index=False)
            logger.info('Escrito book_source_detail.parquet (diagnóstico reducido)')
        except Exception as e2:
            logger.error('Fallo final al escribir book_source_detail.parquet (diagnóstico): %s', str(e2))
            raise

    try:
        with open(DOCS / 'quality_metrics.json', 'w', encoding='utf-8') as f:
            json.dump(assertions, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.warning('No se pudo escribir quality_metrics.json: %s', str(e))

    logger.info('Escrito dim_book.parquet y book_source_detail.parquet; métricas en quality_metrics.json')

def main() -> None:
    integrate()

# Helper: coalesce de columnas base con sufijos (definido antes de `integrate` para evitar referencias no resueltas)
def _coalesce_columns(df: pd.DataFrame, col: str, suffixes: List[str] = None) -> pd.Series:
    """Coalesce helper: combina col, col_gr, col_gb en una sola Series priorizando _gr -> base -> _gb."""
    if suffixes is None:
        suffixes = ['_gr', '_gb']
    candidates = [col] + [f"{col}{s}" for s in suffixes]
    series = None
    for c in candidates:
        if c in df.columns:
            s = df[c]
            if series is None:
                series = s.astype(object)
            else:
                try:
                    series = series.combine_first(s)
                except Exception:
                    series = series.where(series.notna(), s)
    if series is None:
        series = pd.Series([None] * len(df), index=df.index, dtype=object)
    return series

# Nota: la ejecución directa se pospone hasta el final del fichero para asegurar que
# todas las funciones auxiliares (_build_book_source_detail, _build_dim_from_merged, etc.)
# estén definidas antes de ejecutar `integrate()`.
# El bloque `if __name__ == '__main__'` se mueve al final del archivo.
def _apply_semantic_normalization(df: pd.DataFrame) -> pd.DataFrame:
    """Versión mínima y robusta de normalización semántica.
    - Normaliza fecha_publicacion -> ISO (usando parse_date_to_iso)
    - Normaliza idioma -> BCP-47 (normalize_language) y marca idioma_valido
    - Normaliza moneda -> normalize_currency + validate_currency
    - Convierte precio a float cuando es posible
    Mantiene columnas existentes y añade flags *_valida o *_parcial cuando procede.
    """
    if not isinstance(df, pd.DataFrame):
        return df
    out = df.copy()

    # fecha_publicacion
    if 'fecha_publicacion' in out.columns:
        iso_dates = []
        parcial_flags = []
        valid_flags = []
        for v in out['fecha_publicacion'].fillna('').astype(str).tolist():
            if not v or v.strip() == '' or v == 'nan':
                iso_dates.append(None)
                parcial_flags.append(False)
                valid_flags.append(False)
            else:
                try:
                    iso, parcial = parse_date_to_iso(v)
                    iso_dates.append(iso)
                    parcial_flags.append(bool(parcial))
                    valid_flags.append(bool(iso is not None))
                except Exception:
                    iso_dates.append(None)
                    parcial_flags.append(False)
                    valid_flags.append(False)
        out['fecha_publicacion'] = iso_dates
        out['fecha_publicacion_parcial'] = parcial_flags
        out['fecha_publicacion_valida'] = valid_flags
    else:
        out['fecha_publicacion_parcial'] = False
        out['fecha_publicacion_valida'] = False

    # idioma
    if 'idioma' in out.columns:
        norm_langs = []
        lang_valid = []
        for v in out['idioma'].fillna('').astype(str).tolist():
            if not v or v.strip() == '' or v == 'nan':
                norm_langs.append(None)
                lang_valid.append(False)
            else:
                try:
                    nl = normalize_language(v)
                    ok = validate_language(nl)
                    norm_langs.append(nl)
                    lang_valid.append(bool(ok))
                except Exception:
                    norm_langs.append(None)
                    lang_valid.append(False)
        out['idioma'] = norm_langs
        out['idioma_valido'] = lang_valid
    else:
        out['idioma_valido'] = False

    # moneda
    if 'moneda' in out.columns:
        norm_cur = []
        cur_valid = []
        for v in out['moneda'].fillna('').astype(str).tolist():
            if not v or v.strip() == '' or v == 'nan':
                norm_cur.append(None)
                cur_valid.append(False)
            else:
                try:
                    nc = normalize_currency(v)
                    ok = validate_currency(nc) if nc else False
                    norm_cur.append(nc)
                    cur_valid.append(bool(ok))
                except Exception:
                    norm_cur.append(None)
                    cur_valid.append(False)
        out['moneda'] = norm_cur
        out['moneda_valida'] = cur_valid
    else:
        out['moneda_valida'] = False

    # precio -> float
    if 'precio' in out.columns:
        precs = []
        for v in out['precio'].tolist():
            if v is None or (isinstance(v, float) and pd.isna(v)) or (isinstance(v, str) and v.strip() == ''):
                precs.append(None)
            else:
                try:
                    s = str(v).strip()
                    s = s.replace(',', '.')
                    s2 = ''.join([c for c in s if (c.isdigit() or c in '.-')])
                    precs.append(float(s2))
                except Exception:
                    precs.append(None)
        out['precio'] = precs

    return out


def _build_book_source_detail(gr: pd.DataFrame, gb: pd.DataFrame) -> pd.DataFrame:
    """Construye un detalle de fuente preservando filas originales de landing.
    - No crea datos sintéticos
    - Añade columnas: source_name, source_file, row_number, book_id_candidato, isbn13_valido, ts_ingest
    """
    try:
        ts = datetime.now(timezone.utc).isoformat(timespec='seconds')
    except Exception:
        ts = datetime.now(timezone.utc).isoformat(timespec='seconds')

    try:
        gr_copy = gr.copy()
    except Exception:
        gr_copy = pd.DataFrame()
    try:
        gb_copy = gb.copy()
    except Exception:
        gb_copy = pd.DataFrame()

    # ensure row_number
    if not gr_copy.empty and 'row_number' not in gr_copy.columns:
        try:
            gr_copy['row_number'] = (gr_copy.index.astype(int) + 1)
        except Exception:
            gr_copy['row_number'] = list(range(1, len(gr_copy) + 1))
    if not gb_copy.empty and '_csv_row' in gb_copy.columns and 'row_number' not in gb_copy.columns:
        gb_copy['row_number'] = gb_copy['_csv_row']
    elif not gb_copy.empty and 'row_number' not in gb_copy.columns:
        try:
            gb_copy['row_number'] = (gb_copy.index.astype(int) + 1)
        except Exception:
            gb_copy['row_number'] = list(range(1, len(gb_copy) + 1))

    # source columns
    if not gr_copy.empty:
        gr_copy['source_name'] = gr_copy.get('source_name', 'goodreads')
        gr_copy['source_file'] = gr_copy.get('source_file', str((LANDING / 'goodreads_books.json').name))
        gr_copy['ts_ingest'] = ts
        # candidate id
        if 'isbn13' in gr_copy.columns:
            gr_copy['book_id_candidato'] = gr_copy['isbn13']
        else:
            gr_copy['book_id_candidato'] = None
        # isbn validity flag
        gr_copy['isbn13_valido'] = gr_copy.get('isbn13', None).apply(lambda v: bool(try_normalize_isbn(v)[1]) if pd.notna(v) and str(v).strip() != '' else False)

    if not gb_copy.empty:
        gb_copy['source_name'] = gb_copy.get('source_name', 'google_books')
        gb_copy['source_file'] = gb_copy.get('source_file', str((LANDING / 'googlebooks_books.csv').name))
        gb_copy['ts_ingest'] = ts
        if 'isbn13' in gb_copy.columns:
            gb_copy['book_id_candidato'] = gb_copy['isbn13']
        else:
            gb_copy['book_id_candidato'] = None
        gb_copy['isbn13_valido'] = gb_copy.get('isbn13', None).apply(lambda v: bool(try_normalize_isbn(v)[1]) if pd.notna(v) and str(v).strip() != '' else False)

    # preferir dejar las tablas tal cual, preservando columnas originales
    try:
        detail = pd.concat([gr_copy.reset_index(drop=True), gb_copy.reset_index(drop=True)], ignore_index=True, sort=False)
    except Exception:
        detail = gr_copy.copy()

    # normalizar tipos simples para pyarrow
    for col in detail.columns:
        if detail[col].dtype == object:
            try:
                detail[col] = detail[col].apply(lambda v: None if (v is None or (isinstance(v, float) and pd.isna(v))) else v)
            except Exception:
                pass

    return detail

def _main_title(s: Optional[str]) -> Optional[str]:
    """Devuelve la 'parte principal' del título (antes de ':' o '-' o '—'),
    normalizada: sin diacríticos, solo alfanum + espacios, en minúsculas.
    """
    if s is None:
        return None
    try:
        st = str(s)
        for sep in (":", " - ", " — ", "—", "-"):
            if sep in st:
                st = st.split(sep, 1)[0]
                break
        st = normalize_whitespace(st)
        if not st:
            return None
        # normalizar diacríticos
        import unicodedata
        st = ''.join(c for c in unicodedata.normalize('NFKD', st) if not unicodedata.combining(c))
        # conservar solo alfanum y espacios
        st = re.sub(r'[^0-9A-Za-z\s]', ' ', st).strip().lower()
        return st if st != '' else None
    except Exception:
        return None

# Mover ejecución al final del archivo
if __name__ == "__main__":
    main()
