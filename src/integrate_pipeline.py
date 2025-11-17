"""
Integración de Goodreads (JSON) y Google Books (CSV) a artefactos estándar.
Produce standard/dim_book.parquet, standard/book_source_detail.parquet y docs/quality_metrics.json
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Tuple, Mapping, Dict

import numpy as np
import pandas as pd

from utils_isbn import is_valid_isbn13
from utils_quality import (
    compute_quality_metrics,
    listify,
    normalize_language,
    normalize_whitespace,
    parse_date_to_iso,
    uniq_preserve,
    validate_currency,
    validate_language,  # añadido para validar idiomas correctamente
)

ROOT = Path(__file__).resolve().parents[1]
LANDING = ROOT / "landing"
STANDARD = ROOT / "standard"
DOCS = ROOT / "docs"


def _canonical_key(title: str, author: str, publisher: Optional[str], year: Optional[int]) -> str:
    # DEDUPLICACION: genera clave canónica estable para fallback cuando isbn13 no existe
    # KEYWORD: GENERAR_BOOK_KEY, DEDUPLICACION
    def _norm(x: Optional[object]) -> str:
        if x is None or (isinstance(x, float) and pd.isna(x)) or (hasattr(pd, "isna") and pd.isna(x)):  # type: ignore[arg-type]
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
    """Carga archivos de landing asegurando ISBN como string para evitar coerción a int.
    KEYWORD: CARGA_FUENTES
    """
    gr_path = LANDING / "goodreads_books.json"
    gb_path = LANDING / "googlebooks_books.csv"

    with open(gr_path, "r", encoding="utf-8") as f:
        gr_json = json.load(f)
    gr = pd.DataFrame(gr_json.get("records", []))
    gr["source_name"] = "goodreads"
    gr["source_file"] = str(gr_path)

    # Forzar lectura de ISBN como texto.
    gb = pd.read_csv(gb_path, dtype={"isbn13": "string", "isbn10": "string"})
    gb["source_name"] = "google_books"
    gb["source_file"] = str(gb_path)
    return gr, gb


def _normalize_frames(gr: pd.DataFrame, gb: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # Goodreads
    gr = gr.copy()
    gr.rename(columns={
        "title": "titulo",
        "author": "autor_principal",
        "isbn13": "isbn13",
        "isbn10": "isbn10",
    }, inplace=True)
    # Asegurar isbn como string consistente
    for c in ("isbn13", "isbn10"):
        if c in gr.columns:
            gr[c] = gr[c].apply(lambda v: str(v).strip() if pd.notna(v) and str(v).strip() != "" else None)
    gr["idioma"] = None
    gr["categoria"] = None
    gr["fecha_publicacion"] = None
    gr["precio"] = None
    gr["moneda"] = None

    # Google Books
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
    # autores/categoria como listas normalizadas serializadas con ';'
    if "autores" in gb.columns:
        gb["autores"] = gb["autores"].apply(lambda x: ";".join(uniq_preserve(listify(x))) if pd.notna(x) else None)
        # extraer autor_principal desde autores para emparejamiento
        gb["autor_principal"] = gb["autores"].apply(lambda x: str(x).split(";")[0] if pd.notna(x) and str(x).strip() != "" else None)
    else:
        gb["autor_principal"] = None
    if "categoria" in gb.columns:
        gb["categoria"] = gb["categoria"].apply(lambda x: ";".join(uniq_preserve(listify(x))) if pd.notna(x) else None)
    # idioma normalizado
    if "idioma" in gb.columns:
        gb["idioma"] = gb["idioma"].apply(lambda x: normalize_language(x) if pd.notna(x) else None)

    # Normalizar títulos y autores para emparejamiento secundario (cuando isbn faltante)
    gb["_match_title"] = gb["titulo"].apply(lambda s: normalize_whitespace(str(s).lower()) if pd.notna(s) else None)
    gb["_match_author"] = gb["autor_principal"].apply(lambda s: normalize_whitespace(str(s).lower()) if pd.notna(s) else None)

    return gr, gb


def _merge_sources(gr: pd.DataFrame, gb: pd.DataFrame) -> pd.DataFrame:
    # MERGE / NORMALIZACION
    # KEYWORD: MERGE_ISBN, SELECCION_CAMPOS
    left = gr.copy()
    right = gb.copy()

    # RELLENAR isbn13 en Goodreads cuando exista coincidencia exacta de titulo_normalizado + autor_principal
    # Esto evita que registros idénticos queden duplicados tras el merge outer por isbn null
    def _mk_match_key(df):
        t = df.get("titulo") if "titulo" in df else None
        a = df.get("autor_principal") if "autor_principal" in df else None
        t_norm = normalize_whitespace(str(t).lower()) if pd.notna(t) else None
        a_norm = normalize_whitespace(str(a).lower()) if pd.notna(a) else None
        if t_norm and a_norm:
            return f"{t_norm}|{a_norm}"
        if t_norm:
            return f"{t_norm}|"
        return None

    # crear keys en ambos marcos
    left["_match_key"] = left.apply(lambda r: _mk_match_key(r), axis=1)
    right["_match_key"] = right.apply(lambda r: (str(r.get("_match_title")) + "|" + str(r.get("_match_author"))).strip("None") if (r.get("_match_title") is not None or r.get("_match_author") is not None) else None, axis=1)

    # mapa de match_key -> isbn13 desde google_books
    gb_key_to_isbn = {}
    for _, row in right.iterrows():
        key = row.get("_match_key")
        isbn = row.get("isbn13")
        if key and isbn:
            gb_key_to_isbn.setdefault(key, isbn)

    # rellenar isbn13 en left cuando falta y existe en map
    def _fill_isbn_from_key(row):
        if pd.notna(row.get("isbn13")) and str(row.get("isbn13")).strip() != "":
            return row.get("isbn13")
        key = row.get("_match_key")
        if key and key in gb_key_to_isbn:
            return gb_key_to_isbn[key]
        return row.get("isbn13")

    left["isbn13"] = left.apply(_fill_isbn_from_key, axis=1)

    # Evitar outer merge que puede producir muchas filas cuando isbn13 es null.
    # Estrategia: 1) hacer left-join de Goodreads <- GoogleBooks por isbn13 (para filas de Goodreads);
    # 2) añadir filas de GoogleBooks cuyo isbn13 no fue emparejado con ningún Goodreads (gb-only).
    # Esto produce una unión sin multiplicar combinaciones de rows con isbn null.
    merged_left = pd.merge(left, right, on="isbn13", how="left", suffixes=("_gr", "_gb"))

    # Identificar isbn13 presentes en left (post-fill) para excluirlos de gb-only
    left_isbns = set([v for v in merged_left['isbn13'].dropna().unique()])
    # gb_only: filas de right con isbn13 que no están en left_isbns OR filas de right con isbn13 null (no emparejadas)
    gb_only_mask = ~right['isbn13'].isin(left_isbns)
    gb_only = right[gb_only_mask].copy()
    # Expandir gb_only para mantener la misma estructura que merged_left: agregar sufijos '_gr' nulos
    if not gb_only.empty:
        # Renombrar columnas para que concuerden cuando se concatena con merged_left
        gb_only_renamed = gb_only.rename(columns={k: f"{k}" for k in gb_only.columns})
        # Para las columnas que aparecerán como *_gr en merged_left, crear versiones vacías
        gb_cols = list(gb_only_renamed.columns)
        # Construir DataFrame con columnas combinadas: tomar columnas de merged_left y rellenar con valores de gb_only
        # Simplificar: crear rows con prefijo _gb en lugar de duplicar sufijos. Luego transformaremos al formato esperado abajo.
        # Para simplicidad, construiremos un merged equivalente concatenando merged_left y gb_only con columnas alineadas vía reindex.
        # Obtener columnas resultantes esperadas tras merge (approx): combinación de cols_gr and cols_gb
        # Para robustez, vamos a construir un DataFrame final a partir de concatenación de merged_left y una versión 'pseudo-merged' de gb_only.
        # Crear pseudo-merged para gb_only: crear columnas terminadas en _gr con NaN y mantener columnas _gb sin sufijo.
        for col in gb_only_renamed.columns:
            # si col ya existe en merged_left como 'col_gr' o 'col', no hacemos nada
            pass
    # Para mantener compatibilidad con el resto del pipeline, volver a usar merged = merged_left + gb_only rows
    if gb_only is not None and not gb_only.empty:
        # Convertir gb_only a formato similar al obtenido en merged_left: prefijar las columnas originales como *_gb
        gb_prefixed = gb_only.copy()
        # Asegurar que las columnas que merged_left espera (e.g., titulo_gr, titulo) estén presentes; generar columnas *_gr vacías
        # Primero, identificar columnas de right (sin sufijo) que merged_left contiene como 'col' y las que existen con '_gb'
        # En merged_left, las columnas provenientes de right aparecen sin sufijo (por cómo se hizo merge left),
        # pero las columnas de left mantienen sufijo _gr si hay conflicto. Para simplificar, renombraremos gb_prefixed cols a su forma original
        # y luego concatenaremos rows: merged_left (tiene columnas combinadas) y gb_only_rows construidas manualmente.
        # Construir gb_only_rows en la forma de merged_left: mapear columnas del right a las columnas sin sufijo en merged_left,
        # y crear columnas *_gr con NaN para las columnas de left.
        gb_only_rows = []
        for _, r in gb_only.iterrows():
            # crear dict con todas las columnas que merged_left tiene
            row = {}
            # columnas de left (prefijo _gr) vendrán como None
            # intentaremos inferir lista de left columns desde merged_left
            for c in merged_left.columns:
                row[c] = None
            # rellenar las columnas sin sufijo (propias de right) con valores de r
            for c in gb_only.columns:
                # si merged_left contiene columna con mismo nombre, asignar
                if c in row:
                    row[c] = r.get(c)
                else:
                    # intentar colocar en versión sin sufijo
                    row[c] = r.get(c)
            gb_only_rows.append(row)
        gb_only_df = pd.DataFrame(gb_only_rows)
        merged = pd.concat([merged_left, gb_only_df], ignore_index=True, sort=False)
    else:
        merged = merged_left

    # merged ahora contiene la unión sin productos cartesianos

    def pick(a, b):
        # UTIL: selección simple primer valor no-nulo
        return a if pd.notna(a) and a != "" else (b if pd.notna(b) and b != "" else None)

    out = pd.DataFrame()
    out["isbn13"] = merged["isbn13"].where(merged["isbn13"].notna(), None)
    out["isbn10"] = merged.apply(lambda r: pick(r.get("isbn10_gr"), r.get("isbn10")), axis=1)
    out["titulo"] = merged.apply(lambda r: pick(r.get("titulo_gr"), r.get("titulo")), axis=1)
    # origen de título
    out["titulo_source"] = merged.apply(lambda r: ("goodreads" if pd.notna(r.get("titulo_gr")) and r.get("titulo_gr") not in ("", None) else ("google_books" if pd.notna(r.get("titulo")) and r.get("titulo") not in ("", None) else None)), axis=1)
    out["autor_principal"] = merged.apply(lambda r: pick(r.get("autor_principal"), (r.get("autores") or "").split(";")[0] if pd.notna(r.get("autores")) and r.get("autores") else None), axis=1)
    out["autor_principal_source"] = merged.apply(lambda r: ("goodreads" if pd.notna(r.get("autor_principal")) and r.get("autor_principal") not in ("", None) else ("google_books" if pd.notna(r.get("autores")) and r.get("autores") not in ("", None) else None)), axis=1)
    out["autores"] = merged.get("autores") if "autores" in merged.columns else None
    out["editorial"] = merged.get("editorial") if "editorial" in merged.columns else None
    # origen editorial: se busca primero editorial_gr, luego editorial (google)
    out["editorial_source"] = merged.apply(lambda r: ("goodreads" if pd.notna(r.get("editorial_gr")) and r.get("editorial_gr") not in ("", None) else ("google_books" if pd.notna(r.get("editorial")) and r.get("editorial") not in ("", None) else None)), axis=1)

    def parse_fecha(x):
        val, parcial = parse_date_to_iso(x)
        return val
    out["fecha_publicacion"] = merged.apply(lambda r: pick(r.get("fecha_publicacion_gr"), parse_fecha(r.get("fecha_publicacion"))), axis=1)
    out["fecha_publicacion_source"] = merged.apply(lambda r: ("goodreads" if pd.notna(r.get("fecha_publicacion_gr")) and r.get("fecha_publicacion_gr") not in ("", None) else ("google_books" if pd.notna(r.get("fecha_publicacion")) and r.get("fecha_publicacion") not in ("", None) else None)), axis=1)

    out["idioma"] = merged["idioma"] if "idioma" in merged.columns else None
    out["precio"] = merged["precio"] if "precio" in merged.columns else None
    out["moneda"] = merged["moneda"] if "moneda" in merged.columns else None
    out["precio_source"] = merged.apply(lambda r: ("goodreads" if pd.notna(r.get("precio_gr")) and r.get("precio_gr") not in ("", None) else ("google_books" if pd.notna(r.get("precio") ) and r.get("precio") not in ("", None) else None)), axis=1)

    out["categoria"] = merged.apply(lambda r: pick(r.get("categoria_gr"), r.get("categoria")), axis=1) if "categoria_gr" in merged.columns or "categoria" in merged.columns else None

    out["titulo_normalizado"] = out["titulo"].apply(lambda s: normalize_whitespace(str(s).lower()) if pd.notna(s) else None)
    out["anio_publicacion"] = out["fecha_publicacion"].apply(lambda s: int(str(s)[:4]) if pd.notna(s) else None)

    def mk_book_id(r):
        # GENERAR_BOOK_ID: preferir isbn13 válido; si no existe, usar clave canónica SHA1
        # KEYWORD: GENERAR_BOOK_ID, BOOK_ID, DEDUP_LOGIC, REGION_DEDUP
        if r.get("isbn13") and is_valid_isbn13(str(r.get("isbn13"))):
            return str(r.get("isbn13"))
        return _canonical_key(r.get("titulo") or "", r.get("autor_principal") or "", r.get("editorial"), r.get("anio_publicacion"))

    out["book_id"] = out.apply(mk_book_id, axis=1)

    out["isbn13_valido"] = out["isbn13"].apply(lambda x: bool(is_valid_isbn13(str(x))) if pd.notna(x) else False)
    out["idioma_valido"] = out["idioma"].apply(lambda x: validate_language(x) if pd.notna(x) else False)
    out["moneda_valida"] = out["moneda"].apply(lambda x: validate_currency(x))
    out["fecha_publicacion_valida"] = out["fecha_publicacion"].apply(lambda x: isinstance(x, str) and len(x) == 10)

    def fuente(r):
        prefer_gb = any(pd.notna(r.get(c)) for c in ("editorial", "fecha_publicacion", "autores"))
        return "google_books" if prefer_gb else "goodreads"

    out["fuente_ganadora"] = out.apply(fuente, axis=1)
    out["ts_ultima_actualizacion"] = datetime.now(timezone.utc).isoformat(timespec="seconds")

    cols = [
        "book_id","titulo","titulo_normalizado","autor_principal","autores","editorial","anio_publicacion","fecha_publicacion","idioma","isbn10","isbn13","categoria","precio","moneda","fuente_ganadora","ts_ultima_actualizacion",
        "isbn13_valido","idioma_valido","moneda_valida","fecha_publicacion_valida",
    ]
    # Insertar columnas de provenance al esquema (al final de lista para compatibilidad)
    cols = cols + ["titulo_source","autor_principal_source","editorial_source","fecha_publicacion_source","precio_source"]
    out = out[cols]
    # Forzar tipos string para ISBN antes de retorno
    for c in ("isbn10","isbn13"):
        if c in out.columns:
            out[c] = out[c].apply(lambda v: str(v).strip() if pd.notna(v) and str(v).strip() != "" else None)
    return out


# REGION: DEDUPLICATION - LÓGICA DE ELIMINACIÓN DE DUPLICADOS
# KEYWORD: DEDUP_SECTION, DEDUPLICACION_DROP, SELECCION_REGISTRO_PREFERIDO

def _build_source_detail(gr: pd.DataFrame, gb: pd.DataFrame, dim: pd.DataFrame) -> pd.DataFrame:
    """Construye tabla detalle incluyendo flags de validación por registro.
    Añade información de candidatos y score si existe `landing/googlebooks_candidates.json`.
    KEYWORD: SOURCE_DETAIL_CANDIDATES, MATCH_TRACE
    """
    gr_tmp = gr.copy()
    gb_tmp = gb.copy()

    gr_tmp["book_id_candidato"] = gr_tmp.apply(lambda r: str(r.get("isbn13")) if pd.notna(r.get("isbn13")) and is_valid_isbn13(str(r.get("isbn13"))) else _canonical_key(r.get("titulo") or r.get("title"), r.get("autor_principal") or r.get("author"), None, None), axis=1)
    gb_tmp["book_id_candidato"] = gb_tmp.apply(lambda r: str(r.get("isbn13")) if pd.notna(r.get("isbn13")) and is_valid_isbn13(str(r.get("isbn13"))) else _canonical_key(r.get("titulo") or r.get("title"), (str(r.get("autores")).split(";")[0] if pd.notna(r.get("autores")) else None), r.get("editorial") or r.get("publisher"), None), axis=1)

    gr_tmp["row_number"] = np.arange(1, len(gr_tmp) + 1)
    gb_tmp["row_number"] = np.arange(1, len(gb_tmp) + 1)

    gr_tmp["source_name"] = "goodreads"
    gb_tmp["source_name"] = "google_books"

    gr_tmp["source_file"] = str(LANDING / "goodreads_books.json")
    gb_tmp["source_file"] = str(LANDING / "googlebooks_books.csv")

    gr_sel = gr_tmp[[
        "source_name","source_file","row_number","book_id_candidato","titulo","autor_principal","isbn10","isbn13"
    ]].rename(columns={"titulo": "raw_title", "autor_principal": "raw_author"})

    gb_sel = gb_tmp[[
        "source_name","source_file","row_number","book_id_candidato","titulo","autores","editorial","fecha_publicacion","idioma","categoria","isbn10","isbn13","precio","moneda"
    ]].rename(columns={
        "titulo": "raw_title",
        "autores": "raw_authors",
        "editorial": "raw_publisher",
        "fecha_publicacion": "raw_pub_date",
        "idioma": "raw_language",
        "categoria": "raw_categories",
        "precio": "raw_price_amount",
        "moneda": "raw_price_currency",
    })

    detail = pd.concat([gr_sel, gb_sel], ignore_index=True)
    detail["timestamp_ingesta"] = datetime.now(timezone.utc).isoformat(timespec="seconds")

    # Añadir flags de validación a nivel registro
    detail["isbn13_valido"] = detail["isbn13"].apply(lambda x: bool(is_valid_isbn13(str(x))) if pd.notna(x) else False)
    detail["idioma_valido"] = detail.get("raw_language", pd.Series([None]*len(detail))).apply(lambda x: validate_language(x) if pd.notna(x) else False)
    detail["moneda_valida"] = detail.get("raw_price_currency", pd.Series([None]*len(detail))).apply(lambda x: validate_currency(x))
    detail["fecha_publicacion_valida"] = detail.get("raw_pub_date", pd.Series([None]*len(detail))).apply(lambda x: isinstance(x, str) and len(str(x).strip()) in (4,7,10))

    for c in ("isbn10", "isbn13"):
        if c in detail.columns:
            detail[c] = detail[c].apply(lambda v: str(v).strip() if pd.notna(v) and str(v).strip() != "" else None)

    # Cargar candidatos si existe archivo auxiliar creado por enrich
    candidates_path = LANDING / "googlebooks_candidates.json"
    gb_candidates_map: Dict[str, Dict] = {}
    if candidates_path.exists():
        try:
            with open(candidates_path, "r", encoding="utf-8") as f:
                cand_list = json.load(f)
            # mapear por rec_id o input_index para acceso rápido
            for c in cand_list:
                key = c.get("rec_id") or str(c.get("input_index"))
                gb_candidates_map[key] = c
        except Exception:
            gb_candidates_map = {}

    # Añadir columnas de candidatos y score a detail (solo para registros google_books)
    def _get_candidates_for_row(row):
        if row.get("source_name") != "google_books":
            return None
        # intentar mapear por row_number -> buscar entrada donde csv_row_number == row_number
        matches = [v for k, v in gb_candidates_map.items() if v.get("csv_row_number") == row.get("row_number")]
        if matches:
            return matches[0].get("candidates")
        # fallback: intentar usar title+author match
        key_candidates = [v for k, v in gb_candidates_map.items() if v.get("title") == row.get("raw_title") and v.get("author") == row.get("raw_authors")]
        if key_candidates:
            return key_candidates[0].get("candidates")
        return None

    def _get_best_score_for_row(row):
        if row.get("source_name") != "google_books":
            return None
        matches = [v for k, v in gb_candidates_map.items() if v.get("csv_row_number") == row.get("row_number")]
        if matches:
            return matches[0].get("best_score")
        key_candidates = [v for k, v in gb_candidates_map.items() if v.get("title") == row.get("raw_title") and v.get("author") == row.get("raw_authors")]
        if key_candidates:
            return key_candidates[0].get("best_score")
        return None

    detail["gb_candidate_scores"] = detail.apply(_get_candidates_for_row, axis=1)
    detail["gb_best_score"] = detail.apply(_get_best_score_for_row, axis=1)

    # Añadir flags de provenance: si este registro aportó el valor final en dim
    # Construir mapeo book_id -> registro dim (valores finales)
    try:
        dim_map = dim.set_index("book_id").to_dict(orient="index") if (isinstance(dim, pd.DataFrame) and "book_id" in dim.columns) else {}
    except Exception:
        dim_map = {}

    def _matches_final(row, field_raw, field_dim):
        bid = row.get("book_id_candidato")
        if not bid or bid not in dim_map:
            return False
        final_val = dim_map[bid].get(field_dim)
        raw_val = row.get(field_raw)
        # normalizar para comparación simple
        if pd.isna(final_val) and (raw_val is None or (isinstance(raw_val, float) and pd.isna(raw_val))):
            return False
        try:
            return (str(final_val).strip() != "" and str(raw_val).strip() == str(final_val).strip())
        except Exception:
            return False

    detail["contribuye_titulo"] = detail.apply(lambda r: _matches_final(r, "raw_title", "titulo"), axis=1)
    detail["contribuye_autor_principal"] = detail.apply(lambda r: _matches_final(r, "raw_author" if "raw_author" in detail.columns else "raw_authors", "autor_principal"), axis=1)
    detail["contribuye_editorial"] = detail.apply(lambda r: _matches_final(r, "raw_publisher", "editorial"), axis=1)
    detail["contribuye_fecha_publicacion"] = detail.apply(lambda r: _matches_final(r, "raw_pub_date", "fecha_publicacion"), axis=1)
    detail["contribuye_precio"] = detail.apply(lambda r: _matches_final(r, "raw_price_amount", "precio"), axis=1)

    return detail


def integrate() -> None:
    gr_raw, gb_raw = _load_sources()
    gr_n, gb_n = _normalize_frames(gr_raw, gb_raw)
    dim = _merge_sources(gr_n, gb_n)

    dim = dim.sort_values(["isbn13_valido", "anio_publicacion"], ascending=[False, False])
    # DEDUPLICACION: eliminar duplicados por book_id manteniendo el registro preferido
    # KEYWORD: DEDUPLICACION_DROP
    dim = dim.drop_duplicates(subset=["book_id"], keep="first")

    detail = _build_source_detail(gr_n, gb_n, dim)

    # Contadores de filas de entrada por fuente (para métricas de trazabilidad)
    input_counts = {
        "goodreads": int(len(gr_n)) if hasattr(gr_n, '__len__') else 0,
        "google_books": int(len(gb_n)) if hasattr(gb_n, '__len__') else 0,
    }

    metrics = compute_quality_metrics(dim.to_dict(orient="records"))
    # asegurar tipo dict para poder añadir campos sin problemas de typing
    metrics = dict(metrics)
    # Añadir conteo de filas de entrada para trazabilidad (landing)
    metrics["filas_por_fuente_input"] = input_counts

    STANDARD.mkdir(parents=True, exist_ok=True)
    DOCS.mkdir(parents=True, exist_ok=True)

    dim_path = STANDARD / "dim_book.parquet"
    det_path = STANDARD / "book_source_detail.parquet"
    q_path = DOCS / "quality_metrics.json"

    # Escritura Parquet (ISBN forzados como string serializable)
    # Asegurar dtype string explícito en columnas problemáticas
    def _coerce_isbn_cols(df: pd.DataFrame) -> None:
        """Normaliza columnas isbn10/isbn13 a str o None y valida que no queden tipos mixtos (int/float).
        Lanza RuntimeError con detalle si encuentra valores no convertibles.
        """
        for c in ("isbn10", "isbn13"):
            if c in df.columns:
                # Reemplazar cadenas vacías por None para un tratamiento consistente
                df[c] = df[c].replace({"": None})
                # Convertir NaN/None a None y forzar str() en el resto
                def _to_str_or_none(v: object) -> Optional[str]:
                    try:
                        if v is None or (isinstance(v, float) and pd.isna(v)):
                            return None
                        s = str(v).strip()
                        return s if s != "" else None
                    except Exception:
                        return None

                df[c] = df[c].apply(_to_str_or_none)
                # Validar que no quedan valores no-string; si hay, tomar muestra y fallar con mensaje claro
                bad_vals = [v for v in df[c].dropna().unique() if not isinstance(v, str)]
                if bad_vals:
                    raise RuntimeError(f"Columna {c} contiene valores no-string despu\u00E9s de coercion: {bad_vals[:5]} (muestra)")
                # Establecer dtype pandas 'string' para compatibilidad con pyarrow
                try:
                    df[c] = df[c].astype("string")
                except Exception:
                    # último recurso: convertir todo a Python str con None
                    df[c] = df[c].apply(lambda v: None if v is None else str(v))

    _coerce_isbn_cols(dim)
    _coerce_isbn_cols(detail)
    dim.to_parquet(dim_path, index=False)
    detail.to_parquet(det_path, index=False)

    with open(q_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print(f"Escrito {dim_path} y {det_path}; métricas en {q_path}")


def main() -> None:
    integrate()


if __name__ == "__main__":
    main()
