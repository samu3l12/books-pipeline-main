# Esquema canónico: `dim_book.parquet`

Este documento describe el modelo canónico generado por `src/integrate_pipeline.py` y las reglas de mapeo, normalización y supervivencia entre fuentes (`goodreads`, `google_books`). Está pensado para cumplir la rúbrica del proyecto.

Formato general
- Formato de archivo: Parquet (Apache Arrow)
- Convenciones:
  - Nombres de columnas en snake_case.
  - Listas serializadas como `;`-separated strings para compatibilidad con Parquet inicial.
  - Fechas: ISO-8601 `YYYY-MM-DD`. Si falta día/mes se rellena con `01` y se marca `fecha_publicacion_parcial = true`.
  - Idioma: BCP-47 (ej.: `es`, `en-US`).
  - Moneda: ISO-4217 (ej.: `EUR`, `USD`).

Campos mínimos (nombre, tipo, nullability, formato, ejemplo, regla y fuente preferente)

- `book_id` (string, not null)
  - Descripción: Identificador canónico. Preferente: `isbn13` válido; fallback: SHA1 de (titulo, autor_principal, editorial, anio_publicacion).
  - Regla: `if isbn13_valido then book_id = isbn13 else book_id = sha1(titulo|autor|editorial|anio)`.
  - Ejemplo: `9781449374273` o `3f79bb7b435b05321651daefd374cd21`.
  - Fuente: derivado (integración).

- `titulo` (string, not null)
  - Normalización: trim y collapse whitespace; `titulo_normalizado` se genera en minúsculas para matching.
  - Ejemplo: `Data Science for Business`.
  - Fuente preferente: `goodreads` > `google_books` (si `goodreads` aporta título no nulo se usa).

- `titulo_normalizado` (string, nullable)
  - Formato: `normalize_whitespace(lower(titulo))`.

- `autor_principal` (string, nullable)
  - Fuente preferente: `goodreads` luego `google_books`.

- `autores` (string, nullable)
  - Formato: lista serializada separada por `;` (ej.: `Autor A;Autor B`).

- `editorial` (string, nullable)
  - Fuente preferente: `google_books` si contiene valor más completo.

- `anio_publicacion` (Int, nullable)
  - Derivado de `fecha_publicacion` (primeros 4 dígitos) si existe.

- `fecha_publicacion` (string, nullable)
  - Formato: `YYYY-MM-DD` (ISO). Si parser devuelve parcial (AÑO o AÑO-MES) se normaliza añadiendo `-01` y `fecha_publicacion_parcial = true`.
  - Validación: `parse_date_to_iso` en `src/utils_quality.py`.

- `fecha_publicacion_parcial` (boolean)
  - True si la fecha fue completada por heurística (p. ej. solo año o año-mes disponible).

- `idioma` (string, nullable)
  - Normalización: `normalize_language(code)` (BCP-47-like). Validación: `validate_language`.
  - Ejemplo: `en`, `es-ES`.

- `isbn10` (string, nullable)
  - Normalización: string; intentar normalizar con `try_normalize_isbn`.

- `isbn13` (string, nullable)
  - Normalización: string; validación con `is_valid_isbn13`.

- `categoria` (string, nullable)
  - Formato: `;`-separated categories.

- `paginas` (Int, nullable)
  - Tipo nullable `Int64` en pandas/Parquet.

- `formato` (string, nullable)
  - Ej.: `BOOK`, `EBOOK`.

- `precio` (decimal/float, nullable)
  - Normalización: `pd.to_numeric(..., errors='coerce')`.

- `moneda` (string, nullable)
  - Normalización: `normalize_currency(code)` y validación con `validate_currency`.
  - Ejemplo: `EUR`, `USD`.

- `fuente_ganadora` (string, nullable)
  - Valores: `google_books` o `goodreads`. Regla: preferir la fuente con más campos completos (editorial, fecha_publicacion, autores) o `isbn13` si está presente.

- `ts_ultima_actualizacion` (string, not null)
  - ISO timestamp UTC (ej.: `2025-11-17T03:19:47+00:00`).

Flags y metadatos por fila (validación)
- `isbn13_valido` (boolean)
- `idioma_valido` (boolean)
- `moneda_valida` (boolean)
- `fecha_publicacion_valida` (boolean)
- `gb_match_score` (float, nullable)
- `titulo_source`, `autor_principal_source`, `editorial_source`, `fecha_publicacion_source`, `precio_source` (string) — indican la fuente que aportó el valor final.

Archivo `book_source_detail.parquet`
- Contiene por cada fila original de cada fuente un registro con las columnas:
  - `source_id` (string): identificador del registro en la fuente (rec_id, url o índice).
  - `source_name` (string): `goodreads` o `google_books`.
  - `source_file` (string): nombre del fichero en `landing/`.
  - `row_number` (int): fila original en el CSV (para Google Books) o índice en el JSON.
  - `book_id_candidato` (string): isbn13 o id_canónico calculado para candidato.
  - `campos_originales` (JSON/string): mapeo original de campos leídos de la fuente.
  - `flags_validacion` (JSON/string): `isbn13_valido`, `idioma_valido`, `moneda_valida`, `fecha_publicacion_valida`.
  - `provenance_by_field` (JSON/string): para cada campo final en `dim_book` se registra la fuente que aportó el valor.
  - `ts_ingesta` (string): timestamp ISO de la ingesta.

Reglas de deduplicación y supervivencia (resumen)
1. Clave primaria preferente: `isbn13` válido. Si existe, agrupar por `isbn13`.
2. Si falta `isbn13`, generar `merge_key = titulo_normalizado|autor_principal_normalizado` y agrupar por `merge_key`.
3. Selección del registro sobreviviente (por grupo):
   - Preferir registro con `isbn13_valido`.
   - Preferir registro con mayor número de campos no nulos (completitud).
   - Preferir fuente con prioridad configurable (por defecto: `google_books` > `goodreads`).
   - Si empate, preferir `anio_publicacion` más reciente y luego `precio` no nulo.
4. Merge de campos: tomar el primer valor no nulo según la prioridad de fuentes; para listas (autores, categorias) unir y desduplicar preservando orden.

Provenance y trazabilidad
- Cada decisión de asignación (por ejemplo asignación de ISBN desde candidatos) queda registrada en `work/logs/rules/integrate_rules_<fecha>.jsonl` mediante `log_rule_jsonl`.
- Cada request a Google Books y cada scraping a GoodReads queda registrado en `work/logs/requests/<source>_<yyyyMMdd>.csv` mediante `log_request_csv`.
- Se guarda un resumen del run en `work/logs/runs/run_<ts>.json` con métricas parciales.

Notas finales
- `docs/schema.md` debe mantenerse sincronizado con el código; cualquier cambio en `src/integrate_pipeline.py` que añada/renombre campos debe reflejarse aquí.
- `docs/quality_metrics.json` contiene las métricas calculadas por ejecución y sirve para validar la rúbrica.

