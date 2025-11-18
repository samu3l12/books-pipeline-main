# Esquema canónico: `dim_book.parquet`

Este documento describe el modelo canónico generado por `src/integrate_pipeline.py` y las reglas de mapeo, normalización y supervivencia entre fuentes (`goodreads`, `google_books`). Está pensado para cumplir la rúbrica del proyecto y servir como referencia rápida.

Formato general
- Formato de archivo: Parquet (Apache Arrow)
- Convenciones:
  - Nombres de columnas en snake_case.
  - Listas: en Parquet pueden almacenarse como listas nativas; cuando se serializan para compatibilidad se usan `;`-separated strings.
  - Fechas: ISO-8601 `YYYY-MM-DD`. Si falta día/mes se rellena con `01` y se marca `fecha_publicacion_parcial = true`.
  - Idioma: BCP-47 (ej.: `es`, `en-US`).
  - Moneda: ISO-4217 (ej.: `EUR`, `USD`).

Resumen de tablas estándar
- `standard/dim_book.parquet`: tabla dimensional canónica (una fila por libro canónico)
- `standard/book_source_detail.parquet`: detalle por fila original de cada fuente con trazabilidad

Campos clave en `dim_book.parquet` (tipos y reglas resumidas)
- `book_id` (string, not null)
  - Identificador canónico. Regla: preferir `isbn13` normalizado y válido; si no hay `isbn13` válido, usar SHA1 de (titulo, autor_principal, editorial, anio_publicacion).
  - Ejemplo: `9781449374273` o `3f79bb7b435b05321651daefd374cd21`.

- `titulo` (string, not null)
  - Normalización: trim y collapse whitespace. `titulo_normalizado` (lowercased, whitespace normalized) se usa para matching.

- `titulo_normalizado` (string, nullable)
  - Formato: `normalize_whitespace(lower(titulo))`.

- `autor_principal` (string, nullable)
  - Fuente preferente: `goodreads` > `google_books`.

- `autores` (list[string] o string serializado, nullable)
  - Mantener lista nativa cuando Parquet lo soporte; si no, usar `;` como separador.

- `editorial` (string, nullable)
  - Fuente preferente: `google_books` cuando añade más detalle.

- `fecha_publicacion` (string, nullable)
  - ISO `YYYY-MM-DD`. Metadata adicional: `fecha_publicacion_parcial` (bool) y `fecha_publicacion_valida` (bool).
  - Parsing: `parse_date_to_iso` en `src/utils_quality.py`.

- `idioma` (string, nullable)
  - Normalizado con `normalize_language`; flag `idioma_valido`.

- `isbn10` (string, nullable)
  - Preservar si existe; BUT: cuando `isbn13` está vacío y `isbn10` existe y es válido, convertir a `isbn13` (ver regla abajo).

- `isbn13` (string, nullable)
  - Normalizar y validar con `try_normalize_isbn` / `is_valid_isbn13`. `isbn13_norm` y `isbn13_valido` se almacenan para auditoría.

- `categoria` (list[string] o string, nullable)
- `paginas` (Int64, nullable)
- `formato` (string, nullable) — ej.: `BOOK`, `EBOOK`
- `precio` (float, nullable)
- `moneda` (string, nullable) — normalizar con `normalize_currency`; flag `moneda_valida`

Campos de trazabilidad y métricas
- `provenance` (JSON/string)
  - Map por campo final indicando la fuente que aportó el valor (`goodreads`, `google_books`, `merged`, etc.). Ej.: `{"titulo":"goodreads","isbn13":"google_books"}`.
- `ts_ultima_actualizacion` (string, not null)
  - Timestamp ISO UTC de la última actualización/ingesta.
- Flags: `isbn13_valido`, `idioma_valido`, `moneda_valida`, `fecha_publicacion_valida`.

Reglas importantes (dedupe y supervivencia)
1. Preferencia de identificación
   - ISBN‑13 válido es la clave primaria preferente. Si existe `isbn13_valido`, agrupar/identificar por `isbn13`.
   - Si no existe `isbn13`, construir `merge_key = titulo_normalizado|autor_principal_normalizado` y agrupar por `merge_key`.
2. Conversión ISBN
   - Si una fila fuente solo aporta `isbn10` válido y no hay `isbn13`, convertir `isbn10` → `isbn13` (prefijando `978` y recalculando checksum) usando `try_normalize_isbn` y usar el ISBN‑13 resultante como `isbn13` y `book_id`.
   - Esto garantiza que el identificador canónico sea `isbn13` cuando sea posible.
3. Selección del representante (supervivencia)
   - Prioridad general: preferir registros con `isbn13_valido`.
   - Dentro del mismo grupo (mismo `isbn13` o mismo `merge_key`): seleccionar la fila con mayor completitud (más campos no nulos relevantes: titulo, autor_principal, autores, editorial, fecha_publicacion, paginas).
   - Si empate en completitud: no imponer preferencia arbitraria entre fuentes; registrar el empate en logs y conservar el primer encuentro (o aplicar criterio configurable: ejemplo `preferencia_fuente = ['google_books','goodreads']` si se define explícitamente).
   - NO usar matching difuso automático para crear `isbn13` — sólo matching exacto en `_match_key` o título único salvo que el usuario autorice heurísticas adicionales.
4. Merge de campos
   - Para cada campo final: tomar el primer valor no nulo según la prioridad de fuentes definida (por defecto: `goodreads` para título/autor cuando esté presente; `google_books` para metadata editorial/fecha/precio).
   - Para listas (autores, categorias): unir candidatos, desduplicar preservando orden.
5. Provenance
   - Para cada campo final se debe registrar la fuente que aportó el valor (ej.: `provenance` JSON). Si el valor final fue convertido (ej. `isbn10`→`isbn13`), la procedencia registra la fuente original y que el `isbn13` fue derivado de `isbn10` de `google_books`.

`book_source_detail.parquet` (detalle y trazabilidad)
- Objetivo: conservar una fila por registro original de las fuentes con metadatos de validación y trazabilidad.
- Columnas clave:
  - `source_name` (`goodreads` | `google_books`)
  - `source_file` (nombre del fichero en `landing/`)
  - `row_number` (int): índice original en la fuente
  - `book_id_candidato` (string): isbn13/isbn10 o id calculado para el candidato
  - `isbn13_valido`,`ts_ingest` (timestamp)
  - `provenance_by_field` / `campos_originales`: snapshot de campos originales y flags de validación

Logs y trazabilidad
- Las decisiones relevantes (ambigüedades, asignaciones por matching, conversiones ISBN) se registran en `work/logs/rules/integrate_rules_<fecha>.jsonl` mediante `log_rule_jsonl`.
- Las requests a Google Books y las ejecuciones del scraper se loguean en `work/logs/requests/`.
- Resumen del run y métricas se guardan en `docs/quality_metrics.json`.

Notas operativas y de evolución
- No eliminar columnas all-null de forma automática: la eliminación debe ser deliberada y documentada (puede perderse trazabilidad si se hace por defecto).
- Mantener sincronizado `docs/schema.md` con cambios en `src/integrate_pipeline.py` y utilidades.
- Para depurar problemas de matching y provenance, usar `scripts/diagnose_parquet.py` (herramienta de diagnóstico incluida) y revisar `work/logs/rules`.
