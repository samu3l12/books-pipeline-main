# books-pipeline (Mini-pipeline de libros)

Resumen

Proyecto con un flujo sencillo Extracción → Enriquecimiento → Integración destinado a obtener una muestra de libros desde Goodreads, enriquecer con la Google Books API y producir artefactos canónicos (.parquet) con controles de calidad y trazabilidad.

Objetivo de este README: documentar exactamente cómo el repo satisface la rúbrica y dónde encontrar evidencia para cada criterio.

Estructura mínima requerida (presente en este repo)

- README.md (este fichero)
- requirements.txt
- .env.example
- landing/
  - goodreads_books.json
  - googlebooks_books.csv
  - googlebooks_candidates.json
- standard/
  - dim_book.parquet
  - book_source_detail.parquet
- docs/
  - schema.md
  - quality_metrics.json
- src/
  - scrape_goodreads.py
  - enrich_googlebooks.py
  - integrate_pipeline.py
  - utils_quality.py
  - utils_isbn.py

Dependencias (ver `requirements.txt`)

- requests
- beautifulsoup4
- lxml
- pandas
- pyarrow
- python-dotenv

Cómo ejecutar (sin parámetros)

Los scripts se ejecutan desde la raíz del proyecto y funcionan con valores por defecto; se recomienda usar PowerShell en Windows:

- Extraer (Goodreads → JSON):
  - python .\src\scrape_goodreads.py
  - Salida: `landing/goodreads_books.json` (UTF-8, JSON con metadatos)

- Enriquecer (Google Books → CSV + candidatos):
  - Añadir variable `GOOGLE_BOOKS_API_KEY` en `.env` o en variables de entorno si se quiere usar la API. Si no existe clave, el script intenta el fallback según implementación.
  - python .\src\enrich_googlebooks.py
  - Salida: `landing/googlebooks_books.csv` (UTF-8, separador coma) y `landing/googlebooks_candidates.json` (trazabilidad de candidatos)

- Integrar (JSON+CSV → Parquet + métricas):
  - python .\src\integrate_pipeline.py
  - Salida: `standard/dim_book.parquet`, `standard/book_source_detail.parquet`, `docs/quality_metrics.json`

Decisiones clave y mapeo contra la rúbrica

A continuación se documenta cada criterio de la rúbrica con la evidencia correspondiente (archivos y ubicaciones):

1) Estructura del repositorio — Estado: Done
   - Evidencia: listado de ficheros en la raíz; `landing/`, `standard/`, `docs/` y `src/` presentes.

2) Scraping Goodreads (JSON válido) — Estado: Done
   - Evidencia: `landing/goodreads_books.json` contiene 15 registros (metadata.record_count = 15) y campos: title/titulo, author/autor_principal, rating, ratings_count, book_url, isbn10/isbn13 (cuando están). Metadatos con `selectors`, `user_agent`, `fetched_at` y `pauses_seconds`.

3) Metadatos de landing y ética de scraping — Estado: Done
   - Evidencia: `landing/goodreads_books.json` -> `metadata.selectors`, `metadata.user_agent`, `metadata.fetched_at`, `metadata.record_count`, `metadata.notes.pauses_seconds`.
   - Pausas y backoff documentadas en `src/scrape_goodreads.py`.

4) Enriquecimiento Google Books (CSV válido) — Estado: Done (parcial normalización)
   - Evidencia: `landing/googlebooks_books.csv` con cabecera UTF-8 y campos: gb_id,title,subtitle,authors,publisher,pub_date,language,categories,isbn13,isbn10,price_amount,price_currency,paginas,formato.
   - `landing/googlebooks_candidates.json` contiene candidatos con score; `src/enrich_googlebooks.py` solicita `maxResults=5` y genera scoring.

5) Modelo canónico y mapa de campos — Estado: Implementado
   - Evidencia: `docs/schema.md` describe el modelo canónico (`dim_book.parquet`) y reglas de supervivencia.
   - `src/integrate_pipeline.py` contiene función `_canonical_key` y creación de `book_id` preferente por `isbn13`.

6) Normalización semántica — Estado: Parcial / Mejorable
   - Evidencia: `src/utils_quality.py` incluye `parse_date_to_iso`, `normalize_language`, `validate_language`, `normalize_currency` y `validate_currency`.
   - `integrate_pipeline` aplica `_apply_semantic_normalization` para fechas, idioma, moneda y precio. Algunos valores nulos/invalidos aparecen en `docs/quality_metrics.json` y pueden requerir mayor cobertura.

7) Integración, deduplicación y provenance — Estado: Implementado con trazabilidad básica
   - Evidencia: `standard/book_source_detail.parquet` contiene flags `valid` y `exclude_reason` y `provenance_by_field` en `dim`.
   - Reglas: prioridad `isbn13` > canonical key; supervivencia basada en score compuesto y `fuente_ganadora`.

8) Aserciones y métricas de calidad — Estado: Parcial / Mejorable
   - Evidencia: `docs/quality_metrics.json` generado y `src/utils_quality.compute_quality_metrics` usado.
   - Se añadieron aserciones soft en `integrate_pipeline`, y `ASSERT_UNIQUENESS_BOOK_ID` configurable.

9) Artefactos estándar (Parquet + docs) — Estado: Done
   - Evidencia: `standard/dim_book.parquet` y `standard/book_source_detail.parquet` escritos; `docs/schema.md` y `docs/quality_metrics.json` presentes.

Trazabilidad y logs

- `work/utils_logging.py` escribe logs en `work/logs/requests/` (requests por fuente), `work/logs/rules/` (reglas aplicadas) y `work/logs/runs/` (resumen de ejecución). El pipeline llama a `log_rule_jsonl` cuando existe el módulo `work.utils_logging`.
- `integrate_pipeline.py` registra en `work/logs` con `log_rule_jsonl` y asegura que la carpeta no quede vacía.

Notas de robustez y consideraciones técnicas

- Para evitar errores de pyarrow al escribir Parquet, `integrate_pipeline._safe_write_parquet` escribe en fichero temporal y reemplaza atómicamente, y normaliza columnas objetos serializando dicts/listas a JSON. Esto evita el ArrowTypeError causado por tipos mixtos.
- El pipeline no modifica archivos dentro de `landing/`.

Siguientes pasos recomendados (si se quiere perfeccionar para obtener máxima puntuación):

- Mejorar cobertura de normalización (reducir % de fechas/idiomas/monedas inválidas) y registrar por fila el motivo exacto de invalidación en `book_source_detail`.
- Añadir pruebas unitarias y un pequeño smoke test que ejecute el pipeline en modo dry-run y compare conteos previstos.
- Generar el PDF de entrega a partir de este README o del dossier solicitado.

Contacto y autoría

- Código principal: carpeta `src/`.
- Logs y trazabilidad: carpeta `work/logs/`.

---
Archivo actualizado automáticamente: README.md
