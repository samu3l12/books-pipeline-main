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

Nota sobre extracción de ISBN desde páginas individuales

- Control de extracción de ISBN: puedes habilitar/deshabilitar que el scraper visite la página individual del libro para intentar extraer ISBNs adicionales y otros campos (fecha, precio). Esto se controla con la variable de entorno `GOODREADS_USE_ISBN`:
  - `GOODREADS_USE_ISBN=1` (valor por defecto en `.env.example`): el scraper intentará extraer ISBNs desde la página individual cuando `--fetch-details` esté activo.
  - `GOODREADS_USE_ISBN=0`: aunque `--fetch-details` esté activo, el scraper no intentará extraer ISBNs desde la página del libro; seguirá extrayendo otros campos si están disponibles en la página de resultados.
  - Nota: `--fetch-details` sigue controlando si se visitan páginas individuales para extraer `pub_date` y `price`; `GOODREADS_USE_ISBN` solamente condiciona si dentro de esa visita se intentan extraer ISBN adicionales.

- Enriquecer (Google Books → CSV + candidatos):
  - Añadir variable `GOOGLE_BOOKS_API_KEY` en `.env` o en variables de entorno. El script de enriquecimiento requiere la clave; si no está presente lanzará un error.
  - python .\src\enrich_googlebooks.py
  - Salida: `landing/googlebooks_books.csv` (UTF-8, separador coma) y `landing/googlebooks_candidates.json` (trazabilidad de candidatos)

- Integrar (JSON+CSV → Parquet + métricas):
  - python .\src\integrate_pipeline.py
  - Salida: `standard/dim_book.parquet`, `standard/book_source_detail.parquet`, `docs/quality_metrics.json`

Breve nota sobre logging y trazabilidad

- Los scripts intentan usar `work.utils_logging` para escribir trazabilidad en `work/logs/requests/`, `work/logs/rules/` y `work/logs/runs/`. Si `work` no está disponible, el pipeline sigue funcionando con fallbacks no operativos (no se lanzan excepciones).

Cómo funciona el scraping de Goodreads (resumen técnico)

- URL base: `https://www.goodreads.com/search`.
- Peticiones: el scraper construye la URL con parámetros `q` (consulta) y `page` y realiza un GET sobre la página de resultados.
- User-Agent: rota entre varios UA comunes y añade un sufijo `(compatible; BooksPipeline/1.0)`.
- Retries/backoff: usa una `requests.Session` con `Retry(total=5, backoff_factor=0.6)` para manejar 429/5xx y reintentos automáticos.
- Pausas éticas: tras cada página se aplica una pausa aleatoria mínima `min_pause_s` (por defecto 0.8s) más un componente aleatorio (hasta ~0.8s). En el código por defecto se scrapean hasta `max_pages=3` y `max_records=15`.
- Selectores principales (documentados en metadatos):
  - Título: `a.bookTitle span`
  - Autor: `a.authorName span`
  - Rating: `span.minirating`
  - URL libro: `a.bookTitle`
- Extracción de ISBN: se aplica una heurística sobre el texto del `tr` y sobre los enlaces del resultado para intentar extraer `isbn10`/`isbn13` usando `src/utils_isbn.py`. Además, si `--fetch-details` está activo y `GOODREADS_USE_ISBN=1`, el scraper visitará la página individual y buscará ISBNs adicionales en el HTML de la página de libro.
- Deduplificación: después de recolectar registros se deduplica por (title.lower(), author.lower()) y se filtran registros sin título o autor.
- Metadatos: el JSON de salida incluye `metadata` con `selectors`, `user_agent`, `fetched_at`, `record_count`, `notes.pauses_seconds` y `retry_backoff`.

Cómo funciona `enrich_googlebooks.py` (resumen técnico)

- Requisito: necesita la variable de entorno `GOOGLE_BOOKS_API_KEY`. Si no existe, la función `enrich` rompe con RuntimeError.
- Endpoint usado: `https://www.googleapis.com/books/v1/volumes`.
- Construcción de la query `q`:
  - Si existe un `isbn13` válido, se envía `q=isbn:...` (búsqueda por ISBN, prioritaria).
  - Si no hay ISBN, se construye una query concatenando campos (título, `inauthor:`, `inpublisher:`, `subject:`, `lccn:`, `oclc:`) y reemplazando espacios por `+`. Se solicita `maxResults=5` para obtener varios candidatos.
- Retries/backoff: la sesión de requests incluye `Retry(total=4, backoff_factor=0.5)` y se aplica una pausa entre llamadas configurable con `GOOGLE_BOOKS_RATE_LIMIT` (por defecto 0.6s).
- Selección del mejor candidato: si la API devuelve varios `items`, se aplica un scoring que combina:
  - Coincidencia de ISBN (match exacto suma peso grande).
  - Similitud de título (Levenshtein normalizado + token overlap).
  - Solapamiento de autor (token overlap).
  - Empates: preferir item con ISBN-13; si persiste empate, preferir fecha de publicación más reciente.
- Extracción de campos: el código mapea campos de `volumeInfo` y `saleInfo` a columnas del CSV: `gb_id, title, subtitle, authors, publisher, pub_date, language, categories, isbn13, isbn10, price_amount, price_currency, paginas, formato`.
- Normalización: moneda se normaliza a ISO-4217 cuando es posible; precios se intentan parsear a float y se escriben con punto decimal; listas (authors, categories) se serializan con `;`.
- Trazabilidad de candidatos: para cada registro de entrada se escribe un elemento en `googlebooks_candidates.json` con `rec_id`, `input_index`, `csv_row_number`, `title`, `author`, `isbn13_input`, `candidates` (lista con score) y `best_score`.
- Salida: `landing/googlebooks_books.csv` (solo filas para las entradas con match) y `landing/googlebooks_candidates.json`.

Decisiones clave y mapeo contra la rúbrica

A continuación se documenta cada criterio de la rúbrica con la evidencia correspondiente (archivos y ubicaciones):

1) Estructura del repositorio
   - Evidencia: listado de ficheros en la raíz; `landing/`, `standard/`, `docs/` y `src/` presentes.

2) Scraping Goodreads (JSON válido) 
   - Evidencia: `landing/goodreads_books.json` contiene 15 registros (metadata.record_count = 15) y campos: title/titulo, author/autor_principal, rating, ratings_count, book_url, isbn10/isbn13 (cuando están). Metadatos con `selectors`, `user_agent`, `fetched_at` y `pauses_seconds`.

3) Metadatos de landing y ética de scraping — Estado: Done
   - Evidencia: `landing/goodreads_books.json` -> `metadata.selectors`, `metadata.user_agent`, `metadata.fetched_at`, `metadata.record_count`, `metadata.notes.pauses_seconds`.
   - Pausas y backoff documentadas en `src/scrape_goodreads.py`.

4) Enriquecimiento Google Books (CSV válido) — Estado: Done (parcial normalización)
   - Evidencia: `landing/googlebooks_books.csv` con cabecera UTF-8 y campos: gb_id,title,subtitle,authors,publisher,pub_date,language,categories,isbn13,isbn10,price_amount,price_currency,paginas,formato.
   - `landing/googlebooks_candidates.json` contiene candidatos con score; `src/enrich_googlebooks.py` solicita `maxResults=5` y genera scoring.

5) Modelo canónico y mapa de campos
   - Evidencia: `docs/schema.md` describe el modelo canónico (`dim_book.parquet`) y reglas de supervivencia.
   - `src/integrate_pipeline.py` contiene función `_canonical_key` y creación de `book_id` preferente por `isbn13`.

6) Normalización semántica — Estado:
   - Evidencia: `src/utils_quality.py` incluye `parse_date_to_iso`, `normalize_language`, `validate_language`, `normalize_currency` y `validate_currency`.
   - `integrate_pipeline` aplica `_apply_semantic_normalization` para fechas, idioma, moneda y precio. Algunos valores nulos/invalidos aparecen en `docs/quality_metrics.json` y pueden requerir mayor cobertura.

7) Integración, deduplicación y provenance — Estado: Implementado con trazabilidad básica
   - Evidencia: `standard/book_source_detail.parquet` contiene flags `valid` y `exclude_reason` y `provenance_by_field` en `dim`.
   - Reglas: prioridad `isbn13` > canonical key; supervivencia basada en score compuesto y `fuente_ganadora`.

8) Aserciones y métricas de calidad
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


Contacto y autoría

- Código principal: carpeta `src/`.
- Logs y trazabilidad: carpeta `work/logs/`.

---
