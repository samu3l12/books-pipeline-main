# books-pipeline

Mini-pipeline para extracción (Goodreads), enriquecimiento (Google Books API) e integración canónica con controles de calidad y documentación según rúbrica SBDxx.

## Estructura del repositorio

```
books-pipeline/
├─ README.md
├─ requirements.txt
├─ .env.example
├─ landing/
│  ├─ goodreads_books.json
│  └─ googlebooks_books.csv
├─ standard/
│  ├─ dim_book.parquet
│  └─ book_source_detail.parquet
├─ docs/
│  ├─ schema.md
│  └─ quality_metrics.json
└─ src/
   ├─ scrape_goodreads.py
   ├─ enrich_googlebooks.py
   ├─ integrate_pipeline.py
   ├─ utils_quality.py
   └─ utils_isbn.py
```

## Objetivo
Flujo mínimo de datos de libros: Scraping público → Enriquecimiento externo → Integración estandarizada. Se genera un modelo canónico y artefactos de calidad sin alterar archivos en `landing/`.

## Configuración
Variables en `.env` (ver `.env.example`):
- GOOGLE_BOOKS_API_KEY: clave de Google Books API.
- GOOGLE_BOOKS_COUNTRY: código de país para resultados (opcional, ej. ES).
- SCRAPER_API_URL / SCRAPER_API_KEY: proveedor opcional de scraping/proxy para evitar bloqueos IP. Soportado: URL con `{url}` o parámetro `url=...`.

## Pasos del pipeline
1. Scraping Goodreads (`scrape_goodreads.py`): ejecuta búsqueda pública y extrae campos básicos (title, author, rating, ratings_count, book_url, isbn10, isbn13). Reintentos con backoff, UA rotatorio y pausas de cortesía. Genera `landing/goodreads_books.json` con metadatos.
2. Enriquecimiento Google Books (`enrich_googlebooks.py`): para cada registro del JSON busca por ISBN-13 (prioritario) o combinación título+autor. Captura gb_id, title, subtitle, authors, publisher, pub_date, language, categories, isbn13, isbn10, price_amount, price_currency. Resulta en `landing/googlebooks_books.csv` UTF-8 con cabecera.
3. Integración (`integrate_pipeline.py`): lee JSON y CSV, normaliza formatos (fechas ISO-8601, idioma BCP-47, moneda ISO-4217), construye ID (isbn13 válido o hash estable), deduplica, determina supervivencia, produce `standard/dim_book.parquet`, `standard/book_source_detail.parquet` y métricas en `docs/quality_metrics.json`; documentación ampliada en `docs/schema.md`.

## Ejecución local
```powershell
pip install -r requirements.txt
# Scraping (ejemplo consulta "data science")
python .\src\scrape_goodreads.py --query "data science" --max-records 15 --max-pages 3 --timeout 15
# Enriquecimiento (con API si GOOGLE_BOOKS_API_KEY presente en .env)
python .\src\enrich_googlebooks.py --input .\landing\goodreads_books.json --output .\landing\googlebooks_books.csv
# Integración
python .\src\integrate_pipeline.py
```
Notas:
- Para usar un proveedor de scraping, definir SCRAPER_API_URL y SCRAPER_API_KEY en `.env`.
- El scraper añade pausas aleatorias (≈0.8–1.6s) y reintentos (HTTP 429/5xx) para reducir riesgo de bloqueo.

## Selectores y metadatos de scraping
- URL base: `https://www.goodreads.com/search`
- Selectores CSS: título `a.bookTitle span`, autor `a.authorName span`, rating recuento `span.minirating`, enlace `a.bookTitle`
- User-Agent rotatorio + Referer `https://www.google.com/`
- Pausas: ≥0.8s + jitter aleatorio ≤0.8s
- Campo `metadata.record_count` refleja libros válidos (título+autor no nulos)

### Resultado actual (Tarea 1 - Scraping)
- Archivo generado: `landing/goodreads_books.json`
- Fecha de ejecución (fetched_at): 2025-11-16T00:50:08+00:00
- User-Agent usado (ejemplo): `Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0 Safari/537.36 (compatible; BooksPipeline/1.0)`
- Número de registros válidos escritos: 15
- Observaciones: la extracción heurística de ISBN no encontró valores en la mayoría de registros (campos `isbn10`/`isbn13` aparecen como `null` en el JSON). Esto es esperado para resultados de búsqueda en la página de resultados; el enriquecimiento con Google Books debe resolver la mayoría de ISBNs cuando exista correspondencia por título+autor.

## Mapeo de campos (landing → canónico)
| Fuente | Entrada | Canónico |
|--------|---------|----------|
| Goodreads JSON | title | titulo |
| Goodreads JSON | author | autor_principal |
| Goodreads JSON | isbn13 | isbn13 |
| Goodreads JSON | isbn10 | isbn10 |
| Google Books CSV | title | titulo (fallback si vacío en Goodreads) |
| Google Books CSV | authors | autores (serializado ';') |
| Google Books CSV | publisher | editorial |
| Google Books CSV | pub_date | fecha_publicacion (normalizado a YYYY-MM-DD si parcial) |
| Google Books CSV | language | idioma (normalizado BCP-47) |
| Google Books CSV | categories | categoria (serializado ';') |
| Google Books CSV | price_amount | precio |
| Google Books CSV | price_currency | moneda |

## Modelo canónico (`dim_book.parquet`)
Campos principales:
- book_id: isbn13 válido; en ausencia, hash SHA1 de (titulo, autor_principal, editorial, anio_publicacion)
- titulo / titulo_normalizado (lower + trim espacios)
- autor_principal / autores
- editorial
- fecha_publicacion (ISO completa o derivada de parcial año/mes)
- anio_publicacion (derivado)
- idioma (BCP-47 normalizado)
- isbn10 / isbn13 (strings)
- categoria (lista serializada)
- precio (decimal) / moneda (ISO-4217)
- fuente_ganadora (regla supervivencia)
- ts_ultima_actualizacion (UTC ISO)
- flags validación: isbn13_valido, idioma_valido, moneda_valida, fecha_publicacion_valida

## Reglas de deduplicación y supervivencia
1. Clave primaria: isbn13 válido. Si ausente, clave provisional hash.
2. Prioridad de supervivencia: registro con mayor completitud (presencia de editorial, fecha_publicacion, autores) favorece Google Books.
3. Merge de campos: selección de primer valor no nulo preferente Goodreads→GoogleBooks para título y autor principal; unión no duplicada para listas (autores/categorias) en futuras extensiones.
4. Proveniencia resumida en `fuente_ganadora`; detalle completo por registro original en `book_source_detail.parquet`.

## Normalización
- Fechas: se intenta parsear `%Y-%m-%d`, `%Y-%m`, `%Y`; meses o años parciales se materializan al primer día (marca parcial en lógica interna).
- Idioma: lower para subtags y regiones upper (ej. "en-US"). Validación con patrón BCP-47 básico.
- Moneda: validación ISO-4217 restringida a conjunto incluido (USD, EUR, etc.).
- ISBN: validación checksum; ISBN-10 convertible a ISBN-13 se maneja en enriquecimiento.
- Whitespace: colapsado a un espacio; snake_case consistente en columnas.

## Métricas de calidad (`quality_metrics.json`)
Incluye conteo total, nulos por campo clave, porcentaje de fechas, idiomas, monedas e ISBN13 válidos y completitud promedio (% de campos presentes por fila sobre set clave). Extensible a duplicados y distribución por fuente.

## Dependencias
- Python ≥3.10
- requests, beautifulsoup4, lxml, pandas, pyarrow, numpy, python-dotenv

## Decisiones clave
- No se modifican archivos en `landing/` durante integración.
- ISBN forzado a tipo string para evitar problemas Arrow/Parquet.
- Flags booleanos calculados en ambos niveles (dim y detalle) para trazabilidad de validaciones.
- API de Google Books consultada máximo una vez por registro (maxResults=1) para minimizar latencia.
- Soporte opcional de proveedor de scraping mediante variables de entorno.

## Limitaciones y futuras mejoras
- Campos `paginas`, `formato` aún no implementados.
- Proveniencia por campo granular (columna origen) pendiente.
- Métricas duplicados no implementadas.
- Control de rangos precio/fecha aún básico.

## Exportación para entrega
README puede exportarse a PDF e incluir capturas si procede. Estructura satisface rúbrica (Scraping, Enriquecimiento, Integración, Calidad, Documentación).

---
Documentación complementaria y detalles de tipos en `docs/schema.md`.
