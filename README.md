# books-pipeline (Mini-pipeline de libros)

Resumen

Proyecto con un flujo: Extracción → Enriquecimiento → Integración destinado a obtener una muestra de libros desde Goodreads, enriquecer con la Google Books API y producir artefactos canónicos (.parquet) con controles de calidad y trazabilidad.

---

Índice rápido

- Requisitos
- Instalación
- Uso (comandos)
- Variables de entorno importantes
- Cómo funciona (resumen visual)
- Ficheros generados y trazabilidad
- Solución de problemas rápida

---

Requisitos

- Python 3.10+ (probado con 3.14)
- pip
- Acceso a internet para llamadas a Goodreads y Google Books (si se usa)

Dependencias (ver `requirements.txt`)

- requests
- beautifulsoup4
- lxml
- pandas
- pyarrow
- python-dotenv

Instalación (rápida)

PowerShell (Windows):

```powershell
# Crear y activar entorno virtual
python -m venv .venv
.\.venv\Scripts\Activate.ps1
# Instalar dependencias
pip install -r requirements.txt
```

Linux / macOS (bash):

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Preparar variables de entorno

- Copia el ejemplo de variables y edítalo según necesites:

```powershell
copy .env.example .env  # PowerShell
```

- Edita `.env` y añade tu `GOOGLE_BOOKS_API_KEY` si quieres usar el enriquecimiento.
- Variable nueva importante: `GOODREADS_USE_ISBN` (ver más abajo).

Uso — comandos principales

Nota: los ejemplos siguientes están pensados para PowerShell en Windows; en bash sustituye la sintaxis de variables de entorno si hace falta.

1) Scrape — extraer desde Goodreads → JSON

- Scrape básico (solo lista de resultados):

```powershell
python .\src\scrape_goodreads.py --query "data science" --max-records 15 --max-pages 3
```

- Scrape visitando la página individual de cada libro (extrae fechas/precios; opcionalmente ISBNs según la variable de entorno):

```powershell
# habilitar extracción de ISBN desde la página del libro (opcional)
$env:GOODREADS_USE_ISBN='1'
python .\src\scrape_goodreads.py --query "data science" --max-records 15 --max-pages 3 --fetch-details --detail-pause 1.0
```

- Si quieres visitar páginas individuales pero NO extraer ISBNs desde ellas:

```powershell
$env:GOODREADS_USE_ISBN='0'
python .\src\scrape_goodreads.py --query "data science" --max-records 15 --max-pages 3 --fetch-details --detail-pause 1.0
```

Salida: `landing/goodreads_books.json` (contiene `metadata` y `records`).

2) Enrich — Google Books → CSV (+ candidatos)

```powershell
# Asegúrate de tener GOOGLE_BOOKS_API_KEY en .env o en variables de entorno
python .\src\enrich_googlebooks.py --input .\landing\goodreads_books.json --output .\landing\googlebooks_books.csv
```

Salida: `landing/googlebooks_books.csv` y `landing/googlebooks_candidates.json`.

3) Integrar — JSON + CSV → Parquet + métricas

```powershell
python .\src\integrate_pipeline.py
```

Salida: `standard/dim_book.parquet`, `standard/book_source_detail.parquet`, `docs/quality_metrics.json`.

4) Validar métricas

```powershell
python .\scripts\validate_outputs.py
```

---

Variables de entorno importantes (resumen)

- `GOOGLE_BOOKS_API_KEY`: (requerido para `enrich_googlebooks.py`) clave de la API de Google Books.
- `GOODREADS_USE_ISBN` (nuevo): controla si, al visitar la página individual del libro (cuando `--fetch-details` está activo), se intentan extraer ISBNs desde el HTML de la página.
  - `GOODREADS_USE_ISBN=1` (por defecto): el scraper intentará extraer ISBNs desde la página del libro.
  - `GOODREADS_USE_ISBN=0`: no intentará extraer ISBNs desde la página del libro, aunque sí podrá extraer fecha/precio si `--fetch-details` está activo.

---

Cómo funciona (resumen visual)

1) Scrape (Goodreads)
- Construye URL: `https://www.goodreads.com/search?q=<query>&page=<n>`
- Parsea lista de resultados (selectores): título, autor, rating, enlace al libro
- Opcional: visita página individual (`--fetch-details`) para extraer fecha y precio y, si `GOODREADS_USE_ISBN=1`, también ISBNs
- Guarda `landing/goodreads_books.json` con `metadata` y `records`

2) Enrich (Google Books)
- Para cada registro intenta buscar por `isbn:...` si hay ISBN; si no, construye una query por título/autor
- Obtiene candidatos, calcula score y guarda el mejor en `landing/googlebooks_books.csv`
- Guarda `landing/googlebooks_candidates.json` con trazabilidad de candidatos

3) Integrate
- Carga `landing/*.json` y `landing/*.csv`
- Normaliza campos (fecha ISO, idioma BCP-47, moneda ISO-4217, precio float)
- Fusiona por `isbn13` y por clave de título+autor
- Genera `standard/dim_book.parquet` y `standard/book_source_detail.parquet`
- Calcula métricas y escribe `docs/quality_metrics.json`

---

Archivos generados 

- landing/goodreads_books.json — resultados del scraper con metadatos
- landing/googlebooks_books.csv — filas enriquecidas desde Google Books
- landing/googlebooks_candidates.json — candidatos evaluados por entrada
- standard/dim_book.parquet — tabla dimensional canónica
- standard/book_source_detail.parquet — detalle por fila de origen
- docs/quality_metrics.json — métricas de calidad y aserciones

Solución rápida de problemas

- Si `enrich_googlebooks.py` falla por la API: revisa `GOOGLE_BOOKS_API_KEY` en `.env`.
- Si `integrate_pipeline.py` falla al escribir Parquet en Windows: asegúrate de que no haya locks en los ficheros y prueba a borrar `standard/*.parquet` antes de reintentar.
- Si el scraper devuelve pocas fechas/monedas: ejecuta con `--fetch-details` (visita páginas) y aumenta `--detail-pause` para reducir riesgo de bloqueo.

---

