# Esquema canónico: `dim_book.parquet`

Este documento describe el modelo canónico generado por `src/integrate_pipeline.py`, las reglas de mapeo/normalización y las reglas de supervivencia y deduplicación que se aplican entre fuentes (`goodreads`, `google_books`). Está adaptado al código y a las convenciones de este repo.

## 1. Resumen rápido
- Formato: Parquet (Apache Arrow / pyarrow)
- Convenciones:
  - Nombres de columnas en snake_case (ej.: `titulo`, `autor_principal`, `isbn13`).
  - Listas: preferir listas nativas en Parquet; si se serializan para compatibilidad, usar `;` como separador.
  - Fechas: ISO-8601 `YYYY-MM-DD`. Si falta día/mes se rellena con `01` y se marca `fecha_publicacion_parcial = true`.
  - Idioma: BCP-47 (`en`, `es-ES`, `pt-BR`).
  - Moneda: ISO-4217 (`EUR`, `USD`).

---

## 2. Definición de campos (tipos, nullability, formato, ejemplo, reglas)
A continuación se listan los campos finales más relevantes de `dim_book.parquet` y las reglas asociadas, con ejemplos compatibles con el código del repo.

| Campo | Tipo | Nullable | Formato / ejemplo | Reglas / notas |
|---|---:|:---:|---|---|
| `book_id` | string | False | `9781449374273` | Identificador canónico. Preferir `isbn13` válido (normalizado). Si no hay `isbn13` válido, usar `canonical_key` (SHA1) — ver sección dedupe. Siempre presente en el `dim` final.
| `titulo` | string | False | `Data Science from Scratch` | Debe existir en ≥90% filas. Normalizar whitespace y strip. Priority: Google Books > Goodreads (si GB tiene valor no nulo se usa).
| `titulo_normalizado` | string | True | `data science from scratch` | Usado para matching / dedupe (parte principal del título sin subtítulos).
| `autor_principal` | string | True | `Joel Grus` | Priority: GB > GR. Si GB no tiene autor, usar Goodreads.
| `autores` | list[string] | True | `['Joel Grus','...']` | Mantener lista nativa cuando sea posible; si no, serializar con `;`.
| `isbn10` | string | True | `1449374271` | Preservar si existe. Si solo existe isbn10 y es válido, intentar convertir a isbn13 (try_normalize_isbn) y usar isbn13 resultante.
| `isbn13` | string | False* | `9781449374273` | Normalizar y validar con `utils_isbn.try_normalize_isbn`. Se prefiere isbn13 válido como clave primaria. *Si no hay isbn13 válido, se genera `book_id` vía `canonical_key` pero por política del repo los registros sin isbn13 pueden ser excluidos según aserciones configurables.
| `editorial` | string | True | `O'Reilly Media` | Fuente preferente: Google Books. Solo GB aporta editorial fiable.
| `fecha_publicacion` | string | True | `2019-04-14` | ISO `YYYY-MM-DD` (si parcial, rellenar mes/día con '01' y marcar `fecha_publicacion_parcial=true`). Parseo con `parse_date_to_iso` (src/utils_quality.py).
| `fecha_publicacion_parcial` | bool | False | True/False | Marca si la fecha es parcial (solo año o año-mes).
| `fecha_publicacion_valida` | bool | False | True/False | Indica si la fecha pudo parsearse a ISO.
| `idioma` | string | True | `en` | Normalizar con `normalize_language`; flag `idioma_valido`.
| `categoria` | list[string] | True | `['Data Science','Machine Learning']` | Tomada de Google Books cuando esté disponible.
| `paginas` | Int64 | True | `432` | Número de páginas si existe.
| `formato` | string | True | `BOOK` | Ej.: BOOK, EBOOK.
| `precio` | float | True | `27.99` | Precio numérico ≥ 0. Extraído preferentemente de Google Books (`price_amount`).
| `moneda` | string | True | `EUR` | Código ISO-4217 normalizado con `normalize_currency`. Flag `moneda_valida`.
| `provenance` | string (JSON) | True | `{"titulo":"google_books","isbn13":"google_books"}` | JSON que mapea campo → fuente que aportó el valor final.
| `ts_ultima_actualizacion` | string | False | `2025-11-18T22:28:41+00:00` | Timestamp ISO UTC de la última actualización.

> Nota: el `book_source_detail.parquet` contiene las filas originales de `landing/` con columnas adicionales: `source_name`, `source_file`, `row_number`, `book_id_candidato`, `isbn13_valido`, `ts_ingest`, y campos originales (posibles sufijos `_gr/_gb`) para trazabilidad.

---

## 3. Prioridades y fuentes (campo → fuente primaria / secundaria)
La integración aplica prioridades fijas por campo (configurables en el código si se requiere). Resumen:

| Campo | Fuente primaria | Fuente secundaria | Regla |
|---|---|---|---|
| `titulo` | Google Books | Goodreads | Usar GB si existe valor no nulo; si no, usar GR.
| `autores` | Google Books | Goodreads | Unir listas, preferir orden GB.
| `isbn13` | Goodreads / Google Books | — | Validación obligatoria; preferir ISBN-13 válido provisto por cualquiera de las fuentes.
| `editorial` | Google Books | — | Solo GB.
| `fecha_publicacion` | Google Books | — | Fecha original preferida de GB; si falta, usar fecha extraída del book page (si `--fetch-details`).
| `idioma` | Google Books | — | Usar código BCP-47 proporcionado por GB.
| `categoria` | Google Books | — | Lista desde GB.
| `precio`,`moneda` | Google Books | — | Tomar desde GB (`saleInfo`) si existe.

---

## 4. Reglas de deduplicación
- Clave principal: `isbn13` (valor normalizado y validado). Cuando `isbn13_valido==True`, agrupar por `isbn13`.
- Si no existe `isbn13` válido, se construye `canonical_key = sha1(titulo_normalizado | autor_principal | editorial | año)` y se agrupa por esa clave como fallback.
- Implementación práctica (en `src/integrate_pipeline.py`):
  - `gr` (Goodreads) y `gb` (Google Books) se normalizan y se generan campos `_match_title`, `_match_author` y `_match_key` usados para emparejamiento exacto.
  - Se realiza merge por `isbn13` primero y por `_match_key` luego; filas de `gb` no emparejadas se conservan como `gb_only`.
- Si existen múltiples filas con el mismo `isbn13`:
  - Agrupar: `.groupby('isbn13')`.
  - Resolver conflictos aplicando reglas de supervivencia (siguiente sección).
- Política: registros sin `isbn13` válido pueden ser descartados o incluidos solo en `book_source_detail` según las aserciones configuradas (ver `ASSERT_UNIQUENESS_BOOK_ID` en `src/integrate_pipeline.py`).

---

## 5. Reglas de supervivencia (merge / survival rules)
### Lógica general por grupo (mismo `isbn13`)
1. Preferencia de fuente: Google Books > Goodreads. Si GB tiene un valor no nulo para un campo, se usa ese valor.
2. Criterio de completitud: si ambos aportan valor no nulo para un campo de texto, elegir la cadena más larga (mayor contenido). Si hay empate, preferir GB.
3. Para listas (autores, categorias): unir valores, desduplicar preservando orden (preferir orden GB si GB presente).
4. Para `isbn10` cuando `isbn13` falta: intentar convertir `isbn10` → `isbn13` con `try_normalize_isbn`; si la conversión produce `isbn13_valido`, usarlo.
5. `price_amount` y `moneda`: tomar desde GB; validar `price_amount >= 0` y `moneda` con `normalize_currency`.

### Requisitos y aserciones mínimas (rúbrica)
- ≥90% de títulos (`titulo`) deben estar presentes (no nulos) en `dim_book`.
- `isbn13` debe ser único en la tabla final (`duplicados_isbn13 = 0`).
- `price_amount` cuando existe debe ser ≥ 0.

### Survival rules explícitas (mapa rápido)
- `titulo`: GB > GR
- `autores`: GB > GR (unir listas)
- `isbn10`: GB > GR (pero se normaliza a isbn13 cuando procede)
- `precio`/`moneda`: solo GB
- `editorial` / `fecha_publicacion` / `idioma` / `categoria`: solo GB

---

## 6. Provenance y trazabilidad
- `provenance` (campo JSON por fila): para cada campo final registrar la fuente que aportó ese valor (`goodreads`, `google_books`, `merged`, `derived:isbn10->isbn13`).
- `book_source_detail.parquet` contiene la fila original de cada fuente con `row_number`, `source_name`, `source_file`, `book_id_candidato`, flags de validación y snapshots de campos originales. Este fichero es la evidencia primaria para auditoría.
- Decisiones relevantes (ambigüedad de candidatos, asignaciones por `_match_key`, conversiones ISBN) se registran con `log_rule_jsonl` en `work/logs/rules/integrate_rules_<fecha>.jsonl`.

---

## 7. Ejemplo (fila) — visualización CSV
A modo de ejemplo, una fila representativa del `dim_book` (campos clave, serializados en CSV) podría verse así:

```
"Data Science for Business: What You Need to Know about Data Mining and Data-Analytic Thinking",Foster Provost,9781449374273,9781449374273,2013-01-01,en,"['Big data']",432,BOOK,,27.99,USD,"{\"titulo\":\"google_books\",\"isbn13\":\"google_books\"}",2025-11-18T22:28:41+00:00
```

(Esta línea es un ejemplo simplificado: `provenance` es JSON y `autores`/`categoria` pueden ser listas nativas en Parquet.)

---

## 8. Notas operativas y evolución
- No eliminar columnas `all-null` de forma automática: la eliminación debe ser deliberada y documentada.
- Mantener `docs/schema.md` sincronizado con `src/integrate_pipeline.py` y `src/utils_quality.py` cuando se cambien reglas de normalización.
- Para depurar problemas de matching y provenance usar `scripts/diagnose_parquet.py` y revisar `work/logs/rules`.

---

## 9. Preguntas frecuentes rápidas
- ¿Qué pasa si Google Books y Goodreads tienen ISBN distinto para la misma obra? Se prioriza el `isbn13` válido (si coinciden se agrupan). Si hay conflicto evidente y ambos son válidos pero distintos, se registra la ambigüedad en `work/logs/rules` y se conserva la fila con mayor completitud (configurable).
- ¿Se usan heurísticas difusas para emparejar títulos? No por defecto. Solo matching exacto sobre `_match_key` o reglas de título único; heurísticas difusas se pueden activar explícitamente si el usuario lo autoriza.

---

