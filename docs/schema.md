# Esquema canónico: dim_book.parquet

Descripción mínima del esquema canónico generado por `integrate_pipeline.py`.

Formato: Parquet (columnas en snake_case). Ejemplo: `standard/dim_book.parquet`.

Campos principales

- book_id (string, not null): Identificador preferente — isbn13 válido; si no existe, SHA1 de (titulo, autor_principal, editorial, anio_publicacion). Ejemplo: `9780131103627` o `3a1f...`.
- titulo (string, nullable): Título principal.
- titulo_normalizado (string, nullable): Título en minúsculas y whitespace normalizado.
- autor_principal (string, nullable): Autor principal extraído o derivado.
- autores (string, nullable): Lista serializada con `;` (p. ej. `Autor A;Autor B`).
- editorial (string, nullable): Editorial/proveedor.
- anio_publicacion (int, nullable): Año derivado de `fecha_publicacion`.
- fecha_publicacion (string, nullable): Fecha en ISO-8601 `YYYY-MM-DD` (parciales materializadas al primer día del periodo).
- idioma (string, nullable): Código BCP-47 normalizado (ej. `es`, `en-US`).
- isbn10 (string, nullable): ISBN-10 normalizado o null.
- isbn13 (string, nullable): ISBN-13 normalizado o null.
- categoria (string, nullable): Lista serializada con `;`.
- precio (float, nullable): Monto de precio (decimal) con separador punto.
- moneda (string, nullable): Código ISO-4217 (ej. `USD`, `EUR`).
- fuente_ganadora (string, nullable): Fuente elegida por regla de supervivencia (`google_books` o `goodreads`).
- ts_ultima_actualizacion (string, not null): Timestamp UTC ISO del procesamiento (ej. `2025-11-16T12:34:56+00:00`).

Flags de validación (boolean):
- isbn13_valido
- idioma_valido
- moneda_valida
- fecha_publicacion_valida

Notas de normalización
- Fechas parciales (`YYYY` o `YYYY-MM`) se materializan al primer día del periodo (`YYYY-01-01` o `YYYY-MM-01`) y se consideran `fecha_publicacion_valida=True` pero parciales internamente.
- Idioma: normalizado con `normalize_language` y validado con `validate_language`.
- Moneda: validada contra un conjunto ISO-4217 básico.
- Listas: serializadas como `;` para compatibilidad con escritura inicial Parquet.

Reglas de deduplicación y supervivencia
- Clave primaria: isbn13 válido.
- Fallback: SHA1 de `(titulo, autor_principal, editorial, anio_publicacion)`.
- Supervivencia: preferir registros con mayor completitud y/o con `editorial`, `fecha_publicacion` y `autores` (tiende a preferir `google_books`).
- Para listas (autores, categorias) se unifican y eliminan duplicados al serializar.

Provenance
- `book_source_detail.parquet` contiene por fila el registro original y flags de validación para trazar por qué se eligió cada `book_id` y `fuente_ganadora`.

Este documento puede ampliarse con ejemplos por campo, rangos aceptables y reglas de mapeo detalladas.

