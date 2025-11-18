# Script de inspección rápida que muestra: archivos en `landing/` (nº registros),
# artefactos en `standard/` (conteos y muestras) y métricas en `docs/`.
# Útil para debugging manual y comprobaciones rápidas antes de generar la entrega.

import json
from pathlib import Path
import pandas as pd
import csv

root = Path('.')
landing = root / 'landing'
standard = root / 'standard'
docs = root / 'docs'

def read_csv_count(p):
    if not p.exists():
        return 0
    with p.open(encoding='utf-8') as f:
        r = csv.reader(f)
        try:
            next(r)
        except StopIteration:
            return 0
        return sum(1 for _ in r)

print('--- Landing files ---')
for f in sorted(landing.iterdir()):
    if f.suffix.lower() in ('.json', '.csv'):
        if f.suffix.lower() == '.csv':
            print(f.name, 'rows:', read_csv_count(f))
        else:
            try:
                j = json.loads(f.read_text(encoding='utf-8'))
                if isinstance(j, dict) and 'records' in j:
                    print(f.name, 'records:', len(j.get('records', [])))
                else:
                    print(f.name, 'size:', f.stat().st_size)
            except Exception:
                print(f.name, 'size:', f.stat().st_size)

print('\n--- Standard files ---')
if (standard / 'dim_book.parquet').exists():
    d = pd.read_parquet(standard / 'dim_book.parquet')
    print('dim_book.parquet rows:', len(d))
    print('dim_book columns:', list(d.columns))
    print('dim_book sample (first 3):')
    print(d.head(3).to_dict(orient='records'))
else:
    print('dim_book.parquet missing')

if (standard / 'book_source_detail.parquet').exists():
    b = pd.read_parquet(standard / 'book_source_detail.parquet')
    print('\nbook_source_detail.parquet rows:', len(b))
    print('book_source_detail columns:', list(b.columns))
    print('book_source_detail sample (first 3, gb_candidate_scores truncated):')
    s = b.head(3).to_dict(orient='records')
    for rec in s:
        if 'gb_candidate_scores' in rec and isinstance(rec['gb_candidate_scores'], list):
            rec['gb_candidate_scores'] = rec['gb_candidate_scores'][:2]
        print(rec)
else:
    print('book_source_detail.parquet missing')

print('\n--- Docs/metrics ---')
if (docs / 'quality_metrics.json').exists():
    print('quality_metrics.json:')
    print((docs / 'quality_metrics.json').read_text(encoding='utf-8'))
else:
    print('quality_metrics.json missing')
