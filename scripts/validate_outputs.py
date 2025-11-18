"""
Valida los artefactos generados (`docs/quality_metrics.json`, `standard/dim_book.parquet`) contra umbrales recomendados.
Uso: python scripts/validate_outputs.py
Salida: imprime PASS/FAIL por cada métrica y código de salida 0 si todo OK, 2 si falla.

Palabras clave: VALIDATION_SCRIPT, QUALITY_THRESHOLDS
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# Asegurar que 'src' está en sys.path para poder importar utils desde scripts
sys.path.insert(0, str(ROOT / 'src'))

try:
    from utils_quality import get_quality_thresholds
except Exception as e:
    print(f"ERROR importando utils_quality: {e}")
    sys.exit(1)

DOCS = ROOT / "docs"
STANDARD = ROOT / "standard"


def load_metrics(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def compare_metrics(metrics: dict, thresholds: dict) -> bool:
    ok = True
    for k, thr in thresholds.items():
        val = metrics.get(k)
        if val is None:
            print(f"WARN: métrica {k} no encontrada en metrics")
            ok = False
            continue
        # duplicados esperan exact match 0
        if isinstance(thr, int) and thr == 0:
            if val != 0:
                print(f"FAIL: {k} = {val} (esperado {thr})")
                ok = False
            else:
                print(f"PASS: {k} = {val}")
        else:
            try:
                if float(val) >= float(thr):
                    print(f"PASS: {k} = {val} >= {thr}")
                else:
                    print(f"FAIL: {k} = {val} < {thr}")
                    ok = False
            except Exception:
                print(f"WARN: no se pudo comparar {k} (valor: {val})")
                ok = False
    return ok


def main():
    q_path = DOCS / 'quality_metrics.json'
    if not q_path.exists():
        print(f"ERROR: {q_path} no encontrado. Ejecuta el pipeline primero.")
        sys.exit(1)
    metrics = load_metrics(q_path)
    thresholds = get_quality_thresholds()
    ok = compare_metrics(metrics, thresholds)
    if ok:
        print("VALIDACION: OK — Todas las métricas cumplen los umbrales para 'Excelente'.")
        sys.exit(0)
    else:
        print("VALIDACION: FAILED — Algunas métricas no cumplen los umbrales.")
        sys.exit(2)


if __name__ == '__main__':
    main()
