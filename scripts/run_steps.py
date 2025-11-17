"""\nRunner simple para ejecutar los pasos del pipeline uno a uno.
Uso:
    python src/run_steps.py --step scrape
    python src/run_steps.py --step enrich
    python src/run_steps.py --step integrate

Permite iterar y comprobar cada etapa manualmente.
"""
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

STEPS = {
    "scrape": ["python", str(ROOT / "src" / "scrape_goodreads.py")],
    "enrich": ["python", str(ROOT / "src" / "enrich_googlebooks.py"), "--input", str(ROOT / "landing" / "goodreads_books.json"), "--output", str(ROOT / "landing" / "googlebooks_books.csv")],
    "integrate": ["python", str(ROOT / "src" / "integrate_pipeline.py")],
}


def run_step(step: str) -> int:
    if step not in STEPS:
        raise SystemExit(f"Paso desconocido: {step}. Opciones: {', '.join(STEPS.keys())}")
    cmd = STEPS[step]
    print(f"Ejecutando paso: {step} -> {' '.join(cmd)}")
    proc = subprocess.run(cmd, check=False)
    return proc.returncode


def main() -> None:
    parser = argparse.ArgumentParser(description="Run pipeline steps one by one")
    parser.add_argument("--step", required=True, choices=list(STEPS.keys()))
    args = parser.parse_args()
    code = run_step(args.step)
    print(f"Paso {args.step} finalizado con código {code}")


if __name__ == "__main__":
    main()

