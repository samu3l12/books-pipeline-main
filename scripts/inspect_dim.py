import sys
from pathlib import Path
import pandas as pd

def read_parquet_safe(p: Path, name: str):
    if not p.exists():
        print(f"{name} no encontrado en {p}")
        return None
    try:
        return pd.read_parquet(p)
    except Exception as e:
        print(f"Error leyendo {name} desde {p}: {e}")
        return None

def main(directory: str = "standard"):
    # Resolver ruta relativa al proyecto (script está en `scripts/`)
    base = (Path(__file__).resolve().parent.parent / directory) if not Path(directory).is_absolute() else Path(directory)
    p = base / "dim_book.parquet"
    d = read_parquet_safe(p, "dim")
    if d is None:
        return
    print("dim rows, cols:", d.shape)
    print("dim columns sample:", d.columns.tolist()[:40])
    non_null = d.notna().sum(axis=1).sort_values(ascending=False)
    print("top non-null counts per row sample idx:", non_null.head(5).to_dict())

    p2 = base / "book_source_detail.parquet"
    dd = read_parquet_safe(p2, "detail")
    if dd is None:
        return
    print("detail rows, cols:", dd.shape)
    print("detail columns sample:", dd.columns.tolist()[:40])
    print("detail sample head:")
    print(dd.head(10).to_dict(orient="records"))

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "standard")