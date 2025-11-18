"""
Helpers de logging dentro de la carpeta `work/`.
Este módulo reemplaza a `src/utils_logging.py` moviendo la responsabilidad a `work`.
"""
from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT
LOGS = WORK / "logs"
REQ_DIR = LOGS / "requests"
RULES_DIR = LOGS / "rules"
RUNS_DIR = LOGS / "runs"


def ensure_work_dirs() -> None:
    for p in (REQ_DIR, RULES_DIR, RUNS_DIR):
        p.mkdir(parents=True, exist_ok=True)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def log_request_csv(source_name: str, step: str, url_or_file: str, status_code: Optional[int], rows_fetched: Optional[int], duration_s: Optional[float], user_agent: Optional[str], notes: Optional[str] = None) -> None:
    ensure_work_dirs()
    ts = _utc_now_iso()
    date = datetime.now(timezone.utc).strftime("%Y%m%d")
    filename = REQ_DIR / f"{source_name}_{date}.csv"
    header = ["ts_utc", "step", "source_name", "url_or_file", "status_code", "rows_fetched", "duration_s", "user_agent", "notes"]
    row = [ts, step, source_name, url_or_file, status_code if status_code is not None else "", rows_fetched if rows_fetched is not None else "", round(duration_s, 3) if duration_s is not None else "", user_agent or "", notes or ""]
    write_header = not filename.exists()
    try:
        with open(filename, "a", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(header)
            writer.writerow(row)
    except Exception:
        return


def log_rule_jsonl(rule_obj: Dict, fname: Optional[str] = None) -> None:
    ensure_work_dirs()
    date = datetime.now(timezone.utc).strftime("%Y%m%d")
    if fname is None:
        fname = f"integrate_rules_{date}.jsonl"
    path = RULES_DIR / fname
    try:
        with open(path, "a", encoding="utf-8") as f:
            entry = rule_obj.copy()
            entry.setdefault("ts_utc", _utc_now_iso())
            json_line = json.dumps(entry, ensure_ascii=False)
            f.write(json_line + "\n")
    except Exception:
        return


def write_run_summary(run_obj: Dict) -> None:
    ensure_work_dirs()
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    fname = RUNS_DIR / f"run_{ts}.json"
    try:
        with open(fname, "w", encoding="utf-8") as f:
            json.dump(run_obj, f, ensure_ascii=False, indent=2)
    except Exception:
        return

