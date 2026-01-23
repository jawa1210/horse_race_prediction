# finalize_day.py
from __future__ import annotations

from pathlib import Path
import re

from update_results_flat import update_results_flat_for_date, resolve_population_csv_for_date
from update_payouts_flat import update_payouts_flat_for_date
from report_html import build_html_report
from feature_engineering import RAW_DATA_DIR, DATA_DIR

INPUT_DIR = DATA_DIR / "01_preprocessed"


def _mmdd_from_date(target_date: str) -> str:
    m = re.search(r"(\d{4})-(\d{2})-(\d{2})", str(target_date))
    if not m:
        raise ValueError(f"invalid target_date: {target_date}")
    return f"{m.group(2)}{m.group(3)}"


def resolve_horse_csv_for_date(
    target_date: str,
    horse_csv: Path | None = None,
    input_dir: Path | None = None,
    prefer_daily: bool = True,
) -> Path:
    if horse_csv is not None:
        return Path(horse_csv).resolve()

    input_dir = (input_dir or INPUT_DIR).resolve()
    mmdd = _mmdd_from_date(target_date)

    daily = input_dir / f"horse_results_prediction_{mmdd}.csv"
    default = input_dir / "horse_results_prediction.csv"

    if prefer_daily and daily.exists():
        return daily.resolve()
    return default.resolve()


def finalize_day(date: str, topk_per_race: int = 18, verbose: bool = True):
    pop_path = resolve_population_csv_for_date(
        target_date=date,
        population_dir=RAW_DATA_DIR / "prediction_population",
        prefer_daily=True,
    )

    horse_path = resolve_horse_csv_for_date(
        target_date=date,
        input_dir=INPUT_DIR,
        prefer_daily=True,
    )

    update_results_flat_for_date(date, population_csv=pop_path)
    update_payouts_flat_for_date(date, population_csv=pop_path)

    out_path = build_html_report(
        date,
        read_file_name_csv=str(pop_path),
        read_horse_name_csv=str(horse_path),  # ★これが必要
        topk_per_race=topk_per_race,
    )

    if verbose:
        print(f"[finalize_day] done: {out_path}")
    return out_path


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python finalize_day.py YYYY-MM-DD")
        raise SystemExit(1)
    finalize_day(sys.argv[1])
