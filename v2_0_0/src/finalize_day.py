# v2_0_0/src/finalize_day.py
from __future__ import annotations

from pathlib import Path
import re
import datetime as dt

from feature_engineering import RAW_DATA_DIR, DATA_DIR
from report_html import build_html_report

# 後日(bin)更新
from update_results_flat import update_results_flat_for_date, resolve_population_csv_for_date
from update_payouts_flat import update_payouts_flat_for_date

# 当日(realtime)更新
from live_update import live_update_two_stage

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


def _decide_mode(mode: str, date: str) -> str:
    """
    mode:
      - "realtime": result.html 直読み
      - "bin": bin(html/race/*.bin) から更新
      - "auto": 基本 realtime。もし date が今日より前なら bin。
    """
    mode = str(mode).lower().strip()
    if mode in ["realtime", "bin"]:
        return mode

    if mode != "auto":
        raise ValueError(f"invalid mode: {mode} (use realtime/bin/auto)")

    # auto判定：date が今日より前なら「後日更新」とみなして bin
    d = dt.datetime.strptime(date, "%Y-%m-%d").date()
    today = dt.date.today()
    return "bin" if d < today else "realtime"


def finalize_day(
    date: str,
    mode: str = "auto",
    skip_agg_horse: bool = False,
    topk_per_race: int = 18,
    verbose: bool = True,
):
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

    mode2 = _decide_mode(mode, date)
    if verbose:
        print(f"[finalize_day] mode={mode2}")

    # =========================
    # 更新フェーズ（切り替え）
    # =========================
    if mode2 == "realtime":
        # 当日：CSVに無い分だけ result.html に取りに行く
        live_update_two_stage(
            target_date=date,
            population_csv=pop_path,
            verbose=verbose,
        )
    else:
        # 後日：binから確定データを更新（あなたの既存フロー）
        update_results_flat_for_date(date, population_csv=pop_path, verbose=verbose)
        update_payouts_flat_for_date(date, population_csv=pop_path, verbose=verbose)

    # =========================
    # HTML生成
    # =========================
    out_path = build_html_report(
        date,
        read_file_name_csv=str(pop_path),
        read_horse_name_csv=str(horse_path),
        topk_per_race=topk_per_race,
        skip_agg_horse=skip_agg_horse,
    )

    if verbose:
        print(f"[finalize_day] done: {out_path}")
    return out_path
