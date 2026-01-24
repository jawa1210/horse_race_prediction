# v2_0_0/src/live_update.py
from __future__ import annotations

from pathlib import Path
import random
import time
import pandas as pd
import requests

from feature_engineering import RAW_DATA_DIR, DATA_DIR

# 既存パーサ（あなたのコードに合わせて import）
from update_results_flat import _parse_results_from_race_html_bytes
from update_payouts_flat import _parse_payouts_from_race_html_bytes


RESULTS_DIR = DATA_DIR / "results_cache"
RESULTS_FLAT = RESULTS_DIR / "results_flat.csv"
PAYOUTS_FLAT = RESULTS_DIR / "payouts_flat.csv"

DEFAULT_UA = [
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17 Safari/605.1.15",
]

def _ensure_parent(p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)

def fetch_result_html_bytes(race_id: str, timeout: float = 15.0) -> bytes:
    url = f"https://race.netkeiba.com/race/result.html?race_id={race_id}"
    headers = {"User-Agent": random.choice(DEFAULT_UA)}
    r = requests.get(url, headers=headers, timeout=timeout)
    r.raise_for_status()
    return r.content

def _load_race_ids_for_date(target_date: str, population_csv: Path) -> list[str]:
    pop = pd.read_csv(population_csv, sep="\t", dtype={"race_id": str})
    pop["date"] = pop["date"].astype(str).str.strip()
    return sorted(pop.loc[pop["date"] == str(target_date), "race_id"].astype(str).unique())

def _race_ids_with_results(target_date: str, results_flat_path: Path) -> set[str]:
    if not results_flat_path.exists() or results_flat_path.stat().st_size == 0:
        return set()
    try:
        df = pd.read_csv(results_flat_path, dtype={"race_id": str})
    except pd.errors.EmptyDataError:
        return set()
    need = {"date", "race_id", "umaban", "rank"}
    if not need.issubset(df.columns):
        return set()
    df = df[df["date"].astype(str) == str(target_date)]
    return set(df["race_id"].astype(str).unique())

def _race_ids_with_payouts(target_date: str, payouts_flat_path: Path) -> set[str]:
    """
    pay100>0 が1件でもあるだけでは「揃った」判定にしない。
    ワイド/三連複/三連単 が入っていたら “揃った” とみなす。
    """
    if not payouts_flat_path.exists() or payouts_flat_path.stat().st_size == 0:
        return set()
    try:
        df = pd.read_csv(payouts_flat_path, dtype={"race_id": str, "bet_type": str, "combo": str})
    except pd.errors.EmptyDataError:
        return set()

    need = {"date", "race_id", "bet_type", "pay100"}
    if not need.issubset(df.columns):
        return set()

    df = df[df["date"].astype(str) == str(target_date)].copy()
    if df.empty:
        return set()

    df["pay100"] = pd.to_numeric(df["pay100"], errors="coerce").fillna(0).astype(int)
    df = df[df["pay100"] > 0].copy()

    # ★「揃った判定」：wide / sanrenpuku / sanrentan のいずれかがある
    completed_types = {"wide", "sanrenpuku", "sanrentan"}

    done = set()
    for rid, g in df.groupby("race_id"):
        types = set(g["bet_type"].astype(str))
        if len(types & completed_types) > 0:
            done.add(str(rid))
    return done


def _upsert_results_flat(target_date: str, race_id: str, df_new: pd.DataFrame, results_flat_path: Path):
    _ensure_parent(results_flat_path)

    if results_flat_path.exists() and results_flat_path.stat().st_size > 0:
        try:
            old = pd.read_csv(results_flat_path, dtype={"race_id": str})
        except pd.errors.EmptyDataError:
            old = pd.DataFrame(columns=["date","race_id","umaban","rank","agari"])
    else:
        old = pd.DataFrame(columns=["date","race_id","umaban","rank","agari"])

    if "agari" not in old.columns:
        old["agari"] = pd.NA

    # その日のそのレースを消す
    old = old[~((old["date"].astype(str) == str(target_date)) & (old["race_id"].astype(str) == str(race_id)))].copy()

    df2 = df_new.copy()
    df2["date"] = str(target_date)
    df2["race_id"] = str(race_id)

    merged = pd.concat([old, df2], ignore_index=True)
    merged.to_csv(results_flat_path, index=False, encoding="utf-8")

def _upsert_payouts_flat(target_date: str, race_id: str, df_new: pd.DataFrame, payouts_flat_path: Path):
    _ensure_parent(payouts_flat_path)

    if payouts_flat_path.exists() and payouts_flat_path.stat().st_size > 0:
        try:
            old = pd.read_csv(payouts_flat_path, dtype={"race_id": str, "bet_type": str, "combo": str})
        except pd.errors.EmptyDataError:
            old = pd.DataFrame(columns=["date","race_id","bet_type","combo","pay100"])
    else:
        old = pd.DataFrame(columns=["date","race_id","bet_type","combo","pay100"])

    old = old[~((old["date"].astype(str) == str(target_date)) & (old["race_id"].astype(str) == str(race_id)))].copy()

    df2 = df_new.copy()
    df2["date"] = str(target_date)
    df2["race_id"] = str(race_id)

    merged = pd.concat([old, df2], ignore_index=True)
    merged.to_csv(payouts_flat_path, index=False, encoding="utf-8")

def live_update_two_stage(
    target_date: str,
    population_csv: Path | None = None,
    results_flat_path: Path = RESULTS_FLAT,
    payouts_flat_path: Path = PAYOUTS_FLAT,
    sleep_sec: tuple[float, float] = (0.8, 1.6),
    timeout: float = 15.0,
    verbose: bool = True,
) -> dict:
    """
    2段構え：
      A) results_flat に無いレースだけ取りに行って結果を埋める
      B) payouts_flat に無いレースのうち、結果があるレースだけ取りに行って払戻を埋める
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    population_csv = (population_csv or (RAW_DATA_DIR / "prediction_population" / "population.csv")).resolve()
    race_ids = _load_race_ids_for_date(target_date, population_csv)

    got_res = _race_ids_with_results(target_date, results_flat_path)
    got_pay = _race_ids_with_payouts(target_date, payouts_flat_path)

    # --- Stage A: 結果がないレースだけ ---
    need_res = [rid for rid in race_ids if rid not in got_res]

    # --- Stage B: 払戻がないレース（ただし結果があるものだけ） ---
    # ※ Stage A 後に got_res が増えるので、B は A の後に再計算する
    updated_res = []
    updated_pay = []
    failed = []

    def _sleep():
        time.sleep(random.uniform(*sleep_sec))

    # A
    for rid in need_res:
        try:
            html = fetch_result_html_bytes(rid, timeout=timeout)
            df_res = _parse_results_from_race_html_bytes(rid, html)  # ここで落ちる=未確定
            if df_res is not None and len(df_res) > 0:
                _upsert_results_flat(target_date, rid, df_res, results_flat_path)
                updated_res.append(rid)
        except Exception as e:
            # 未確定はよくあるので verbose時だけ軽く
            failed.append((rid, "results", repr(e)))
        _sleep()

    # Aを反映した上で再取得
    got_res2 = _race_ids_with_results(target_date, results_flat_path)
    got_pay2 = _race_ids_with_payouts(target_date, payouts_flat_path)

    need_pay = [rid for rid in race_ids if (rid in got_res2) and (rid not in got_pay2)]

    # B
    for rid in need_pay:
        try:
            html = fetch_result_html_bytes(rid, timeout=timeout)
            df_pay = _parse_payouts_from_race_html_bytes(rid, html)  # まだ払戻出てないと空
            if df_pay is not None and len(df_pay) > 0:
                _upsert_payouts_flat(target_date, rid, df_pay, payouts_flat_path)
                updated_pay.append(rid)
        except Exception as e:
            failed.append((rid, "payouts", repr(e)))
        _sleep()

    if verbose:
        print("[live_update_two_stage]")
        print(" date:", target_date)
        print(" stageA need_res :", len(need_res))
        print(" stageA updated  :", len(updated_res), updated_res[:8])
        print(" stageB need_pay :", len(need_pay))
        print(" stageB updated  :", len(updated_pay), updated_pay[:8])
        if failed:
            print(" failed (first5):", failed[:5])

    return {
        "updated_results": updated_res,
        "updated_payouts": updated_pay,
        "failed": failed,
        "population_csv": str(population_csv),
    }

def fetch_result_html_bytes(race_id: str, timeout: float = 15.0) -> bytes:
    url = f"https://race.netkeiba.com/race/result.html?race_id={race_id}"
    headers = {"User-Agent": random.choice(DEFAULT_UA)}
    for k in range(2):
        r = requests.get(url, headers=headers, timeout=timeout)
        if r.status_code == 200:
            return r.content
        time.sleep(1.0 + k)
    r.raise_for_status()
    return r.content
