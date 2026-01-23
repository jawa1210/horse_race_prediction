# update_results_flat.py
from __future__ import annotations

from pathlib import Path
import re
import pandas as pd
from bs4 import BeautifulSoup

from feature_engineering import RAW_DATA_DIR, DATA_DIR


# -------------------------
# Utilities
# -------------------------
def _mmdd_from_date(target_date: str) -> str:
    """'2026-01-18' -> '0118'"""
    m = re.search(r"(\d{4})-(\d{2})-(\d{2})", str(target_date))
    if not m:
        raise ValueError(f"invalid target_date: {target_date}")
    return f"{m.group(2)}{m.group(3)}"


def resolve_population_csv_for_date(
    target_date: str,
    population_csv: Path | None = None,
    population_dir: Path | None = None,
    prefer_daily: bool = True,
) -> Path:
    """
    優先順位:
      1) 引数 population_csv が指定されていればそれ
      2) prefer_daily=True なら population_MMDD.csv があればそれ
      3) なければ population.csv
    """
    if population_csv is not None:
        return Path(population_csv).resolve()

    population_dir = (population_dir or (RAW_DATA_DIR / "prediction_population")).resolve()
    mmdd = _mmdd_from_date(target_date)

    daily = population_dir / f"population_{mmdd}.csv"
    default = population_dir / "population.csv"

    if prefer_daily and daily.exists():
        return daily.resolve()
    return default.resolve()


# -------------------------
# Path auto-resolver
# -------------------------
def _this_file_dir() -> Path:
    return Path(__file__).resolve().parent


def _find_upwards(start: Path, rel: Path, max_up: int = 6) -> Path | None:
    cur = start
    for _ in range(max_up + 1):
        cand = (cur / rel).resolve()
        if cand.exists():
            return cand
        if cur.parent == cur:
            break
        cur = cur.parent
    return None


def resolve_default_paths() -> dict[str, Path]:
    population_csv = (RAW_DATA_DIR / "prediction_population" / "population.csv").resolve()
    results_flat_path = (DATA_DIR / "results_cache" / "results_flat.csv").resolve()

    start = _this_file_dir()
    html_race_dir = (
        _find_upwards(start, Path("..") / ".." / "common" / "data" / "html" / "race")
        or _find_upwards(start, Path("..") / "common" / "data" / "html" / "race")
        or _find_upwards(start, Path("common") / "data" / "html" / "race")
        or (start / ".." / ".." / "common" / "data" / "html" / "race").resolve()
    )

    return {
        "population_csv": population_csv,
        "results_flat_path": results_flat_path,
        "html_race_dir": html_race_dir,
    }


# -------------------------
# HTML -> results parser
# -------------------------
def _ensure_parent(p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)


def _read_bin_html(path: Path) -> bytes:
    b = path.read_bytes()
    # netkeibaのbin HTMLに混入することがあるタグを除去
    b = b.replace(b"<diary_snap_cut>", b"").replace(b"</diary_snap_cut>", b"")
    return b


def _pick_col(df: pd.DataFrame, patterns: list[str]) -> str | None:
    cols = [str(c) for c in df.columns]
    for pat in patterns:
        for c in cols:
            if re.search(pat, c):
                return c
    return None


def _parse_results_from_race_html_bytes(race_id: str, html: bytes) -> pd.DataFrame:
    """
    race HTML(bytes) から
      race_id, umaban, rank, agari
    を抜く。
    """
    soup = BeautifulSoup(html, "lxml")

    # 結果テーブル（環境でclassが揺れるので両対応）
    table = soup.find("table", class_="RaceTable01") or soup.find("table", class_="race_table_01 nk_tb_common")
    if table is None:
        raise ValueError(f"result table not found: {race_id}")

    df = pd.read_html(html)[0].copy()
    df.columns = df.columns.astype(str).str.replace(" ", "")

    col_rank = _pick_col(df, [r"^着順$", r"着順", r"^着$"])
    col_uma = _pick_col(df, [r"^馬番$", r"馬番", r"^馬$"])
    col_agari = _pick_col(df, [r"上り3F", r"上り", r"^上り$"])

    if col_rank is None or col_uma is None:
        raise ValueError(f"rank/umaban col not found: {race_id} cols={list(df.columns)}")

    out = pd.DataFrame(
        {
            "race_id": str(race_id),
            "umaban": pd.to_numeric(df[col_uma], errors="coerce"),
            "rank": pd.to_numeric(df[col_rank], errors="coerce"),
        }
    )

    # 上がり（存在しないときはNA）
    if col_agari is not None:
        out["agari"] = pd.to_numeric(df[col_agari], errors="coerce")
    else:
        out["agari"] = pd.NA

    out = out.dropna(subset=["umaban", "rank"]).copy()
    out["umaban"] = out["umaban"].astype(int)
    out["rank"] = out["rank"].astype(int)
    # agari は float のままでOK（NAも保持）
    return out[["race_id", "umaban", "rank", "agari"]]


# -------------------------
# Public API
# -------------------------
def update_results_flat_for_date(
    target_date: str,
    population_csv: Path | None = None,
    html_race_dir: Path | None = None,
    results_flat_path: Path | None = None,
    prefer_daily_population: bool = True,
    verbose: bool = True,
) -> tuple[pd.DataFrame, Path]:
    """
    common/data/html/race/{race_id}.bin から結果を抽出して
    DATA_DIR/results_cache/results_flat.csv に追記・更新する。

    保存列:
      date, race_id, umaban, rank, agari
    """
    defaults = resolve_default_paths()

    population_csv = resolve_population_csv_for_date(
        target_date=target_date,
        population_csv=population_csv,
        population_dir=RAW_DATA_DIR / "prediction_population",
        prefer_daily=prefer_daily_population,
    )
    html_race_dir = (html_race_dir or defaults["html_race_dir"]).resolve()
    results_flat_path = (results_flat_path or defaults["results_flat_path"]).resolve()

    if not population_csv.exists():
        raise FileNotFoundError(f"population_csv not found: {population_csv}")
    if not html_race_dir.exists():
        raise FileNotFoundError(f"common html/race dir not found: {html_race_dir}")

    pop = pd.read_csv(population_csv, sep="\t", dtype={"race_id": str})
    pop["date"] = pop["date"].astype(str).str.strip()

    race_ids = sorted(pop.loc[pop["date"] == str(target_date), "race_id"].astype(str).unique())
    if not race_ids:
        raise ValueError(f"{target_date} の race_id が population_csv にありません: {population_csv}")

    rows: list[pd.DataFrame] = []
    failed: list[tuple[str, str]] = []

    for rid in race_ids:
        bin_path = html_race_dir / f"{rid}.bin"
        if not bin_path.exists():
            failed.append((rid, "html_not_found"))
            continue
        try:
            html = _read_bin_html(bin_path)
            df1 = _parse_results_from_race_html_bytes(rid, html)
            df1["date"] = str(target_date)
            rows.append(df1[["date", "race_id", "umaban", "rank", "agari"]])
        except Exception as e:
            failed.append((rid, repr(e)))

    new_df = (
        pd.concat(rows, ignore_index=True)
        if rows
        else pd.DataFrame(columns=["date", "race_id", "umaban", "rank", "agari"])
    )

    # 既存を読み込み（なければ空）
    _ensure_parent(results_flat_path)
    if results_flat_path.exists() and results_flat_path.stat().st_size > 0:
        try:
            old = pd.read_csv(results_flat_path, dtype={"race_id": str})
        except pd.errors.EmptyDataError:
            old = pd.DataFrame(columns=["date", "race_id", "umaban", "rank", "agari"])
    else:
        old = pd.DataFrame(columns=["date", "race_id", "umaban", "rank", "agari"])

    # 旧ファイルにagari列が無い場合でも落ちないように補完
    if "agari" not in old.columns:
        old["agari"] = pd.NA

    merged = pd.concat([old, new_df], ignore_index=True)

    merged["date"] = merged["date"].astype(str)
    merged["race_id"] = merged["race_id"].astype(str)
    merged["umaban"] = pd.to_numeric(merged["umaban"], errors="coerce")
    merged["rank"] = pd.to_numeric(merged["rank"], errors="coerce")
    merged["agari"] = pd.to_numeric(merged["agari"], errors="coerce")

    # date/race_id/umaban/rank は必須、agari は任意
    merged = merged.dropna(subset=["date", "race_id", "umaban", "rank"]).copy()
    merged["umaban"] = merged["umaban"].astype(int)
    merged["rank"] = merged["rank"].astype(int)

    merged = merged.drop_duplicates(subset=["date", "race_id", "umaban"], keep="last")
    merged = merged.sort_values(["date", "race_id", "rank", "umaban"])
    merged.to_csv(results_flat_path, index=False, encoding="utf-8")

    if verbose:
        print("[update_results_flat_for_date]")
        print(" population_csv:", population_csv)
        print(" html_race_dir :", html_race_dir)
        print(" results_flat  :", results_flat_path)
        print(" new rows      :", len(new_df))
        if failed:
            print("[WARN] failed races (first 10)")
            print(pd.DataFrame(failed, columns=["race_id", "reason"]).head(10).to_string(index=False))

    return new_df, results_flat_path
