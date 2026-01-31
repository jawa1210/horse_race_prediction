# v2_0_0/src/update_payouts_flat.py
from __future__ import annotations

from pathlib import Path
import re
import pandas as pd
from bs4 import BeautifulSoup

from feature_engineering import RAW_DATA_DIR, DATA_DIR
from update_results_flat import _parse_results_from_race_html_bytes


# =========================================================
# Paths
# =========================================================
def _mmdd_from_date(target_date: str) -> str:
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
    if population_csv is not None:
        return Path(population_csv).resolve()

    population_dir = (population_dir or (RAW_DATA_DIR / "prediction_population")).resolve()
    mmdd = _mmdd_from_date(target_date)

    daily = population_dir / f"population_{mmdd}.csv"
    default = population_dir / "population.csv"

    if prefer_daily and daily.exists():
        return daily.resolve()
    return default.resolve()


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
    payouts_flat_path = (DATA_DIR / "results_cache" / "payouts_flat.csv").resolve()

    start = _this_file_dir()
    html_race_dir = (
        _find_upwards(start, Path("..") / ".." / "common" / "data" / "html" / "race")
        or _find_upwards(start, Path("..") / "common" / "data" / "html" / "race")
        or _find_upwards(start, Path("common") / "data" / "html" / "race")
        or (start / ".." / ".." / "common" / "data" / "html" / "race").resolve()
    )

    return {
        "population_csv": population_csv,
        "payouts_flat_path": payouts_flat_path,
        "html_race_dir": html_race_dir,
    }


def _ensure_parent(p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)


def _read_bin_html(path: Path) -> bytes:
    b = path.read_bytes()
    b = b.replace(b"<diary_snap_cut>", b"").replace(b"</diary_snap_cut>", b"")
    return b


# =========================================================
# Features mapping: race_id -> actual date
# =========================================================
FEATURES_PATH = DATA_DIR / "02_features" / "features.csv"


def _read_tab_or_csv(path: Path, dtype: dict | None = None) -> pd.DataFrame:
    text = path.read_text(encoding="utf-8", errors="ignore")
    sep = "\t" if ("\t" in (text.splitlines()[0] if text.splitlines() else "")) else ","
    return pd.read_csv(path, sep=sep, dtype=dtype)


def _load_raceid_to_actual_date_map() -> dict[str, str]:
    if not FEATURES_PATH.exists():
        return {}

    feat = _read_tab_or_csv(FEATURES_PATH, dtype={"race_id": str, "horse_id": str})
    if "race_id" not in feat.columns or "date" not in feat.columns:
        return {}

    feat["race_id"] = feat["race_id"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(12)
    feat["date"] = pd.to_datetime(feat["date"], errors="coerce")

    m = (
        feat.dropna(subset=["race_id", "date"])
        .drop_duplicates("race_id")[["race_id", "date"]]
        .assign(date_str=lambda d: d["date"].dt.strftime("%Y-%m-%d"))
    )
    return dict(zip(m["race_id"].tolist(), m["date_str"].tolist()))


def _apply_actual_date(df: pd.DataFrame, rid2date: dict[str, str]) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    if "race_id" not in df.columns:
        return df

    out = df.copy()
    out["race_id"] = out["race_id"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(12)
    out["date"] = out["race_id"].map(rid2date)
    out = out.dropna(subset=["date"]).copy()
    out["date"] = out["date"].astype(str)
    return out


# =========================================================
# Parsing helpers（あなたの現状をそのまま）
# =========================================================
BET_MAP = {
    "単勝": "tansho",
    "複勝": "fukusho",
    "枠連": "wakuren",
    "馬連": "umaren",
    "馬単": "umatan",
    "ワイド": "wide",
    "3連複": "sanrenpuku",
    "三連複": "sanrenpuku",
    "3連単": "sanrentan",
    "三連単": "sanrentan",
}

TRCLASS_TO_KEY = {
    "Tansho": "単勝",
    "Fukusho": "複勝",
    "Wakuren": "枠連",
    "Umaren": "馬連",
    "Wide": "ワイド",
    "Umatan": "馬単",
    "Sanrenpuku": "3連複",
    "Sanrentan": "3連単",
    "Tan3": "3連単",
    "Fuku3": "3連複",
}

def _unique_keep_order(xs: list[str]) -> list[str]:
    seen = set()
    out = []
    for x in xs:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out

def _norm_combo(bet_type: str, combo: str) -> str:
    nums = re.findall(r"\d+", str(combo))
    if not nums:
        return ""
    if bet_type in ["umaren", "wide", "sanrenpuku"]:
        nums = sorted([int(x) for x in nums])
        return "-".join(map(str, nums))
    if bet_type in ["tansho", "fukusho"]:
        return str(int(nums[0]))
    return "-".join(map(str, [int(x) for x in nums]))

def _extract_pay_candidates(pay_text: str) -> list[int]:
    s = str(pay_text)
    nums = re.findall(r"(\d[\d,]*)\s*円", s)
    out: list[int] = []
    for n in nums:
        n2 = n.replace(",", "")
        if n2.isdigit():
            out.append(int(n2))
    if out:
        return out
    nums2 = re.findall(r"\d[\d,]*", s)
    for n in nums2:
        n2 = n.replace(",", "")
        if n2.isdigit():
            out.append(int(n2))
    return out

def _extract_combos_from_td_text(bet_type: str, td_text: str) -> list[str]:
    t = re.sub(r"\s+", " ", str(td_text)).strip()

    if bet_type in ["tansho", "fukusho"]:
        nums = re.findall(r"\d+", t)
        return _unique_keep_order([str(int(n)) for n in nums])

    if bet_type in ["umaren", "wide"]:
        nums = re.findall(r"\d+", t)
        if len(nums) >= 2 and len(nums) % 2 == 0 and "-" not in t and "→" not in t:
            pairs = []
            for i in range(0, len(nums), 2):
                pairs.append(f"{nums[i]}-{nums[i+1]}")
            combos = [_norm_combo(bet_type, p) for p in pairs]
            combos = [c for c in combos if c]
            return _unique_keep_order(combos)

        pairs = re.findall(r"\d+\s*-\s*\d+", t)
        combos = [_norm_combo(bet_type, p) for p in pairs]
        combos = [c for c in combos if c]
        return _unique_keep_order(combos)

    if bet_type == "umatan":
        seq = re.findall(r"\d+\s*[→\-]\s*\d+", t)
        if seq:
            combos = [_norm_combo(bet_type, s) for s in seq]
            combos = [c for c in combos if c]
            return _unique_keep_order(combos)
        nums = re.findall(r"\d+", t)
        if len(nums) == 2:
            return [f"{int(nums[0])}-{int(nums[1])}"]
        return []

    if bet_type == "sanrenpuku":
        tri = re.findall(r"\d+\s*-\s*\d+\s*-\s*\d+", t)
        if tri:
            combos = [_norm_combo(bet_type, s) for s in tri]
            combos = [c for c in combos if c]
            return _unique_keep_order(combos)
        nums = re.findall(r"\d+", t)
        if len(nums) == 3:
            return ["-".join(map(str, sorted([int(x) for x in nums])))]
        return []

    if bet_type == "sanrentan":
        tri = re.findall(r"\d+\s*[→\-]\s*\d+\s*[→\-]\s*\d+", t)
        if tri:
            combos = [_norm_combo(bet_type, s) for s in tri]
            combos = [c for c in combos if c]
            return _unique_keep_order(combos)
        nums = re.findall(r"\d+", t)
        if len(nums) == 3:
            return ["-".join(map(str, [int(x) for x in nums]))]
        return []

    return []

def _top3_from_results_html(race_id: str, html: bytes) -> list[int] | None:
    try:
        df = _parse_results_from_race_html_bytes(race_id, html)
        if df is None or len(df) == 0:
            return None
        df = df.sort_values("rank")
        top3 = df["umaban"].astype(int).head(3).tolist()
        if len(top3) >= 3:
            return top3[:3]
        return None
    except Exception:
        return None

def _combo_from_top3(bet_type: str, top3: list[int]) -> str | None:
    w1, w2, w3 = top3
    if bet_type == "umatan":
        return f"{w1}-{w2}"
    if bet_type == "sanrenpuku":
        return "-".join(map(str, sorted([w1, w2, w3])))
    if bet_type == "sanrentan":
        return f"{w1}-{w2}-{w3}"
    return None

def _detect_bet_key_from_tr(tr) -> str:
    th = tr.find("th")
    if th:
        key = th.get_text(strip=True)
        key = key.replace("三連複", "3連複").replace("三連単", "3連単").strip()
        if key in BET_MAP:
            return key
        img = th.find("img")
        if img:
            for cand in [(img.get("alt") or ""), (img.get("title") or "")]:
                cand = cand.strip().replace("三連複", "3連複").replace("三連単", "3連単")
                if cand in BET_MAP:
                    return cand
    cls = tr.get("class") or []
    cls = cls[0] if cls else ""
    key = TRCLASS_TO_KEY.get(cls, "")
    if key in BET_MAP:
        return key
    t = tr.get_text(" ", strip=True)
    t = t.replace("三連複", "3連複").replace("三連単", "3連単")
    for k in BET_MAP.keys():
        if k in t:
            return k
    return ""

def _extract_rows_from_one_table(pay_table) -> list[dict]:
    rows = []
    for tr in pay_table.find_all("tr"):
        tds = tr.find_all("td")
        if not tds:
            continue

        key = _detect_bet_key_from_tr(tr)
        bet_type = BET_MAP.get(key, "")

        if not bet_type:
            td0 = tds[0].get_text(strip=True).replace("三連複", "3連複").replace("三連単", "3連単")
            if td0 in BET_MAP:
                bet_type = BET_MAP[td0]
                tds = tds[1:]

        if not bet_type:
            continue

        combo_text = tds[0].get_text(" ", strip=True)
        pay_texts = [td.get_text(" ", strip=True) for td in tds[1:]]

        combos = _extract_combos_from_td_text(bet_type, combo_text)

        pays: list[int] = []
        for pt in pay_texts:
            pays.extend(_extract_pay_candidates(pt))

        n = min(len(combos), len(pays))
        for i in range(n):
            c, p = combos[i], pays[i]
            if c and p:
                rows.append({"bet_type": bet_type, "combo": c, "pay100": int(p)})
    return rows

def _extract_payonly_from_wide_table(pay_tables) -> dict[str, int]:
    out: dict[str, int] = {}
    for t in pay_tables:
        summ = (t.get("summary") or "")
        if "ワイド" not in summ:
            continue
        for tr in t.find_all("tr"):
            cls = tr.get("class") or []
            cls = cls[0] if cls else ""
            cls_to_bt = {"Umatan": "umatan", "Fuku3": "sanrenpuku", "Tan3": "sanrentan"}
            bt = cls_to_bt.get(cls)
            if not bt:
                continue
            tds = tr.find_all("td")
            if len(tds) < 2:
                continue
            pay_text = tds[1].get_text(" ", strip=True)
            pays = _extract_pay_candidates(pay_text)
            if pays:
                out[bt] = int(pays[0])
    return out

def _parse_payouts_from_race_html_bytes(race_id: str, html: bytes) -> pd.DataFrame:
    soup = BeautifulSoup(html, "lxml")

    pay_tables = soup.find_all(
        "table",
        class_=re.compile(r"(pay_table|Pay_Table|pay_table_01|Payout_Detail_Table)", re.I),
    )
    if not pay_tables:
        all_tables = soup.find_all("table")
        for t in all_tables:
            summ = (t.get("summary") or "")
            if ("払戻" in summ) or ("払い戻し" in summ) or ("払戻" in t.get_text()) or ("払い戻し" in t.get_text()):
                pay_tables.append(t)

    rows: list[dict] = []
    for t in pay_tables:
        rows.extend(_extract_rows_from_one_table(t))

    top3 = _top3_from_results_html(race_id, html)
    payonly = _extract_payonly_from_wide_table(pay_tables)

    exist_types = set([r["bet_type"] for r in rows])
    if top3:
        for bt in ["umatan", "sanrenpuku", "sanrentan"]:
            if bt in exist_types:
                continue
            if bt not in payonly:
                continue
            combo = _combo_from_top3(bt, top3)
            if combo:
                rows.append({"bet_type": bt, "combo": combo, "pay100": int(payonly[bt])})

    if not rows:
        return pd.DataFrame(columns=["race_id", "bet_type", "combo", "pay100"])

    df = pd.DataFrame(rows)
    df["pay100"] = pd.to_numeric(df["pay100"], errors="coerce")
    df = (
        df.dropna(subset=["pay100"])
        .groupby(["bet_type", "combo"], as_index=False)["pay100"]
        .max()
    )

    df.insert(0, "race_id", str(race_id))
    df["pay100"] = pd.to_numeric(df["pay100"], errors="coerce").fillna(0).astype(int)
    df = df[df["pay100"] > 0].copy()
    return df[["race_id", "bet_type", "combo", "pay100"]]


# =========================================================
# Public API
# =========================================================
def update_payouts_flat_for_date(
    target_date: str,
    population_csv: Path | None = None,
    html_race_dir: Path | None = None,
    payouts_flat_path: Path | None = None,
    verbose: bool = True,
) -> tuple[pd.DataFrame, Path]:
    """
    ★ date は target_date ではなく features.csv から得た実開催日に上書きする
    """
    defaults = resolve_default_paths()
    population_csv = resolve_population_csv_for_date(
        target_date=target_date,
        population_csv=population_csv,
        population_dir=None,
        prefer_daily=True,
    )

    html_race_dir = (html_race_dir or defaults["html_race_dir"]).resolve()
    payouts_flat_path = (payouts_flat_path or defaults["payouts_flat_path"]).resolve()

    if not population_csv.exists():
        raise FileNotFoundError(f"population.csv not found: {population_csv}")
    if not html_race_dir.exists():
        raise FileNotFoundError(f"common html/race dir not found: {html_race_dir}")

    pop = pd.read_csv(population_csv, sep="\t", dtype={"race_id": str})
    pop["date"] = pop["date"].astype(str).str.strip()
    race_ids = sorted(pop.loc[pop["date"] == str(target_date), "race_id"].astype(str).unique())
    if not race_ids:
        raise ValueError(f"{target_date} の race_id が population.csv にありません")

    rows = []
    failed = []

    for rid in race_ids:
        bin_path = html_race_dir / f"{rid}.bin"
        if not bin_path.exists():
            failed.append((rid, "html_not_found"))
            continue
        try:
            html = _read_bin_html(bin_path)
            df1 = _parse_payouts_from_race_html_bytes(rid, html)
            if len(df1):
                # 仮 date（あとで実開催日に上書き）
                df1.insert(0, "date", str(target_date))
                rows.append(df1)
        except Exception as e:
            failed.append((rid, repr(e)))

    new_df = (
        pd.concat(rows, ignore_index=True)
        if rows
        else pd.DataFrame(columns=["date", "race_id", "bet_type", "combo", "pay100"])
    )

    # ★実開催日へ補正（ここが肝）
    rid2date = _load_raceid_to_actual_date_map()
    new_df = _apply_actual_date(new_df, rid2date)

    _ensure_parent(payouts_flat_path)
    if payouts_flat_path.exists() and payouts_flat_path.stat().st_size > 0:
        try:
            old = pd.read_csv(payouts_flat_path, dtype={"race_id": str, "bet_type": str, "combo": str})
        except pd.errors.EmptyDataError:
            old = pd.DataFrame(columns=["date", "race_id", "bet_type", "combo", "pay100"])
    else:
        old = pd.DataFrame(columns=["date", "race_id", "bet_type", "combo", "pay100"])

    merged = pd.concat([old, new_df], ignore_index=True)
    merged["date"] = merged["date"].astype(str)
    merged["race_id"] = merged["race_id"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(12)
    merged["bet_type"] = merged["bet_type"].astype(str)
    merged["combo"] = merged["combo"].astype(str)
    merged["pay100"] = pd.to_numeric(merged["pay100"], errors="coerce").fillna(0).astype(int)

    merged = merged.sort_values(["date", "race_id", "bet_type", "combo", "pay100"])
    merged = merged.drop_duplicates(subset=["date", "race_id", "bet_type", "combo"], keep="last")
    merged.to_csv(payouts_flat_path, index=False, encoding="utf-8")

    if verbose:
        print("[update_payouts_flat_for_date]")
        print(" population_csv:", population_csv)
        print(" html_race_dir :", html_race_dir)
        print(" payouts_flat  :", payouts_flat_path)
        print(" new rows      :", len(new_df))
        if failed:
            print("[WARN] failed races (first 10)")
            print(pd.DataFrame(failed, columns=["race_id", "reason"]).head(10).to_string(index=False))

    return new_df, payouts_flat_path
