from __future__ import annotations

from pathlib import Path
import re
import pandas as pd
from bs4 import BeautifulSoup

from feature_engineering import RAW_DATA_DIR, DATA_DIR

def _mmdd_from_date(target_date: str) -> str:
    # "2026-01-18" -> "0118"
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
    html_race_dir = _find_upwards(start, Path("..") / ".." / "common" / "data" / "html" / "race")
    if html_race_dir is None:
        html_race_dir = _find_upwards(start, Path("..") / "common" / "data" / "html" / "race")
    if html_race_dir is None:
        html_race_dir = _find_upwards(start, Path("common") / "data" / "html" / "race")
    if html_race_dir is None:
        html_race_dir = (start / ".." / ".." / "common" / "data" / "html" / "race").resolve()

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

def _unique_keep_order(xs: list[str]) -> list[str]:
    seen = set()
    out = []
    for x in xs:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out

def _extract_combos_from_td_text(bet_type: str, td_text: str) -> list[str]:
    t = re.sub(r"\s+", " ", str(td_text))

    if bet_type in ["tansho", "fukusho"]:
        # 馬番だけ（重複は後で除去）
        nums = re.findall(r"\d+", t)
        combos = [str(int(n)) for n in nums]
        return _unique_keep_order(combos)

    if bet_type in ["umaren", "wide"]:
        pairs = re.findall(r"\d+\s*-\s*\d+", t)
        combos = [_norm_combo(bet_type, p) for p in pairs]
        combos = [c for c in combos if c]
        return _unique_keep_order(combos)

    if bet_type in ["umatan"]:
        # → があれば優先、なければ 2頭の順序付きっぽいもの
        seq = re.findall(r"\d+\s*[→\-]\s*\d+", t)
        combos = [_norm_combo(bet_type, s) for s in seq]
        combos = [c for c in combos if c]
        return _unique_keep_order(combos)

    if bet_type in ["sanrenpuku"]:
        tri = re.findall(r"\d+\s*-\s*\d+\s*-\s*\d+", t)
        combos = [_norm_combo(bet_type, s) for s in tri]
        combos = [c for c in combos if c]
        return _unique_keep_order(combos)

    if bet_type in ["sanrentan"]:
        tri = re.findall(r"\d+\s*[→\-]\s*\d+\s*[→\-]\s*\d+", t)
        combos = [_norm_combo(bet_type, s) for s in tri]
        combos = [c for c in combos if c]
        return _unique_keep_order(combos)

    # fallback
    c = _norm_combo(bet_type, t)
    return [c] if c else []



# 券種名 → bet_type
BET_MAP = {
    "単勝": "tansho",
    "複勝": "fukusho",
    "馬連": "umaren",
    "馬単": "umatan",
    "ワイド": "wide",
    "3連複": "sanrenpuku",
    "三連複": "sanrenpuku",
    "3連単": "sanrentan",
    "三連単": "sanrentan",
}

def _clean_num(s: str) -> str:
    return re.sub(r"[^\d\-]", "", str(s))

def _clean_pay(s: str) -> int | None:
    t = re.sub(r"[^\d]", "", str(s))
    if not t:
        return None
    try:
        return int(t)
    except Exception:
        return None

def _norm_combo(bet_type: str, combo: str) -> str:
    # comboは "3-7" "3 7" "3→7" などが来うるので数字だけ拾う
    nums = re.findall(r"\d+", str(combo))
    if not nums:
        return ""
    if bet_type in ["umaren", "wide", "sanrenpuku"]:
        nums = sorted([int(x) for x in nums])
        return "-".join(map(str, nums))
    if bet_type in ["tansho", "fukusho"]:
        return str(int(nums[0]))
    # umatan / sanrentan は順序
    return "-".join(map(str, [int(x) for x in nums]))

def _extract_combo_candidates(bet_type: str, combo_text: str) -> list[str]:
    t = re.sub(r"\s+", " ", str(combo_text)).strip()

    # 単勝/複勝：馬番1つずつにする
    if bet_type in ["tansho", "fukusho"]:
        nums = re.findall(r"\d+", t)
        return nums  # 1頭ずつ

    # 馬単/三連単：順序あり（→ or - を含む形を優先）
    if bet_type in ["umatan", "sanrentan"]:
        c = re.findall(r"\d+\s*[→\-]\s*\d+(?:\s*[→\-]\s*\d+)?", t)
        if c:
            return c

    # 馬連/ワイド：2頭（ハイフン結合を優先）
    if bet_type in ["umaren", "wide"]:
        c = re.findall(r"\d+\s*-\s*\d+", t)
        if c:
            return c

    # 三連複：3頭（ハイフン結合）
    if bet_type in ["sanrenpuku"]:
        c = re.findall(r"\d+\s*-\s*\d+\s*-\s*\d+", t)
        if c:
            return c

    # 最後の手段：数字全部を1候補
    nums = re.findall(r"\d+", t)
    return [" ".join(nums)] if nums else []


def _extract_pay_candidates(pay_text: str) -> list[int]:
    pays = re.findall(r"\d[\d,]*", str(pay_text))
    out = []
    for p in pays:
        p2 = p.replace(",", "")
        if p2.isdigit():
            out.append(int(p2))
    return out


def _cell_lines(td) -> list[str]:
    # <br> を行として扱えるようにする
    txt = td.get_text("\n", strip=True)
    return [x.strip() for x in txt.split("\n") if x.strip()]

def _parse_pay_int(s: str) -> int | None:
    s2 = re.sub(r"[^\d]", "", str(s))
    return int(s2) if s2.isdigit() else None

def _extract_payout_rows_from_pay_table(pay_table) -> list[dict]:
    rows = []
    for tr in pay_table.find_all("tr"):
        th = tr.find("th")
        tds = tr.find_all("td")
        if th is None or len(tds) < 2:
            continue

        key = th.get_text(strip=True)
        if key not in BET_MAP:
            continue
        bet_type = BET_MAP[key]

        # ★ ここを “行分割” じゃなく “全文 regex 抽出” にする
        combo_text = tds[0].get_text(" ", strip=True)
        pay_text   = tds[1].get_text(" ", strip=True)

        combos = _extract_combos_from_td_text(bet_type, combo_text)

        # pay は数字を全部拾う（カンマ除去済）
        pays = _extract_pay_candidates(pay_text)

        # ペアリング（数がズレても安全に min）
        n = min(len(combos), len(pays))
        for i in range(n):
            if combos[i] and pays[i]:
                rows.append({"bet_type": bet_type, "combo": combos[i], "pay100": int(pays[i])})

    return rows



def _parse_payouts_from_race_html_bytes(race_id: str, html: bytes) -> pd.DataFrame:
    soup = BeautifulSoup(html, "lxml")

    # よくある払戻テーブル class
    pay_tables = soup.find_all("table", class_=re.compile(r"pay_table|Pay_Table|pay_table_01", re.I))
    if not pay_tables:
        # クラスが見つからない場合は、"払戻" を含む table を探す
        all_tables = soup.find_all("table")
        for t in all_tables:
            if "払戻" in t.get_text():
                pay_tables.append(t)

    rows = []
    for t in pay_tables:
        rows.extend(_extract_payout_rows_from_pay_table(t))

    if not rows:
        # 払戻がまだ出てない（未来レース等）
        return pd.DataFrame(columns=["race_id", "bet_type", "combo", "pay100"])

    df = pd.DataFrame(rows)
    if len(df):
        # 同一 combo が複数出たら最大 pay を採用（保険）
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


def update_payouts_flat_for_date(
    target_date: str,
    population_csv: Path | None = None,
    html_race_dir: Path | None = None,
    payouts_flat_path: Path | None = None,
    verbose: bool = True,
) -> tuple[pd.DataFrame, Path]:
    defaults = resolve_default_paths()
    population_csv = resolve_population_csv_for_date(
        target_date=target_date,
        population_csv=population_csv,
        population_dir=None,   # ← 渡さない
        prefer_daily=True,
    )
    html_race_dir = (html_race_dir or defaults["html_race_dir"]).resolve()
    payouts_flat_path = (payouts_flat_path or defaults["payouts_flat_path"]).resolve()

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
                df1.insert(0, "date", str(target_date))
                rows.append(df1)
        except Exception as e:
            failed.append((rid, repr(e)))

    new_df = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["date","race_id","bet_type","combo","pay100"])

    _ensure_parent(payouts_flat_path)
    if payouts_flat_path.exists() and payouts_flat_path.stat().st_size > 0:
        try:
            old = pd.read_csv(payouts_flat_path, dtype={"race_id": str, "bet_type": str, "combo": str})
        except pd.errors.EmptyDataError:
            old = pd.DataFrame(columns=["date","race_id","bet_type","combo","pay100"])
    else:
        old = pd.DataFrame(columns=["date","race_id","bet_type","combo","pay100"])

    merged = pd.concat([old, new_df], ignore_index=True)
    merged["date"] = merged["date"].astype(str)
    merged["race_id"] = merged["race_id"].astype(str)
    merged["bet_type"] = merged["bet_type"].astype(str)
    merged["combo"] = merged["combo"].astype(str)
    merged["pay100"] = pd.to_numeric(merged["pay100"], errors="coerce").fillna(0).astype(int)
    # まず並べ替えて「新しい方」が後ろに来るようにする（念のため）
    merged = merged.sort_values(["date","race_id","bet_type","combo","pay100"])

    # 同一(date,race_id,bet_type,combo)は最後勝ち
    merged = merged.drop_duplicates(subset=["date","race_id","bet_type","combo"], keep="last")

    merged.to_csv(payouts_flat_path, index=False, encoding="utf-8")

    if verbose:
        print("[update_payouts_flat_for_date]")
        print(" population_csv:", population_csv)
        print(" html_race_dir :", html_race_dir)
        print(" payouts_flat  :", payouts_flat_path)
        if failed:
            print("[WARN] failed races (first 10)")
            print(pd.DataFrame(failed, columns=["race_id","reason"]).head(10).to_string(index=False))

    return new_df, payouts_flat_path
