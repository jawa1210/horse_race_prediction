from __future__ import annotations

from pathlib import Path
import re
import pandas as pd
from bs4 import BeautifulSoup

from feature_engineering import RAW_DATA_DIR, DATA_DIR


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

def _extract_payout_rows_from_pay_table(pay_table) -> list[dict]:
    """
    netkeibaの払戻表は race.bin 内に存在。
    ただし HTML構造が微妙に揺れるので、
    行ごとに「券種名(th)」「組み合わせ」「払戻」を拾う方式。
    """
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

        # 典型: td[0]=組み合わせ, td[1]=払戻
        combo_text = tds[0].get_text(" ", strip=True)
        pay_text = tds[1].get_text(" ", strip=True)

        # 複勝/ワイドは複数候補が同じセルに並ぶことがあるので分割を試みる
        # まず払戻を全部抜く
        pays = re.findall(r"\d[\d,]*", pay_text)
        pays = [p.replace(",", "") for p in pays]
        pays = [int(p) for p in pays if p.isdigit()]

        # 組み合わせ側も数字列を取り出す（例: "3 7 10" や "3-7 7-10" など）
        # ここでは単純化：pay数と対応させるため、"組み合わせ"の候補を複数抽出
        # 1) "3-7" のようなハイフン結合を優先抽出
        combo_candidates = re.findall(r"\d+\s*[-→]\s*\d+\s*[-→]?\s*\d*", combo_text)
        if not combo_candidates:
            # 2) 空白区切りの数字列
            combo_candidates = re.findall(r"(?:\d+\s+){1,2}\d+", combo_text)
        if not combo_candidates:
            # 3) 最後の手段：数字だけ
            combo_candidates = [" ".join(re.findall(r"\d+", combo_text))] if re.findall(r"\d+", combo_text) else []

        # pays が1個ならそのまま
        if len(pays) <= 1:
            pay100 = pays[0] if pays else None
            combo_norm = _norm_combo(bet_type, combo_text)
            if pay100 is not None and combo_norm:
                rows.append({"bet_type": bet_type, "combo": combo_norm, "pay100": pay100})
            continue

        # pays が複数なら、combo_candidates と対応づける
        # 長さが合えばペアにする、合わなければ「候補を順に使う」
        for i, pay100 in enumerate(pays):
            if i < len(combo_candidates):
                combo_norm = _norm_combo(bet_type, combo_candidates[i])
            else:
                combo_norm = _norm_combo(bet_type, combo_text)
            if pay100 is not None and combo_norm:
                rows.append({"bet_type": bet_type, "combo": combo_norm, "pay100": int(pay100)})

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

    df = pd.DataFrame(rows).drop_duplicates(subset=["bet_type", "combo"], keep="last")
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
    population_csv = (population_csv or defaults["population_csv"]).resolve()
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

    # 同一(date,race_id,bet_type,combo)は最後勝ち
    merged = merged.drop_duplicates(subset=["date","race_id","bet_type","combo"], keep="last")
    merged = merged.sort_values(["date","race_id","bet_type","combo"])
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
