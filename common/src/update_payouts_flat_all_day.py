import re
import pandas as pd
from bs4 import BeautifulSoup

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

def _norm_combo(bet_type: str, combo: str) -> str:
    nums = re.findall(r"\d+", str(combo))
    if not nums:
        return ""
    if bet_type in ["umaren", "wide", "sanrenpuku"]:
        nums = sorted([int(x) for x in nums])
        return "-".join(map(str, nums))
    if bet_type in ["tansho", "fukusho"]:
        return str(int(nums[0]))
    # umatan / sanrentan: ordered
    return "-".join(map(str, [int(x) for x in nums]))

def _extract_payout_rows_from_table(pay_table) -> list[dict]:
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

        combo_text = tds[0].get_text(" ", strip=True)
        pay_text = tds[1].get_text(" ", strip=True)

        pays = re.findall(r"\d[\d,]*", pay_text)
        pays = [int(p.replace(",", "")) for p in pays if p.replace(",", "").isdigit()]

        # combo候補（複勝・ワイドは複数並ぶ）
        combo_candidates = re.findall(r"\d+\s*[-→]\s*\d+\s*[-→]?\s*\d*", combo_text)
        if not combo_candidates:
            combo_candidates = re.findall(r"(?:\d+\s+){1,2}\d+", combo_text)
        if not combo_candidates:
            nums = re.findall(r"\d+", combo_text)
            combo_candidates = [" ".join(nums)] if nums else []

        if len(pays) <= 1:
            pay100 = pays[0] if pays else None
            combo_norm = _norm_combo(bet_type, combo_text)
            if pay100 and combo_norm:
                rows.append({"bet_type": bet_type, "combo": combo_norm, "pay100": int(pay100)})
            continue

        for i, pay100 in enumerate(pays):
            if i < len(combo_candidates):
                combo_norm = _norm_combo(bet_type, combo_candidates[i])
            else:
                combo_norm = _norm_combo(bet_type, combo_text)
            if pay100 and combo_norm:
                rows.append({"bet_type": bet_type, "combo": combo_norm, "pay100": int(pay100)})
    return rows

def parse_payouts_from_race_html(html: bytes, race_id: str) -> pd.DataFrame:
    soup = BeautifulSoup(html, "lxml")

    pay_tables = soup.find_all("table", class_=re.compile(r"pay_table|Pay_Table|pay_table_01", re.I))
    if not pay_tables:
        for t in soup.find_all("table"):
            if "払戻" in t.get_text():
                pay_tables.append(t)

    rows = []
    for t in pay_tables:
        rows.extend(_extract_payout_rows_from_table(t))

    if not rows:
        return pd.DataFrame(columns=["race_id", "bet_type", "combo", "pay100"])

    df = pd.DataFrame(rows).drop_duplicates(subset=["bet_type", "combo"], keep="last")
    df.insert(0, "race_id", str(race_id))
    df["pay100"] = pd.to_numeric(df["pay100"], errors="coerce").fillna(0).astype(int)
    df = df[df["pay100"] > 0].copy()
    return df[["race_id", "bet_type", "combo", "pay100"]]
