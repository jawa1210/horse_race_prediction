from __future__ import annotations

from pathlib import Path
from typing import Tuple, List, Dict, Optional

import re
import time
import random

import pandas as pd
from tqdm.notebook import tqdm
from bs4 import BeautifulSoup
import requests



# =========================
# Config
# =========================
RAWDF_DIR = Path("..", "data", "rawdf")

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:115.0) Gecko/20100101 Firefox/115.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:115.0) Gecko/20100101 Firefox/115.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.2 Safari/605.1.15",
]


# =========================
# race_info meta parsing
# =========================
around_mapping = {"右": 0, "左": 1, "直": 2}
inout_mapping = {"内": 0, "外": 1}
course_abc_mapping = {"A": 0, "B": 1, "C": 2}

def parse_course_meta_from_info1(info1_text: str):
    t = re.sub(r"\s+", "", str(info1_text))
    m = re.search(r"(右|左|直)", t)
    around_dir = around_mapping.get(m.group(1), None) if m else None
    m = re.search(r"(内|外)", t)
    around_inout = inout_mapping.get(m.group(1), None) if m else None
    m = re.search(r"(A|B|C)", t)
    course_abc = course_abc_mapping.get(m.group(1), None) if m else None
    return around_dir, around_inout, course_abc


# =========================
# payout parsing (db.netkeiba race html)
# =========================
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

REQUIRED_BET_TYPES = {"tansho", "fukusho", "umaren", "umatan", "wide", "sanrenpuku", "sanrentan"}

def _norm_combo(bet_type: str, combo: str) -> str:
    """
    combo例:
      - 単勝/複勝: "7"
      - 馬連/ワイド: "3-7" or "3 7"
      - 三連系: "3-7-10" etc
    """
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

def _extract_payout_rows_from_table(pay_table) -> List[Dict]:
    """
    払戻表の1テーブルから行を抜く（複勝/ワイドは複数出るので複数行に展開）
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

        combo_text = tds[0].get_text(" ", strip=True)
        pay_text = tds[1].get_text(" ", strip=True)

        # 払戻金（数値列）を全部抜く
        pays = re.findall(r"\d[\d,]*", pay_text)
        pays = [int(p.replace(",", "")) for p in pays if p.replace(",", "").isdigit()]

        # 組み合わせ候補を複数抜く
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

        # 複数payがある場合は順に対応付け（合わなければcombo_textを使う）
        for i, pay100 in enumerate(pays):
            if i < len(combo_candidates):
                combo_norm = _norm_combo(bet_type, combo_candidates[i])
            else:
                combo_norm = _norm_combo(bet_type, combo_text)
            if pay100 and combo_norm:
                rows.append({"bet_type": bet_type, "combo": combo_norm, "pay100": int(pay100)})

    return rows

def parse_payouts_from_race_html(html: bytes, race_id: str) -> pd.DataFrame:
    """
    db.netkeiba.com/race/{race_id} のHTML(.bin)から払戻を抽出し、縦持ちDFで返す
    """
    soup = BeautifulSoup(html, "lxml")

    pay_tables = soup.find_all("table", class_=re.compile(r"pay_table|Pay_Table|pay_table_01", re.I))
    if not pay_tables:
        # 最後の手段："払戻" を含む table を拾う
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


# =========================
# results incremental (race_table_01)
# =========================
def create_results(
    html_path_list: List[Path],
    save_dir: Path = RAWDF_DIR,
    save_filename: str = "results.csv",
) -> pd.DataFrame:
    """
    race HTML(.bin)から results を作り、既存CSVに無い race_id だけ追記して保存する差分版
    ※ index=race_id で複数行（頭数分）が正常なので、index重複削除はしない
    """
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / save_filename

    # 既存ロード
    if save_path.exists() and save_path.stat().st_size > 0:
        base = pd.read_csv(save_path, sep="\t", index_col=0)
        base.index = base.index.astype(str)
        existing_race_ids = set(base.index.unique())
    else:
        base = pd.DataFrame()
        existing_race_ids = set()

    dfs = {}

    for html_path in tqdm(html_path_list, desc="create_results_incremental"):
        race_id = str(html_path.stem)
        if race_id in existing_race_ids:
            continue

        try:
            html = html_path.read_bytes().replace(b"<diary_snap_cut>", b"").replace(b"</diary_snap_cut>", b"")
            soup_table = BeautifulSoup(html, "lxml").find("table", class_="race_table_01 nk_tb_common")
            if soup_table is None:
                print(f"table not found at {race_id}")
                continue

            df = pd.read_html(html)[0]

            # horse_id
            horse_id_list = []
            horse_a_list = soup_table.find_all("a", href=re.compile(r"^/horse/"))
            for a in horse_a_list:
                horse_id = re.findall(r"\d{10}", a["href"])[0]
                horse_id_list.append(horse_id)
            df["horse_id"] = horse_id_list

            # jockey_id
            jockey_id_list = []
            jockey_a_list = soup_table.find_all("a", href=re.compile(r"^/jockey/"))
            for a in jockey_a_list:
                jockey_id = re.findall(r"\d{5}", a["href"])[0]
                jockey_id_list.append(jockey_id)
            df["jockey_id"] = jockey_id_list

            # trainer_id
            trainer_id_list = []
            trainer_a_list = soup_table.find_all("a", href=re.compile(r"^/trainer/"))
            for a in trainer_a_list:
                trainer_id = re.findall(r"\d{5}", a["href"])[0]
                trainer_id_list.append(trainer_id)
            df["trainer_id"] = trainer_id_list

            # owner_id
            owner_id_list = []
            owner_a_list = soup_table.find_all("a", href=re.compile(r"^/owner/"))
            for a in owner_a_list:
                owner_id = re.findall(r"\d{6}", a["href"])[0]
                owner_id_list.append(owner_id)
            df["owner_id"] = owner_id_list

            df.index = [race_id] * len(df)
            dfs[race_id] = df

        except Exception as e:
            print(f"failed at {race_id}: {e}")
            continue

    if not dfs:
        return base

    add = pd.concat(dfs.values(), axis=0)
    add.index.name = "race_id"
    add.columns = add.columns.astype(str).str.replace(" ", "")

    merged = pd.concat([base, add], axis=0) if len(base) else add
    merged.index = merged.index.astype(str)

    merged.to_csv(save_path, sep="\t")
    return merged


# =========================
# horse_results incremental (ajax)
# =========================
def _extract_race_ids_from_html_fragment(html_fragment: str) -> List[str]:
    soup = BeautifulSoup(html_fragment, "lxml")
    race_ids = []
    for a in soup.find_all("a", href=True):
        m = re.search(r"/race/(\d{12})", a["href"])
        if m:
            race_ids.append(m.group(1))
    return race_ids

def create_horse_results(
    html_path_list: List[Path],
    save_dir: Path = RAWDF_DIR,
    save_filename: str = "horse_results.csv",
    sleep_sec: float = 1.2,
) -> pd.DataFrame:
    """
    horse_id ごとに ajax API から戦績を取得し、
    既存CSVとの差分（日付基準）だけを追加保存する。
    """
    save_path = save_dir / save_filename
    save_dir.mkdir(parents=True, exist_ok=True)

    if save_path.exists() and save_path.stat().st_size > 0:
        base_df = pd.read_csv(save_path, sep="\t", index_col=0)
        base_df.index = base_df.index.astype(str)
        base_df["日付"] = pd.to_datetime(base_df["日付"], errors="coerce")

        last_date_map = base_df.groupby(level=0)["日付"].max().to_dict()
    else:
        base_df = pd.DataFrame()
        last_date_map = {}

    session = requests.Session()
    dfs = {}

    for html_path in tqdm(html_path_list, desc="create_horse_results"):
        horse_id = str(html_path.stem)

        time.sleep(sleep_sec + random.random() * 1.5)

        url = (
            "https://db.netkeiba.com/horse/"
            "ajax_horse_results.html"
            f"?input=UTF-8&output=json&id={horse_id}"
        )

        headers = {
            "User-Agent": random.choice(USER_AGENTS),
            "Referer": f"https://db.netkeiba.com/horse/{horse_id}/",
        }

        try:
            r = session.get(url, headers=headers, timeout=30)
            if r.status_code != 200:
                print(f"[HTTP {r.status_code}] {horse_id}")
                time.sleep(5 + random.random() * 5)
                continue

            j = r.json()
            if j.get("status") != "OK":
                print(f"[AJAX NG] {horse_id}")
                continue

            html_fragment = j.get("data", "")
            if not html_fragment.strip():
                print(f"[EMPTY] {horse_id}")
                continue

            tables = pd.read_html(html_fragment)
            if not tables:
                print(f"[NO TABLE] {horse_id}")
                continue

            df = tables[0].copy()
            df["日付"] = pd.to_datetime(df["日付"], errors="coerce")

            race_ids = _extract_race_ids_from_html_fragment(html_fragment)
            if len(race_ids) >= len(df):
                df["race_id"] = race_ids[:len(df)]
            else:
                df["race_id"] = race_ids + [None] * (len(df) - len(race_ids))

            last_date = last_date_map.get(horse_id)
            if last_date is not None:
                df = df[df["日付"] > last_date]

            if df.empty:
                continue

            df.index = [horse_id] * len(df)
            dfs[horse_id] = df

        except Exception as e:
            print(f"[FAIL] {horse_id}: {e}")
            time.sleep(10 + random.random() * 10)
            continue

    if dfs:
        add_df = pd.concat(dfs.values(), axis=0)
        add_df.index.name = "horse_id"
        merged = pd.concat([base_df, add_df], axis=0) if len(base_df) else add_df
    else:
        merged = base_df

    if merged.empty:
        print("No update.")
        return merged.reset_index()

    merged = merged.reset_index()
    merged.columns = merged.columns.str.replace(" ", "")
    merged = merged.drop_duplicates(
        subset=["horse_id", "日付", "レース名", "距離"],
        keep="last",
    )
    merged = merged.sort_values(["horse_id", "日付"])
    merged.set_index("horse_id").to_csv(save_path, sep="\t")

    return merged.reset_index()


# =========================
# race_info + payouts_flat incremental
# =========================
def create_race_info(
    html_path_list: List[Path],
    save_dir: Path = RAWDF_DIR,
    race_info_filename: str = "race_info.csv",
    payouts_filename: str = "payouts_flat.csv",
    force_update_race_info: bool = False,
    force_update_payouts: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    race HTML(.bin)から
      - race_info.csv（1行/レース）
      - payouts_flat.csv（複数行/レース、複勝/ワイド等も複数行）
    を差分更新で作る。

    payoutsは「券種が揃ってないレース」は次回も再取得して埋められる仕様。
    """
    save_dir.mkdir(parents=True, exist_ok=True)
    race_info_path = save_dir / race_info_filename
    payouts_path = save_dir / payouts_filename

    # --- load base_info ---
    if race_info_path.exists() and race_info_path.stat().st_size > 0:
        base_info = pd.read_csv(race_info_path, sep="\t", dtype={"race_id": str})
        existing_info = set(base_info["race_id"].astype(str).unique())
    else:
        base_info = pd.DataFrame(columns=["race_id", "title", "info1", "info2", "around_dir", "around_inout", "course_abc"])
        existing_info = set()

    # --- load base_pay ---
    if payouts_path.exists() and payouts_path.stat().st_size > 0:
        base_pay = pd.read_csv(payouts_path, sep="\t", dtype={"race_id": str, "bet_type": str, "combo": str})
    else:
        base_pay = pd.DataFrame(columns=["race_id", "bet_type", "combo", "pay100"])

    # race_id -> set(bet_type)
    race_to_types: Dict[str, set] = {}
    if len(base_pay) and "race_id" in base_pay.columns and "bet_type" in base_pay.columns:
        race_to_types = base_pay.groupby("race_id")["bet_type"].apply(lambda s: set(map(str, s))).to_dict()

    add_info_rows = []
    add_pay_dfs = []

    for html_path in tqdm(html_path_list, desc="create_race_info_and_payouts_incremental"):
        race_id = str(html_path.stem)

        need_info = force_update_race_info or (race_id not in existing_info)

        # ★ここが肝：券種が揃ってないレースは埋め直す
        types_have = race_to_types.get(race_id, set())
        need_pay = force_update_payouts or (len(REQUIRED_BET_TYPES - types_have) > 0)

        if not need_info and not need_pay:
            continue

        try:
            html = html_path.read_bytes()
            soup = BeautifulSoup(html, "lxml")

            # ---- race_info ----
            if need_info:
                intro = soup.find("div", class_="data_intro")
                if intro is None:
                    # たまに構造違いがある
                    # print(f"data_intro not found at {race_id}")
                    pass
                else:
                    h1 = intro.find("h1")
                    p_list = intro.find_all("p")
                    if len(p_list) >= 2:
                        info1_text = p_list[0].get_text(" ", strip=True)
                        info2_text = p_list[1].get_text(" ", strip=True)
                        around_dir, around_inout, course_abc = parse_course_meta_from_info1(info1_text)

                        add_info_rows.append({
                            "race_id": race_id,
                            "title": h1.get_text(strip=True) if h1 else "",
                            "info1": info1_text,
                            "info2": info2_text,
                            "around_dir": around_dir,
                            "around_inout": around_inout,
                            "course_abc": course_abc,
                        })

            # ---- payouts_flat ----
            if need_pay:
                dfp = parse_payouts_from_race_html(html, race_id=race_id)
                if len(dfp):
                    add_pay_dfs.append(dfp)

        except Exception as e:
            print(f"failed at {race_id}: {e}")
            continue

    # ---- merge race_info ----
    if add_info_rows:
        add_info = pd.DataFrame(add_info_rows)
        for c in ["around_dir", "around_inout", "course_abc"]:
            add_info[c] = pd.to_numeric(add_info[c], errors="coerce").fillna(-1).astype("int32")

        merged_info = pd.concat([base_info, add_info], ignore_index=True)
        merged_info = merged_info.drop_duplicates(subset=["race_id"], keep="last")
        merged_info = merged_info.sort_values(["race_id"])
        merged_info.to_csv(race_info_path, sep="\t", index=False)
    else:
        merged_info = base_info

    # ---- merge payouts ----
    if add_pay_dfs:
        add_pay = pd.concat(add_pay_dfs, ignore_index=True)
        add_pay["race_id"] = add_pay["race_id"].astype(str)
        add_pay["bet_type"] = add_pay["bet_type"].astype(str)
        add_pay["combo"] = add_pay["combo"].astype(str)
        add_pay["pay100"] = pd.to_numeric(add_pay["pay100"], errors="coerce").fillna(0).astype(int)
        add_pay = add_pay[add_pay["pay100"] > 0].copy()

        merged_pay = pd.concat([base_pay, add_pay], ignore_index=True)
        merged_pay = merged_pay.drop_duplicates(subset=["race_id", "bet_type", "combo"], keep="last")
        merged_pay = merged_pay.sort_values(["race_id", "bet_type", "combo"])
        merged_pay.to_csv(payouts_path, sep="\t", index=False)
    else:
        merged_pay = base_pay

    return merged_info, merged_pay

# def create_race_info(
#     html_path_list: list[Path],
#     save_dir: Path = RAWDF_DIR,
#     save_filename: str = "race_info.csv",
# ) -> pd.DataFrame:
#         """
#         db.netkeiba.com の race詳細HTMLから race_info(raw) を作る
#         - info1/info2 は生テキストで保存
#         - around_dir / around_inout / course_abc を追加
#         """
#         dfs: dict[str, pd.DataFrame] = {}

#         for html_path in tqdm(html_path_list):
#             race_id = str(html_path.stem)
#             try:
#                 with open(html_path, "rb") as f:
#                     html = f.read()

#                 intro = BeautifulSoup(html, "lxml").find("div", class_="data_intro")
#                 if intro is None:
#                     print(f"data_intro not found at {race_id}")
#                     continue

#                 h1 = intro.find("h1")
#                 p_list = intro.find_all("p")
#                 if len(p_list) < 2:
#                     print(f"p_list too short at {race_id}")
#                     continue

#                 info1_text = p_list[0].get_text(" ", strip=True)
#                 info2_text = p_list[1].get_text(" ", strip=True)

#                 around_dir, around_inout, course_abc = parse_course_meta_from_info1(info1_text)

#                 df = pd.DataFrame([{
#                     "race_id": race_id,
#                     "title": h1.get_text(strip=True) if h1 else "",
#                     "info1": info1_text,
#                     "info2": info2_text,
#                     "around_dir": around_dir,
#                     "around_inout": around_inout,
#                     "course_abc": course_abc,
#                 }])

#                 dfs[race_id] = df

#             except Exception as e:
#                 print(f"failed at {race_id}: {e}")
#                 continue

#         if not dfs:
#             cols = ["race_id", "title", "info1", "info2", "around_dir", "around_inout", "course_abc"]
#             out = pd.DataFrame(columns=cols)
#             save_dir.mkdir(parents=True, exist_ok=True)
#             out.to_csv(save_dir / save_filename, sep="\t", index=False)
#             return out

#         concat_df = pd.concat(dfs.values(), ignore_index=True)

#         for c in ["around_dir", "around_inout", "course_abc"]:
#             concat_df[c] = pd.to_numeric(concat_df[c], errors="coerce").fillna(-1).astype("int32")

#         save_dir.mkdir(parents=True, exist_ok=True)
#         concat_df.to_csv(save_dir / save_filename, sep="\t", index=False)
#         return concat_df