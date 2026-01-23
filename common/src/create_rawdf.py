import pandas as pd
from pathlib import Path
from tqdm.notebook import tqdm
from bs4 import BeautifulSoup
import re
import requests
import time
import random

around_mapping = {"右":0, "左":1, "直":2}
inout_mapping = {"内":0, "外":1}
course_abc_mapping = {"A":0, "B":1, "C":2}

def parse_course_meta_from_info1(info1_text: str):
    t = re.sub(r"\s+", "", str(info1_text))
    m = re.search(r"(右|左|直)", t)
    around_dir = around_mapping.get(m.group(1), None) if m else None
    m = re.search(r"(内|外)", t)
    around_inout = inout_mapping.get(m.group(1), None) if m else None
    m = re.search(r"(A|B|C)", t)
    course_abc = course_abc_mapping.get(m.group(1), None) if m else None
    return around_dir, around_inout, course_abc


RAWDF_DIR = Path("..", "data", "rawdf")
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:115.0) Gecko/20100101 Firefox/115.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:115.0) Gecko/20100101 Firefox/115.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.2 Safari/605.1.15",
]


def create_results(html_path_list: list[Path], save_dir: Path = RAWDF_DIR, save_filename: str = "results.csv") -> pd.DataFrame:
    """
    raceページのhtmlを読み込んで、レース結果テーブルに加工する関数
    """
    dfs = {}
    for html_path in tqdm(html_path_list):
        with open(html_path, "rb") as f:
            try:
                race_id = html_path.stem
                html = f.read().replace(b"<diary_snap_cut>", b"").replace(b"<diary_snap_cut>", b"")
                soup = BeautifulSoup(html, "lxml").find(
                    "table", class_="race_table_01 nk_tb_common"
                )
                df = pd.read_html(html)[0]

                # horse_id列追加
                horse_id_list = []
                horse_a_list = soup.find_all("a", href=re.compile(r"^/horse/"))
                for a in horse_a_list:
                    horse_id = re.findall(r"\d{10}", a["href"])[0]
                    horse_id_list.append(horse_id)
                df["horse_id"] = horse_id_list

                # jockey_id列追加
                jockey_id_list = []
                jockey_a_list = soup.find_all("a", href=re.compile(r"^/jockey/"))
                jockey_id_list = []
                for a in jockey_a_list:
                    jockey_id = re.findall(r"\d{5}", a["href"])[0]
                    jockey_id_list.append(jockey_id)
                df["jockey_id"] = jockey_id_list

                # trainer_id列追加
                trainer_id_list = []
                trainer_a_list = soup.find_all("a", href=re.compile(r"^/trainer/"))
                trainer_id_list = []
                for a in trainer_a_list:
                    trainer_id = re.findall(r"\d{5}", a["href"])[0]
                    trainer_id_list.append(trainer_id)
                df["trainer_id"] = trainer_id_list

                # owner_id列追加
                owner_id_list = []
                owner_a_list = soup.find_all("a", href=re.compile(r"^/owner/"))
                owner_id_list = []
                for a in owner_a_list:
                    owner_id = re.findall(r"\d{6}", a["href"])[0]
                    owner_id_list.append(owner_id)
                df["owner_id"] = owner_id_list

                df.index = [race_id]*len(df)
                dfs[race_id] = df
            except IndexError as e:
                print(f"table not found at {race_id}")
                continue
    concat_df = pd.concat(dfs.values())
    concat_df.index.name = "race_id"
    concat_df.columns = concat_df.columns.str.replace(" ", "")
    save_dir.mkdir(parents=True, exist_ok=True)
    concat_df.to_csv(save_dir/save_filename, sep="\t")
    return concat_df

def _extract_race_ids_from_html_fragment(html_fragment: str) -> list[str]:
    soup = BeautifulSoup(html_fragment, "lxml")
    race_ids = []
    for a in soup.find_all("a", href=True):
        m = re.search(r"/race/(\d{12})", a["href"])
        if m:
            race_ids.append(m.group(1))
    return race_ids


def create_horse_results(
    html_path_list: list[Path],
    save_filename: str = "horse_results.csv",
    sleep_sec: float = 1.2,
) -> pd.DataFrame:
    """
    horse_id ごとに ajax API から戦績を取得し、
    既存CSVとの差分（日付基準）だけを追加保存する。
    """

    save_dir = RAWDF_DIR
    save_path = save_dir / save_filename
    save_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------
    # 1. 既存CSVロード
    # -------------------------
    if save_path.exists():
        base_df = pd.read_csv(save_path, sep="\t", index_col=0)
        base_df.index = base_df.index.astype(str)
        base_df["日付"] = pd.to_datetime(base_df["日付"], errors="coerce")

        last_date_map = (
            base_df.groupby(level=0)["日付"]
            .max()
            .to_dict()
        )
    else:
        base_df = pd.DataFrame()
        last_date_map = {}

    session = requests.Session()
    dfs = {}

    # -------------------------
    # 2. 差分取得
    # -------------------------
    for html_path in tqdm(html_path_list):
        horse_id = str(html_path.stem)

        # request 前 sleep（規則性を壊す）
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

            # 明示的に status チェック
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

            # ★追加：race_id を html_fragment 内のリンクから抽出して付与
            race_ids = _extract_race_ids_from_html_fragment(html_fragment)

            # 表の行数に合わせて調整（多すぎ/少なすぎ対策）
            if len(race_ids) >= len(df):
                df["race_id"] = race_ids[:len(df)]
            else:
                df["race_id"] = race_ids + [None] * (len(df) - len(race_ids))

            last_date = last_date_map.get(horse_id)
            if last_date is not None:
                df = df[df["日付"] > last_date]

            if df.empty:
                print(f"[SKIP NO UPDATE] {horse_id}")
                continue

            df.index = [horse_id] * len(df)
            dfs[horse_id] = df

        except Exception as e:
            print(f"[FAIL] {horse_id}: {e}")
            # 失敗時は長めに待つ
            time.sleep(10 + random.random() * 10)
            continue

    # -------------------------
    # 3. 結合
    # -------------------------
    if dfs:
        add_df = pd.concat(dfs.values(), axis=0)
        add_df.index.name = "horse_id"

        merged = pd.concat([base_df, add_df], axis=0) if len(base_df) else add_df
    else:
        merged = base_df

    if merged.empty:
        print("No update.")
        return merged.reset_index()

    # -------------------------
    # 4. 整形 & 保存
    # -------------------------
    merged = merged.reset_index()
    merged.columns = merged.columns.str.replace(" ", "")
    merged = merged.drop_duplicates(
        subset=["horse_id", "日付", "レース名", "距離"],
        keep="last",
    )
    merged = merged.sort_values(["horse_id", "日付"])

    merged.set_index("horse_id").to_csv(save_path, sep="\t")

    return merged.reset_index()


def create_race_info(
    html_path_list: list[Path],
    save_dir: Path = RAWDF_DIR,
    save_filename: str = "race_info.csv",
) -> pd.DataFrame:
        """
        db.netkeiba.com の race詳細HTMLから race_info(raw) を作る
        - info1/info2 は生テキストで保存
        - around_dir / around_inout / course_abc を追加
        """
        dfs: dict[str, pd.DataFrame] = {}

        for html_path in tqdm(html_path_list):
            race_id = str(html_path.stem)
            try:
                with open(html_path, "rb") as f:
                    html = f.read()

                intro = BeautifulSoup(html, "lxml").find("div", class_="data_intro")
                if intro is None:
                    print(f"data_intro not found at {race_id}")
                    continue

                h1 = intro.find("h1")
                p_list = intro.find_all("p")
                if len(p_list) < 2:
                    print(f"p_list too short at {race_id}")
                    continue

                info1_text = p_list[0].get_text(" ", strip=True)
                info2_text = p_list[1].get_text(" ", strip=True)

                around_dir, around_inout, course_abc = parse_course_meta_from_info1(info1_text)

                df = pd.DataFrame([{
                    "race_id": race_id,
                    "title": h1.get_text(strip=True) if h1 else "",
                    "info1": info1_text,
                    "info2": info2_text,
                    "around_dir": around_dir,
                    "around_inout": around_inout,
                    "course_abc": course_abc,
                }])

                dfs[race_id] = df

            except Exception as e:
                print(f"failed at {race_id}: {e}")
                continue

        if not dfs:
            cols = ["race_id", "title", "info1", "info2", "around_dir", "around_inout", "course_abc"]
            out = pd.DataFrame(columns=cols)
            save_dir.mkdir(parents=True, exist_ok=True)
            out.to_csv(save_dir / save_filename, sep="\t", index=False)
            return out

        concat_df = pd.concat(dfs.values(), ignore_index=True)

        for c in ["around_dir", "around_inout", "course_abc"]:
            concat_df[c] = pd.to_numeric(concat_df[c], errors="coerce").fillna(-1).astype("int32")

        save_dir.mkdir(parents=True, exist_ok=True)
        concat_df.to_csv(save_dir / save_filename, sep="\t", index=False)
        return concat_df