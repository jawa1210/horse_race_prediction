from pathlib import Path
import pandas as pd
from bs4 import BeautifulSoup
import time
from tqdm.notebook import tqdm
from urllib.request import urlopen, Request
import re
import numpy as np
import json
import requests

DATA_DIR = Path("..", "data")
RAW_DATA_DIR = Path("..", "..", "common", "data")
POPULATION_INPUT_DIR = RAW_DATA_DIR/"prediction_population"
INPUT_DIR = DATA_DIR/"01_preprocessed"
OUTPUT_DIR = DATA_DIR/"02_features"
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
HTML_DIR = Path("..", "data", "html",)
HTML_RACE_DIR = HTML_DIR/"race"
HTML_HORSE_DIR = HTML_DIR/"horse"

sex_mapping = {"牡": 0, "牝": 1, "セ": 2}
weather_mapping = {"晴": 0, "曇": 1, "小雨": 2, "雨": 3, "雪": 4, "小雪": 5, "nan": 6}
race_type_mapping = {"ダ": 0, "芝": 1, "障": 2}
race_class_mapping = {
    "新馬": 0,
    "未勝利": 1,
    "1勝クラス": 2,
    "2勝クラス": 3,
    "3勝クラス": 4,
    "オープン": 5,
    "G3": 6,
    "G2": 7,
    "G1": 8,
    "OP": 5,
    "特別": 5,
    "500万下": 2,
    "1000万下": 3,
    "1600万下": 4,
    "C3": 9,
    "C2": 10,
    "C1": 11,
    "B3": 12,
    "B2": 13,
    "B1": 14,
    "A3": 15,
    "A2": 16,
    "A1": 17,
    "重賞": 18
}
ground_state_mapping = {
    "良": 0,
    "重": 1,
    "稍重": 2,
    "不良": 3,
    "稍": 2,
    "不": 3
}
around_mapping = {
    "右": 0,
    "左": 1,
    "直": 2
}

race_course_mapping = {
    "東京": 0, "阪神": 1, "中山": 2, "京都": 3, "中京": 4, "札幌": 5, "函館": 6, "新潟": 7,
    "小倉": 8, "福島": 9, "函館": 10, "盛岡": 11, "水沢": 12, "金沢": 13, "高知": 14, "大井": 15, "川崎": 16, "浦和": 17, "船橋": 18, "名古屋": 19, "笠松": 20, "園田": 21, "姫路": 22, "門別": 23, "帯広": 24
}

import re
import json
import base64
import zlib
import requests
import pandas as pd
from bs4 import BeautifulSoup

def _loads_json_or_jsonp(text: str):
    text = (text or "").strip()
    if not text:
        raise ValueError("empty response text")
    m = re.match(r'^[\w$]+\((.*)\)\s*;?\s*$', text, flags=re.DOTALL)
    if m:
        text = m.group(1).strip()
    return json.loads(text)

def _decompress_netkeiba_data(data_str: str):
    raw = base64.b64decode(data_str)
    try:
        s = zlib.decompress(raw).decode("utf-8", errors="ignore")
    except zlib.error:
        s = zlib.decompress(raw, -zlib.MAX_WBITS).decode("utf-8", errors="ignore")
    s = s.strip()
    # 展開結果が JSON の場合
    try:
        return json.loads(s)
    except Exception:
        return s  # HTML/文字列の可能性もあるのでそのまま返す

def _print_keys(prefix, obj, max_items=30):
    if isinstance(obj, dict):
        print(prefix, "dict keys:", list(obj.keys())[:max_items])
    elif isinstance(obj, list):
        print(prefix, "list len:", len(obj), "first:", str(obj[0])[:120] if obj else None)
    else:
        print(prefix, "type:", type(obj), "head:", str(obj)[:200])

def fetch_odds_ninki_map_via_api(race_id: str, session: requests.Session | None = None, timeout: int = 10):
    sess = session or requests.Session()
    sess.headers.update({
        "User-Agent": "Mozilla/5.0",
        "Referer": "https://race.netkeiba.com/",
    })

    # Cookie付与（403回避に効くことがある）
    try:
        sess.get(f"https://race.netkeiba.com/race/shutuba.html?race_id={race_id}", timeout=timeout)
    except Exception:
        pass

    url = "https://race.netkeiba.com/api/api_get_jra_odds.html"
    params = {
        "pid": "api_get_jra_odds",
        "input": "UTF-8",
        "output": "jsonp",
        "callback": "cb",
        "race_id": race_id,
        "type": 1,          # 単勝
        "action": "init",
        "sort": "odds",
        "compress": 1,      # 圧縮されて返る
        "_": 1,
    }

    r = sess.get(url, params=params, timeout=timeout)
    r.raise_for_status()

    j = _loads_json_or_jsonp(r.text)

    # 圧縮データを展開 → payload = {"official_datetime":..., "odds": {...}} になる
    payload = j
    if isinstance(j, dict) and isinstance(j.get("data"), str) and j["data"].startswith("eN"):
        payload = _decompress_netkeiba_data(j["data"])

    # ここで確定パース（"01": ["33.5", None, 10] を読む）
    umaban_to_odds, umaban_to_ninki = parse_tansho_and_ninki_from_payload(payload)

    return umaban_to_odds, umaban_to_ninki


def parse_tansho_and_ninki_from_payload(payload: dict) -> tuple[dict[int, float], dict[int, int]]:
    """
    payload['odds']['1'] が {"01": ["33.5", None, 10], ...} の形なので、
    umaban(int) -> odds(float), umaban(int) -> ninki(int) を返す
    """
    umaban_to_odds: dict[int, float] = {}
    umaban_to_ninki: dict[int, int] = {}

    if not isinstance(payload, dict):
        return umaban_to_odds, umaban_to_ninki
    odds_root = payload.get("odds")
    if not isinstance(odds_root, dict):
        return umaban_to_odds, umaban_to_ninki

    tansho = odds_root.get("1")  # 単勝
    if not isinstance(tansho, dict):
        return umaban_to_odds, umaban_to_ninki

    for k, v in tansho.items():
        # k: "01" など
        try:
            u = int(str(k))
        except:
            continue

        # v: ["33.5", None, 10] のような list/tuple
        if isinstance(v, (list, tuple)) and len(v) >= 1:
            o = v[0]
            if o not in (None, "", "-", "--", "---.-"):
                try:
                    umaban_to_odds[u] = float(str(o).replace(",", ""))
                except:
                    pass

            # 人気は3番目に入っている（len>=3 の場合）
            if len(v) >= 3:
                n = v[2]
                if n not in (None, "", "*", "**", "***", "****", "-", "--"):
                    try:
                        umaban_to_ninki[u] = int(str(n))
                    except:
                        pass

    return umaban_to_odds, umaban_to_ninki



def scrape_html_target_race(race_id: str):
    """
    netkeiba.comのraceページのhtmlをスクレイピングして、htmlを取得する関数
    """
    url = f"https://race.netkeiba.com/race/shutuba.html?race_id={race_id}"
    headers = {"User-Agent": "Mozilla/5.0"}
    request = Request(url, headers=headers)
    html = urlopen(request).read()
    time.sleep(1)
    return html


def create_race_info(html) -> pd.DataFrame:
    soup = BeautifulSoup(html, "lxml")

    # 予測(出馬表)と学習(レース詳細)でDOMが違うので、候補を複数探 demonstrated
    intro = soup.find("div", class_="data_intro")
    if intro is None:
        # shutubaページ側でよくある候補（サイト改修で変わりうる）
        intro = soup.find("div", class_="RaceData01")
    if intro is None:
        intro = soup.find("div", class_="RaceData02")

    if intro is None:
        # デバッグ用：titleだけでも見て状況を掴む
        title_tag = soup.find("title")
        title_txt = title_tag.get_text(strip=True) if title_tag else "(no title)"
        raise ValueError(f"race info block not found. page title={title_txt}")

    info_dict = {}
    h1 = intro.find("h1") or soup.find("h1")
    info_dict["title"] = h1.get_text(strip=True) if h1 else ""

    p_list = intro.find_all("p")
    # pが無い場合もあるのでガード
    p0 = p_list[0].get_text(" ", strip=True) if len(p_list) > 0 else ""
    p1 = p_list[1].get_text(" ", strip=True) if len(p_list) > 1 else ""

    info_dict["info1"] = [p0]
    info_dict["info2"] = [p1]

    df = pd.DataFrame(info_dict)
    # ここでは race_id がHTMLにないので呼び出し元で付けるほうが安全
    return df

def scrape_html_db_race(race_id: str):
    """
    db.netkeiba.com の race 詳細ページHTMLを取得（学習時と同じDOMが期待できる）
    """
    url = f"https://db.netkeiba.com/race/{race_id}"
    headers = {"User-Agent": "Mozilla/5.0"}
    request = Request(url, headers=headers)
    html = urlopen(request).read()
    time.sleep(1)
    return html



class FeatureCreator:
    def __init__(
            self,
            results_filepath: Path = INPUT_DIR/"results.csv",
            race_info_filepath: Path = INPUT_DIR/"race_info.csv",
            horse_results_filepath: Path = INPUT_DIR/"horse_results.csv",
            output_dir: Path = OUTPUT_DIR,
    ):
        self.results = pd.read_csv(results_filepath, sep="\t")
        self.race_info = pd.read_csv(race_info_filepath, sep="\t")
        self.horse_results = pd.read_csv(horse_results_filepath, sep="\t")
        self.output_dir = output_dir
        # 学習母集団の作成
        self.population = self.results[["race_id", "horse_id"]].merge(
            self.race_info[["race_id", "date"]], on="race_id"
        )

    def agg_horse_n_races(self, n_races: list[int] = [3, 5, 10, 1000]):
        """
        直近nレースの着順と賞金の平均を集計する関数
        """
        grouped_df = (
            self.population.merge(
                self.horse_results, on=["horse_id"], suffixes=("", "_horse")
            )
            .query("date>date_horse")
            .sort_values("date_horse", ascending=False)
            .groupby(["race_id", "horse_id"])
        )
        merged_df = self.population.copy()
        for n_race in n_races:
            df = (
                grouped_df.head(n_race)
                .groupby(["race_id", "horse_id"])[["rank", "prize"]]
                .mean()
            ).add_suffix(f"_{n_race}races")
            merged_df = merged_df.merge(
                df,
                on=["race_id", "horse_id"],
            )
        self.agg_horse_n_races_df = merged_df

    def create_features(self):
        """
        特徴量作成処理を実行し、populationテーブルにすべての特徴量を結合する
        """
        self.agg_horse_n_races()
        features = (
            self.population.merge(self.results, on=["race_id", "horse_id"])
            .merge(self.race_info, on=["race_id", "date"])
            .merge(
                self.agg_horse_n_races_df,
                on=["race_id", "date", "horse_id"],
                how="left",
            )
        )
        features.to_csv(self.output_dir/"features.csv", sep="\t", index=None)
        return features


class PredictionFeatureCreator:
    def __init__(
        self,
        population_filepath: Path = POPULATION_INPUT_DIR/"population.csv",
        horse_results_prediction_feilepath: Path = INPUT_DIR/"horse_results_prediction.csv",
        output_dir: Path = OUTPUT_DIR,
    ):
        self.population = pd.read_csv(population_filepath, sep="\t")
        self.horse_results = pd.read_csv(horse_results_prediction_feilepath, sep="\t")
        self.output_dir = output_dir     # ★追加
        self.output_dir.mkdir(exist_ok=True, parents=True)  # ★ついでに安全

        # ★キーを文字列に統一（前の修正が入ってるならここも）
        self.population["race_id"] = self.population["race_id"].astype(str)
        self.population["horse_id"] = self.population["horse_id"].astype(str)
        self.horse_results["horse_id"] = self.horse_results["horse_id"].astype(str)

    def agg_horse_n_races(self, n_races: list[int] = [3, 5, 10, 1000]):
        """
        直近nレースの着順と賞金の平均を集計する関数
        """
        grouped_df = (
            self.population.merge(
                self.horse_results, on=["horse_id"], suffixes=("", "_horse")
            )
            .query("date > date_horse")
            .sort_values("date_horse", ascending=False)
            .groupby(["race_id", "horse_id"])
        )
        merged_df = self.population.copy()
        for n_race in n_races:
            df = (
                grouped_df.head(n_race)
                .groupby(["race_id", "horse_id"])[["rank", "prize"]]
                .mean()
            ).add_suffix(f"_{n_race}races")
            merged_df = merged_df.merge(
                df,
                on=["race_id", "horse_id"],
            )
        self.agg_horse_n_races_df = merged_df

    def fetch_syutuba_table_html(self, race_id: str) -> str:
        """
        レースidを指定すると、出馬ページのhtmlをスクレイピングする関数
        """
        html = scrape_html_target_race(race_id)  # 出馬ページをスクレイピング
        self.html = html

    def fetch_results(self, predict: bool = False) -> pd.DataFrame:
        soup_table = BeautifulSoup(self.html, "lxml").find(
            "table",
            class_=[
                "Shutuba_Table", "RaceTable01", "ShutubaTable",
                "tablesorter", "tablesorter-default"
            ],
        )
        if soup_table is None:
            raise ValueError("出馬表のテーブルが見つかりません。HTML構造が変わった可能性があります。")

        df = pd.read_html(self.html)[0]

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(-1)

        df.columns = df.columns.astype(str).str.strip()
        df.columns = df.columns.str.replace("\n", "", regex=True)

        expected_cols = ["性齢", "馬体重 (増減)", "斤量", "枠", "馬 番"]
        for col in expected_cols:
            if col not in df.columns:
                print(f"Warning: '{col}' が DataFrame に存在しません（出馬表の列構造が異なる可能性）")

        df["race_id"] = str(self.race_id)

        n = len(df)

        def _pad_or_trim(values: list, n: int, fill=None) -> list:
            return (values[:n] + [fill] * n)[:n]

        # ------------------------------------------------------------
        # 1) horse_id / jockey_id / trainer_id
        # ------------------------------------------------------------
        horse_id_list = []
        for a in soup_table.find_all("a", href=re.compile(r"/horse/\d{10}")):
            m = re.findall(r"\d{10}", a.get("href", ""))
            if m:
                horse_id_list.append(m[0])
        df["horse_id"] = _pad_or_trim(horse_id_list, n, fill=None)

        jockey_id_list = []
        for a in soup_table.find_all("a", href=re.compile(r"/jockey/result/recent/\d{5}")):
            m = re.findall(r"\d{5}", a.get("href", ""))
            if m:
                jockey_id_list.append(m[0])
        df["jockey_id"] = _pad_or_trim(jockey_id_list, n, fill=None)

        trainer_id_list = []
        for a in soup_table.find_all("a", href=re.compile(r"/trainer/result/recent/\d{5}")):
            m = re.findall(r"\d{5}", a.get("href", ""))
            if m:
                trainer_id_list.append(m[0])
        df["trainer_id"] = _pad_or_trim(trainer_id_list, n, fill=None)

        # ------------------------------------------------------------
        # 2) 単勝オッズ / 人気
        # ------------------------------------------------------------
        if not predict:
            # これまで通り（ただし行ズレ注意）
            tansyo_odds_list = []
            popularity_list = []

            for tr in soup_table.find_all("tr"):
                tds = tr.find_all("td")
                if not tds:
                    continue

                odds_td = tr.find("td", class_=re.compile(r"Popular")) or tr.find("td", class_=re.compile(r"Txt_R"))
                odds_val = None
                if odds_td:
                    sp = odds_td.find("span")
                    if sp:
                        txt = sp.get_text(strip=True)
                        odds_val = txt if txt != "" else None

                pop_td = tr.find("td", class_=re.compile(r"Popular_Ninki")) or tr.find("td", class_=re.compile(r"Ninki"))
                pop_val = None
                if pop_td:
                    sp = pop_td.find("span")
                    if sp:
                        txt = sp.get_text(strip=True)
                        pop_val = txt if txt != "" else None

                tansyo_odds_list.append(odds_val)
                popularity_list.append(pop_val)

            tansyo_odds_list = _pad_or_trim(tansyo_odds_list, n, fill=None)
            popularity_list = _pad_or_trim(popularity_list, n, fill=None)

            df["tansyo_odds"] = pd.to_numeric(tansyo_odds_list, errors="coerce")
            df["popularity"]  = pd.to_numeric(popularity_list, errors="coerce")

        else:
            # ★必ず先に列を用意（KeyError防止）
            df["tansyo_odds"] = np.nan
            df["popularity"]  = np.nan

            # ここでAPIから埋める
            umaban_to_odds, umaban_to_ninki = fetch_odds_ninki_map_via_api(str(self.race_id))

            waku = pd.to_numeric(df["枠"], errors="coerce")
            uma  = pd.to_numeric(df["馬 番"], errors="coerce")

            fixed_odds = []
            fixed_ninki = []
            for u in uma:
                if pd.isna(u):
                    fixed_odds.append(None)
                    fixed_ninki.append(None)
                    continue
                u_i = int(u)
                fixed_odds.append(umaban_to_odds.get(u_i))
                fixed_ninki.append(umaban_to_ninki.get(u_i))

            df["tansyo_odds"] = pd.to_numeric(fixed_odds, errors="coerce")
            df["popularity"]  = pd.to_numeric(fixed_ninki, errors="coerce")

            print("[INFO] tansyo_odds via API:", int(df["tansyo_odds"].notna().sum()), "/", len(df))
            print("[INFO] popularity  via API:", int(df["popularity"].notna().sum()), "/", len(df))

            # 人気が取れない時だけ保険
            if df["popularity"].isna().all() and df["tansyo_odds"].notna().any():
                df["popularity"] = df["tansyo_odds"].rank(method="dense", ascending=True)


        # ------------------------------------------------------------
        # 3) 学習時と同じ前処理
        # ------------------------------------------------------------
        df["rank"] = np.nan

        df["性齢"] = df.get("性齢", "").astype(str)
        df["sex"] = df["性齢"].str[0].map(sex_mapping)
        df["age"] = pd.to_numeric(df["性齢"].str[1:], errors="coerce")

        bw_col = "馬体重 (増減)"
        if bw_col in df.columns:
            bw_str = df[bw_col].astype(str)
            df["weight"] = pd.to_numeric(bw_str.str.extract(r"(\d+)")[0], errors="coerce")
            df["weight_diff"] = pd.to_numeric(bw_str.str.extract(r"\(([-+]?\d+)\)")[0], errors="coerce")
        else:
            df["weight"] = np.nan
            df["weight_diff"] = np.nan

        df["impost"] = pd.to_numeric(df["斤量"], errors="coerce") if "斤量" in df.columns else np.nan
        df["wakuban"] = pd.to_numeric(df["枠"], errors="coerce") if "枠" in df.columns else np.nan
        df["umaban"]  = pd.to_numeric(df["馬 番"], errors="coerce") if "馬 番" in df.columns else np.nan

        df["agari"] = np.nan

        if "umaban" in df.columns:
            df = df.sort_values(["umaban"], na_position="last")

        use_cols = [
            "race_id","horse_id","jockey_id","trainer_id","rank",
            "wakuban","umaban","sex","age","weight","weight_diff",
            "tansyo_odds","popularity","impost","agari",
        ]
        for c in use_cols:
            if c not in df.columns:
                df[c] = np.nan
        df = df[use_cols]

        df["race_id"] = df["race_id"].astype(str)
        df["horse_id"] = df["horse_id"].astype(str)
        df["jockey_id"] = df["jockey_id"].astype(str)
        df["trainer_id"] = df["trainer_id"].astype(str)

        self.results = df
        return df




    # def fetch_race_info(self) -> pd.DataFrame:
    #     """
    #     出馬ページのhtmlを受け取ると、レース情報を取得し、
    #     学習時と同じ形式に前処理する関数
    #     """
    #     df_info = create_race_info(self.html)
    #     race_info = {}
    #     # info1からデータを抽出
    #     for index in range(len(df_info)):
    #         # 各行のrace_id取得
    #         race_id = df_info["race_id"][index]

    #         # 各行の "info1" 列に格納されている文字列に対して正規表現を使用
    #         info1_text = df_info["info1"][index]

    #         # "芝" または "ダート" が含まれている部分を正規表現で検索
    #         race_type = re.findall(r"[芝ダ障]+", info1_text)
    #         if race_type:
    #             race_type = race_type[0]
    #             # race_typeでマッピング
    #             race_type = race_type_mapping.get(race_type, None)
    #         else:
    #             race_type = None  # 一致しない場合は None に設定

    #         # 方向（右、左、直）を抽出
    #         around = re.findall(r"[右左直]+", info1_text)
    #         if around:
    #             around = around[0]
    #             # aroundでマッピング
    #             around = around_mapping.get(around, None)
    #         else:
    #             around = None  # 一致しない場合は None に設定

    #         # 天気を抽出（晴、曇、小雨、雨、小雪、雪）
    #         regex_weather = "|".join(weather_mapping.keys())
    #         weather = re.findall(rf"({regex_weather})", info1_text)
    #         if weather:
    #             weather = weather[0]  # 一致した最初のクラスを取得
    #             # weatherでマッピング
    #             weather = weather_mapping.get(weather, None)
    #         else:
    #             weather = None  # 一致しない場合は None

    #         # レース距離を取得
    #         course_len = int(re.findall(r"\d+", info1_text)[0])

    #         ground_state = re.findall(r"[良稍重不]+", info1_text)
    #         if ground_state:
    #             ground_state = ground_state[0]
    #             # ground_stateでマッピング
    #             ground_state = ground_state_mapping.get(ground_state, None)
    #         else:
    #             ground_state = None

    #         # info2からデータを抽出
    #         info2_text = df_info["info2"][index]
    #         # race_classを正規表現でマッチさせてマッピングする
    #         regex_race_class = "|".join(race_class_mapping.keys())
    #         race_class = re.findall(rf"({regex_race_class})", info2_text)
    #         if race_class:
    #             race_class = race_class[0]  # 一致した最初のクラスを取得
    #             # race_class_mappingでマッピング
    #             race_class = race_class_mapping.get(race_class, None)
    #         else:
    #             race_class = None  # 一致しない場合は None

    #         # race_courseを正規表現でマッチさせてマッピングする
    #         regex_race_course = "|".join(race_course_mapping.keys())
    #         race_course = re.findall(rf"({regex_race_course})", info2_text)
    #         if race_course:
    #             race_course = race_course[0]  # 一致した最初のクラスを取得
    #             # race_class_mappingでマッピング
    #             race_course = race_course_mapping.get(race_course, None)
    #         else:
    #             race_course = None  # 一致しない場合は None

    #         # 日付を抽出
    #         pattern = r"(\d{4})年(\d{1,2})月(\d{1,2})日"  # 年、月、日をキャプチャグループで抽出
    #         matches = re.findall(pattern, info2_text)

    #         # 結果をyyyy-mm-dd形式に変換
    #         date = [f"{year}-{int(month):02d}-{int(day):02d}" for year, month, day in matches][0]

    #         # レース情報をまとめる
    #         race_info.append({
    #             "race_id": race_id,
    #             "date": date,
    #             "race_type": race_type,
    #             "around": around,
    #             "course_len": course_len,
    #             "weather": weather,
    #             "ground_state": ground_state,
    #             "race_class": race_class,
    #             "place": race_course,
    #         })

    #     # 最終的なDataFrameを作成
    #     race_info = pd.DataFrame(race_info)
    #     self.race_info = race_info

    def fetch_race_info(self) -> pd.DataFrame:
        """
        レース情報を取得し、学習時と同じ形式に前処理する（予測用）
        - 出馬表ページではなく db.netkeiba.com/race/{race_id} を使う（DOMが安定）
        """
        # 1) db側raceページを取得
        html_db = scrape_html_db_race(self.race_id)

        # 2) data_intro から info1/info2 を作る（学習側と同じ取り方）
        df_info = create_race_info(html_db)  # <- あなたの create_race_info は info1/info2 を1行で返す想定
        # df_info は 1行想定なので 0行ならエラーにする
        if len(df_info) == 0:
            raise ValueError(f"race_info parse failed for race_id={self.race_id}")

        info1_text = str(df_info.loc[0, "info1"])
        info2_text = str(df_info.loc[0, "info2"])

        # 3) info1 から抽出
        race_type = re.findall(r"[芝ダ障]+", info1_text)
        race_type = race_type_mapping.get(race_type[0], None) if race_type else None

        around = re.findall(r"[右左直]+", info1_text)
        around = around_mapping.get(around[0], None) if around else None

        regex_weather = "|".join(weather_mapping.keys())
        weather = re.findall(rf"({regex_weather})", info1_text)
        weather = weather_mapping.get(weather[0], None) if weather else None

        course_len_m = re.findall(r"\d+", info1_text)
        course_len = int(course_len_m[0]) if course_len_m else None

        ground_state = re.findall(r"[良稍重不]+", info1_text)
        ground_state = ground_state_mapping.get(ground_state[0], None) if ground_state else None

        # 4) info2 から抽出
        regex_race_class = "|".join(race_class_mapping.keys())
        race_class = re.findall(rf"({regex_race_class})", info2_text)
        race_class = race_class_mapping.get(race_class[0], None) if race_class else None

        regex_race_course = "|".join(race_course_mapping.keys())
        race_course = re.findall(rf"({regex_race_course})", info2_text)
        place = race_course_mapping.get(race_course[0], None) if race_course else None

        # 日付
        pattern = r"(\d{4})年(\d{1,2})月(\d{1,2})日"
        matches = re.findall(pattern, info2_text)
        if matches:
            year, month, day = matches[0]
            date = f"{year}-{int(month):02d}-{int(day):02d}"
        else:
            # 最悪 population から引く（同race_idがpopulationにある想定）
            # それも無理なら None
            try:
                date = str(self.population.query("race_id == @self.race_id").iloc[0]["date"])
            except Exception:
                date = None

        # 5) 1行のDataFrameにする
        race_info = pd.DataFrame([{
            "race_id": self.race_id,
            "date": date,
            "race_type": race_type,
            "around": around,
            "course_len": course_len,
            "weather": weather,
            "ground_state": ground_state,
            "race_class": race_class,
            "place": place,
        }])

        self.race_info = race_info
        race_info["race_id"] = race_info["race_id"].astype(str)
        return race_info


    def create_features(self, race_id, predict=False, skip_agg_horse: bool = False) -> pd.DataFrame:
        """
        特徴量作成処理を実行し、populationテーブルにすべての特徴量を統合する
        先にagg_horse_n_races()を実行しておいた場合は、skip_agg_horse=Trueとすればスキップできる
        """
        # 馬の過去成績集計
        if not skip_agg_horse:
            self.agg_horse_n_races()

        self.predict=predict

        # 各種テーブルの取得
        self.race_id=race_id
        self.fetch_syutuba_table_html(race_id)  # race_idを使用
        self.fetch_results(self.predict)
        self.fetch_race_info()

        # 特徴量の統合
        # ★ キーのdtypeを最後にもう一回保険で合わせる
        for df in [self.population, self.results, self.race_info]:
            if "race_id" in df.columns:
                df["race_id"] = df["race_id"].astype(str)
            if "horse_id" in df.columns:
                df["horse_id"] = df["horse_id"].astype(str)

        features = (
            self.population.merge(self.results, on=["race_id", "horse_id"], how="inner")
            .merge(self.race_info, on=["race_id", "date"], how="left")
        )

        # aggは作ったときだけ結合
        if hasattr(self, "agg_horse_n_races_df"):
            self.agg_horse_n_races_df["race_id"] = self.agg_horse_n_races_df["race_id"].astype(str)
            self.agg_horse_n_races_df["horse_id"] = self.agg_horse_n_races_df["horse_id"].astype(str)

            features = features.merge(
                self.agg_horse_n_races_df,
                on=["race_id", "date", "horse_id"],
                how="left",
            )

        features.to_csv(self.output_dir / f"prediction_features_{race_id}.csv", sep="\t", index=None)
        return features
    
    def debug(self):
        print("odds- in html?", "odds-" in self.prior_df)
        print("ninki- in html?", "ninki-" in self.prior_df)
        print("Popular in html?", "Popular" in self.prior_df)

