from pathlib import Path
import pandas as pd
from bs4 import BeautifulSoup
import time
from tqdm.notebook import tqdm
from urllib.request import urlopen, Request
import re
import numpy as np

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
    soup = BeautifulSoup(html, "lxml").find("div", class_="data_intro")
    info_dict = {}
    info_dict["title"] = soup.find("h1").text
    p_list = soup.find_all("p")
    info_dict["info1"] = re.findall(
        r"[\w:]+", p_list[0].text.replace(" ", "")
    )
    info_dict["info2"] = re.findall(r"\w+", p_list[1].text)
    df = pd.DataFrame().from_dict(info_dict, orient="index").T
    concat_df = pd.concat(df.values())
    concat_df.index.name = "race_id"
    concat_df.columns = concat_df.columns.str.replace(" ", "")
    return concat_df


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
        horse_results_prediction_feilepath: Path = INPUT_DIR/"horse_results_prediction.csv"
    ):
        self.population = pd.read_csv(population_filepath, sep="\t")
        self.horse_results = pd.read_csv(horse_results_prediction_feilepath, sep="\t")

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

    def fetch_results(self) -> pd.DataFrame:
        """
        出馬ページのhtmlを受け取ると、レース結果テーブルを取得し、
        学習時と同じ形式に前処理する関数
        """
        # htmlからレース結果テーブルを抽出
        soup = BeautifulSoup(self.html, "lxml").find(
            "table", class_=["Shutuba_Table", "RaceTable01", "ShutubaTable", "tablesorter", "tablesorter-default"
                             ])

        if soup is None:
            raise ValueError("出馬表のテーブルが見つかりません。")

        df = pd.read_html(self.html)[0]

        df.columns = df.columns.get_level_values(-1)

        # 不要な空白や改行を削除
        df.columns = df.columns.str.strip()
        df.columns = df.columns.str.replace("\n", "", regex=True)


        # 必要な列がすべて含まれているか確認
        expected_cols = ["性齢", "馬体重 (増減)", "斤量", "人気", "Unnamed: 9_level_1", "枠", "馬 番"]
        for col in expected_cols:
            if col not in df.columns:
                print(f"Warning: '{col}' が DataFrame に存在しません")


        df["race_id"]=self.race_id

        # horse_id列追加
        horse_id_list = []
        horse_a_list = soup.find_all("a", href=re.compile(r"/horse/\d{10}"))
        for a in horse_a_list:
            horse_id = re.findall(r"\d{10}", a["href"])[0]
            horse_id_list.append(horse_id)
        df["horse_id"] = horse_id_list

        # jockey_id列追加
        jockey_id_list = []
        jockey_a_list = soup.find_all("a", href=re.compile(r"/jockey/result/recent/\d{5}"))
        jockey_id_list = []
        for a in jockey_a_list:
            jockey_id = re.findall(r"\d{5}", a["href"])[0]
            jockey_id_list.append(jockey_id)
        df["jockey_id"] = jockey_id_list

        # trainer_id列追加
        trainer_id_list = []
        trainer_a_list = soup.find_all("a", href=re.compile(r"/trainer/result/recent/"))
        trainer_id_list = []
        for a in trainer_a_list:
            trainer_id = re.findall(r"\d{5}", a["href"])[0]
            trainer_id_list.append(trainer_id)
        df["trainer_id"] = trainer_id_list

        # 単勝オッズ列追加
        tansyo_odds_list = []
        tansyo_odds_list = [td.find("span").text for td in soup.find_all("td", class_="Txt_R Popular")]

        # 人気列追加
        popularity_list = []
        popularity_list = [td.find("span").text for td in soup.find_all("td", class_="Popular Popular_Ninki Txt_C")]


        # # owner_id列追加
        # owner_id_list = []
        # owner_a_list = soup.find_all("a", href=re.compile(r"^/owner/"))
        # owner_id_list = []
        # for a in owner_a_list:
        #     owner_id = re.findall(r"\d{6}", a["href"])[0]
        #     owner_id_list.append(owner_id)
        # df["owner_id"] = owner_id_list

        df["rank"] = np.nan

        df["性齢"] = df["性齢"].astype(str)
        df["sex"] = df["性齢"].str[0].map(sex_mapping)
        df["age"] = df["性齢"].str[1:].astype(int)

        df["sex"] = df["性齢"].str[0].map(sex_mapping)
        df["age"] = df["性齢"].str[1:].astype(int)
        df["weight"] = df["馬体重 (増減)"].str.extract(r"(\d+)").astype(int)
        df["weight"] = pd.to_numeric(df["weight"], errors="coerce")
        df["weight_diff"] = df["馬体重 (増減)"].str.extract(r"\(([-+]?\d+)\)")  # 符号付き数値のみ抽出
        df["weight_diff"] = pd.to_numeric(df["weight_diff"], errors="coerce")  # 数値変換
        df["tansyo_odds"] = tansyo_odds_list
        df["tansyo_odds"] = pd.to_numeric(df["tansyo_odds"], errors="coerce")
        df["popularity"] = popularity_list
        df["popularity"] = pd.to_numeric(df["popularity"], errors="coerce")
        #df["popularity"]=df["popularity"].astype(int)
        df["impost"] = df["斤量"].astype(float)
        df["wakuban"] = df["枠"].astype(int)
        df["umaban"] = df["馬 番"].astype(int)
        df["agari"] = np.nan
        # データが着順に並んでいることによるリーク防止のため、各レースを馬番順にソートする
        df = df.sort_values(["umaban"])
        # 使用する列を選択
        df = df[
            [
                "race_id",
                "horse_id",
                "jockey_id",
                "trainer_id",
                "rank",
                "wakuban",
                "umaban",
                "sex",
                "age",
                "weight",
                "weight_diff",
                "tansyo_odds",
                "popularity",
                "impost",
                "agari",
            ]
        ]
        self.results = df

    def fetch_race_info(self) -> pd.DataFrame:
        """
        出馬ページのhtmlを受け取ると、レース情報を取得し、
        学習時と同じ形式に前処理する関数
        """
        df_info = create_race_info(self.html)
        race_info = {}
        # info1からデータを抽出
        for index in range(len(df_info)):
            # 各行のrace_id取得
            race_id = df_info["race_id"][index]

            # 各行の "info1" 列に格納されている文字列に対して正規表現を使用
            info1_text = df_info["info1"][index]

            # "芝" または "ダート" が含まれている部分を正規表現で検索
            race_type = re.findall(r"[芝ダ障]+", info1_text)
            if race_type:
                race_type = race_type[0]
                # race_typeでマッピング
                race_type = race_type_mapping.get(race_type, None)
            else:
                race_type = None  # 一致しない場合は None に設定

            # 方向（右、左、直）を抽出
            around = re.findall(r"[右左直]+", info1_text)
            if around:
                around = around[0]
                # aroundでマッピング
                around = around_mapping.get(around, None)
            else:
                around = None  # 一致しない場合は None に設定

            # 天気を抽出（晴、曇、小雨、雨、小雪、雪）
            regex_weather = "|".join(weather_mapping.keys())
            weather = re.findall(rf"({regex_weather})", info1_text)
            if weather:
                weather = weather[0]  # 一致した最初のクラスを取得
                # weatherでマッピング
                weather = weather_mapping.get(weather, None)
            else:
                weather = None  # 一致しない場合は None

            # レース距離を取得
            course_len = int(re.findall(r"\d+", info1_text)[0])

            ground_state = re.findall(r"[良稍重不]+", info1_text)
            if ground_state:
                ground_state = ground_state[0]
                # ground_stateでマッピング
                ground_state = ground_state_mapping.get(ground_state, None)
            else:
                ground_state = None

            # info2からデータを抽出
            info2_text = df_info["info2"][index]
            # race_classを正規表現でマッチさせてマッピングする
            regex_race_class = "|".join(race_class_mapping.keys())
            race_class = re.findall(rf"({regex_race_class})", info2_text)
            if race_class:
                race_class = race_class[0]  # 一致した最初のクラスを取得
                # race_class_mappingでマッピング
                race_class = race_class_mapping.get(race_class, None)
            else:
                race_class = None  # 一致しない場合は None

            # race_courseを正規表現でマッチさせてマッピングする
            regex_race_course = "|".join(race_course_mapping.keys())
            race_course = re.findall(rf"({regex_race_course})", info2_text)
            if race_course:
                race_course = race_course[0]  # 一致した最初のクラスを取得
                # race_class_mappingでマッピング
                race_course = race_course_mapping.get(race_course, None)
            else:
                race_course = None  # 一致しない場合は None

            # 日付を抽出
            pattern = r"(\d{4})年(\d{1,2})月(\d{1,2})日"  # 年、月、日をキャプチャグループで抽出
            matches = re.findall(pattern, info2_text)

            # 結果をyyyy-mm-dd形式に変換
            date = [f"{year}-{int(month):02d}-{int(day):02d}" for year, month, day in matches][0]

            # レース情報をまとめる
            race_info.append({
                "race_id": race_id,
                "date": date,
                "race_type": race_type,
                "around": around,
                "course_len": course_len,
                "weather": weather,
                "ground_state": ground_state,
                "race_class": race_class,
                "place": race_course,
            })

        # 最終的なDataFrameを作成
        race_info = pd.DataFrame(race_info)
        self.race_info = race_info

    def create_features(self, race_id, skip_agg_horse: bool = False) -> pd.DataFrame:
        """
        特徴量作成処理を実行し、populationテーブルにすべての特徴量を統合する
        先にagg_horse_n_races()を実行しておいた場合は、skip_agg_horse=Trueとすればスキップできる
        """
        # 馬の過去成績集計
        if not skip_agg_horse:
            self.agg_horse_n_races()

        # 各種テーブルの取得
        self.race_id=race_id
        self.fetch_syutuba_table_html(race_id)  # race_idを使用
        self.fetch_results()
        self.fetch_race_info()

        # 特徴量の統合
        features = (
            self.population.merge(self.results, on=["race_id", "horse_id"])
            .merge(self.race_info, on=["race_id", "date"])
            .merge(
                self.agg_horse_n_races_df,
                on=["race_id", "date", "horse_id"],
                how="left",
            )
        )
        features.to_csv(self.output_dir / "prediction_features.csv", sep="\t", index=None)
        return features
