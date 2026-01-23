from pathlib import Path

import pandas as pd
import re

COMMON_DATA_DIR = Path("..", "..", "common", "data")
INPUT_DIR = COMMON_DATA_DIR/"rawdf"
MAPPING_DIR = COMMON_DATA_DIR/"mapping"
OUTPUT_DIR = Path("..", "data", "01_preprocessed")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

sex_mapping={"牡":0,"牝":1,"セ":2}
weather_mapping={"晴":0,"曇":1,"小雨":2,"雨":3,"雪":4,"小雪":5,"nan":6}
race_type_mapping={"ダ":0,"芝":1,"障":2}
race_class_mapping={
    "新馬":0,
    "未勝利":1,
    "1勝クラス":2,
    "2勝クラス":3,
    "3勝クラス":4,
    "オープン":5,
    "G3":6,
    "G2":7,
    "G1":8,
    "OP":5,
    "特別":5,
    "500万下":2,
    "1000万下":3,
    "1600万下":4,
    "C3":9,
    "C2":10,
    "C1":11,
    "B3":12,
    "B2":13,
    "B1":14,
    "A3":15,
    "A2":16,
    "A1":17,
    "重賞":18
}
ground_state_mapping={
    "良":0,
    "重":1,
    "稍重":2,
    "不良":3,
    "稍":2,
    "不":3
}
around_mapping={
    "右":0,
    "左":1,
    "直":2
}

race_course_mapping = {
    "東京":0, "阪神":1, "中山":2, "京都":3, "中京":4, "札幌":5, "函館":6, "新潟":7,
    "小倉":8, "福島":9, "函館":10, "盛岡":11,"水沢":12, "金沢":13, "高知":14, "大井":15, "川崎":16,"浦和":17,"船橋":18,"名古屋":19,"笠松":20,"園田":21,"姫路":22,"門別":23,"帯広":24
}

INOUT_MAP = {"内": 0, "外": 1}
ABC_MAP = {"A": 0, "B": 1, "C": 2}

def _to_text(x) -> str:
    """
    旧形式: "['芝右2000m', '天候:晴', ...]" みたいな文字列 or 実list
    新形式: "芝2000m (右 外 A) 天候:晴 ..." みたいな文字列
    → どっちも 1 本の文字列に正規化
    """
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return ""
    if isinstance(x, list):
        return " ".join([str(v) for v in x])
    s = str(x).strip()

    if s.startswith("[") and s.endswith("]"):
        try:
            obj = ast.literal_eval(s)
            if isinstance(obj, list):
                return " ".join([str(v) for v in obj])
        except Exception:
            pass
    return s

def ensure_race_info_schema_raw(raw_race_info_path: Path) -> pd.DataFrame:
    """
    common/data/rawdf/race_info.csv が旧形式でも、新しい列が無い場合でも、
    列を揃えて保存し直す（再スクレイプ不要）
    """
    df = pd.read_csv(raw_race_info_path, sep="\t")

    # 必須列が無ければ追加（欠損コードでOK）
    for c in ["around_dir", "around_inout", "course_abc"]:
        if c not in df.columns:
            df[c] = -1

    # dtype 固定
    for c in ["around_dir", "around_inout", "course_abc"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(-1).astype("int32")

    if "race_id" in df.columns:
        df["race_id"] = df["race_id"].astype(str)

    df.to_csv(raw_race_info_path, sep="\t", index=False)
    return df


def process_results(
        input_dir: Path = INPUT_DIR,
        output_dir: Path = OUTPUT_DIR,
        save_filename: str = "results.csv",
        sex_mapping :dict =sex_mapping
        ) -> pd.DataFrame:
    """
    レース結果テーブルのrawデータをinput_dirから読み込んで、加工し、
    output_dirに保存する関数
    """
    df = pd.read_csv(input_dir/save_filename, sep="\t")
    df["rank"] = pd.to_numeric(df["着順"], errors="coerce")
    df.dropna(subset=["rank"], inplace=True)
    df["rank"]=df["rank"].astype(int)
    df["sex"]=df["性齢"].str[0].map(sex_mapping)
    df["age"]=df["性齢"].str[1:].astype(int)
    df["weight"]=df["馬体重"].str.extract(r"(\d+)").astype(int)
    df["weight"]=pd.to_numeric(df["weight"],errors="coerce")
    df["weight_diff"]=df["馬体重"].str.extract(r"\((.)+\)").astype(int)
    df["weight_diff"]=pd.to_numeric(df["weight_diff"],errors="coerce")
    df["tansyo_odds"]=df["単勝"].astype(float)
    df["popularity"]=df["人気"].astype(int)
    df["impost"]=df["斤量"].astype(float)
    df["wakuban"]=df["枠番"].astype(int)
    df["umaban"]=df["馬番"].astype(int)
    df["agari"]=df["上り"]
    #データが着順に並んでいることによるリーク防止のため、各レースを馬番順にソートする
    df=df.sort_values(["race_id","umaban"])
    #使用する列を選択
    df=df[
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
    df.to_csv(output_dir / save_filename, sep="\t", index=False)
    return df

def process_horse_results(
        input_dir: Path = INPUT_DIR,
        output_dir: Path = OUTPUT_DIR,
        save_filename: str = "horse_results.csv",
        race_type_mapping: dict=race_type_mapping,
        weather_mapping: dict=weather_mapping,
        ground_state_mapping:dict=ground_state_mapping,
        race_class_mapping:dict=race_class_mapping
        ) -> pd.DataFrame:
    """
    レース結果テーブルのrawデータをinput_dirから読み込んで、加工し、
    output_dirに保存する関数
    """
    df = pd.read_csv(input_dir/save_filename, sep="\t")
    df["rank"] = pd.to_numeric(df["着順"], errors="coerce")
    df.dropna(subset=["rank"], inplace=True)
    df["rank"]=df["rank"].astype(int)
    df["date"]=pd.to_datetime(df["日付"])
    df["weather"]=df["天気"].map(weather_mapping)
    df["race_type"]=df["距離"].str[0].map(race_type_mapping)
    df["course_len"]=df["距離"].str.extract(r"(\d+)").astype(int)
    df["ground_state"]=df["馬場"].map(ground_state_mapping)
    df["agari"]=df["上り"]
    df["rank_diff"]=df["着差"].map(lambda x:0 if x<0 else x)
    df["prize"]=df["賞金"].fillna(0)
    regex_race_class="|".join(race_class_mapping.keys())
    df["race_class"]=(
        df["レース名"].str.extract(rf"({regex_race_class})")[0].map(race_class_mapping)
    )
    df.rename(columns={"頭数":"n_horses"},inplace=True)
    #使用する列を選択
    df=df[
        [
            "horse_id",
            "date",
            "rank",
            "prize",
            "rank_diff",
            "weather",
            "race_type",
            "course_len",
            "ground_state",
            "agari",
            "race_class",
            "n_horses"
        ]
    ]
    df.to_csv(output_dir / save_filename, sep="\t", index=False)
    return df

# def process_race_info(input_dir: Path = INPUT_DIR,
#         output_dir: Path = OUTPUT_DIR,
#         save_filename: str = "race_info.csv",
#         race_type_mapping: dict=race_type_mapping,
#         weather_mapping: dict=weather_mapping,
#         ground_state_mapping:dict=ground_state_mapping,
#         race_class_mapping:dict=race_class_mapping,
#         around_mapping:dict=around_mapping,
#         race_course_mapping:dict=race_course_mapping
# ) -> pd.DataFrame:
#     """
#     レース情報を加工し、必要な列を作成して新しいDataFrameを返す関数
#     """
#     # 空のリストを作成して最終的なデータを格納
#     race_info = []
#     df = pd.read_csv(input_dir/save_filename, sep="\t")

#     # info1からデータを抽出
#     for index in range(len(df)):
#         #各行のrace_id取得
#         race_id=df["race_id"][index]

#         # 各行の "info1" 列に格納されている文字列に対して正規表現を使用
#         info1_text = df["info1"][index]

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

#         #レース距離を取得
#         course_len = int(re.findall(r"\d+", info1_text)[0])

#         ground_state = re.findall(r"[良稍重不]+", info1_text)
#         if ground_state:
#             ground_state = ground_state[0]
#             # ground_stateでマッピング
#             ground_state = ground_state_mapping.get(ground_state, None)
#         else:
#             ground_state = None

#         # info2からデータを抽出
#         info2_text = df["info2"][index]
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

#         #日付を抽出
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
#             })

#     # 最終的なDataFrameを作成
#     df_race_info = pd.DataFrame(race_info)
#     df_race_info.to_csv(output_dir / save_filename, sep="\t", index=False)
#     return df_race_info

def process_race_info(input_dir: Path = INPUT_DIR,
        output_dir: Path = OUTPUT_DIR,
        save_filename: str = "race_info.csv",
        race_type_mapping: dict=race_type_mapping,
        weather_mapping: dict=weather_mapping,
        ground_state_mapping:dict=ground_state_mapping,
        race_class_mapping:dict=race_class_mapping,
        around_mapping:dict=around_mapping,
        race_course_mapping:dict=race_course_mapping
) -> pd.DataFrame:
    """
    race_info(rawdf) を前処理して、学習/予測で使える数値特徴へ変換。
    - 旧: info1/info2 が "['...','...']" でも対応
    - 新: info1/info2 が生テキストでも対応
    - around_dir / around_inout / course_abc もここで作る（取れない場合は -1）
    """
    df = pd.read_csv(input_dir / save_filename, sep="\t")

    # 旧/新両対応：必ず文字列へ正規化
    df["info1_text"] = df["info1"].map(_to_text) if "info1" in df.columns else ""
    df["info2_text"] = df["info2"].map(_to_text) if "info2" in df.columns else ""

    regex_weather = "|".join(map(re.escape, weather_mapping.keys()))
    regex_race_class = "|".join(map(re.escape, race_class_mapping.keys()))
    regex_race_course = "|".join(map(re.escape, race_course_mapping.keys()))

    rows = []
    for _, r in df.iterrows():
        race_id = str(r.get("race_id", ""))
        info1 = str(r.get("info1_text", ""))
        info2 = str(r.get("info2_text", ""))

        t1 = re.sub(r"\s+", "", info1)
        t2 = re.sub(r"\s+", "", info2)

        # race_type
        m = re.search(r"(芝|ダ|障)", t1)
        race_type = race_type_mapping.get(m.group(1), None) if m else None

        # around_dir
        m = re.search(r"(右|左|直)", t1)
        around_dir = around_mapping.get(m.group(1), None) if m else None

        # around_inout
        m = re.search(r"(内|外)", t1)
        around_inout = INOUT_MAP.get(m.group(1), None) if m else None

        # course_abc
        m = re.search(r"(A|B|C)", t1)
        course_abc = ABC_MAP.get(m.group(1), None) if m else None

        # weather
        m = re.search(rf"({regex_weather})", t1)
        weather = weather_mapping.get(m.group(1), None) if m else None

        # course_len
        m = re.search(r"(\d+)", t1)
        course_len = int(m.group(1)) if m else None

        # ground_state（長い語優先）
        m = re.search(r"(不良|稍重|良|重|稍|不)", t1)
        ground_state = ground_state_mapping.get(m.group(1), None) if m else None

        # race_class
        m = re.search(rf"({regex_race_class})", t2)
        race_class = race_class_mapping.get(m.group(1), None) if m else None

        # place
        m = re.search(rf"({regex_race_course})", t2)
        place = race_course_mapping.get(m.group(1), None) if m else None

        # date
        m = re.search(r"(\d{4})年(\d{1,2})月(\d{1,2})日", info2)
        if m:
            y, mo, d = m.group(1), m.group(2), m.group(3)
            date = f"{y}-{int(mo):02d}-{int(d):02d}"
        else:
            date = None

        rows.append({
            "race_id": race_id,
            "date": date,
            "race_type": race_type,
            "around_dir": around_dir,
            "around_inout": around_inout,
            "course_abc": course_abc,
            "course_len": course_len,
            "weather": weather,
            "ground_state": ground_state,
            "race_class": race_class,
            "place": place,
        })

    out = pd.DataFrame(rows)

    # dtype 固定（LightGBM 事故防止）
    out["race_type"] = pd.to_numeric(out["race_type"], errors="coerce").fillna(-1).astype("int32")
    out["around_dir"] = pd.to_numeric(out["around_dir"], errors="coerce").fillna(-1).astype("int32")
    out["around_inout"] = pd.to_numeric(out["around_inout"], errors="coerce").fillna(-1).astype("int32")
    out["course_abc"] = pd.to_numeric(out["course_abc"], errors="coerce").fillna(-1).astype("int32")
    out["course_len"] = pd.to_numeric(out["course_len"], errors="coerce").fillna(-1).astype("int32")

    out["weather"] = pd.to_numeric(out["weather"], errors="coerce").fillna(6).astype("int32")          # 欠損=6
    out["ground_state"] = pd.to_numeric(out["ground_state"], errors="coerce").fillna(9).astype("int32") # 欠損=9

    out["race_class"] = pd.to_numeric(out["race_class"], errors="coerce").fillna(-1).astype("int32")
    out["place"] = pd.to_numeric(out["place"], errors="coerce").fillna(-1).astype("int32")

    output_dir.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_dir / save_filename, sep="\t", index=False)
    return out


