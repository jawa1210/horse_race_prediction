import pandas as pd
from pathlib import Path
from tqdm.notebook import tqdm
from bs4 import BeautifulSoup
import re

RAWDF_DIR = Path("..", "data", "rawdf")


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


def create_horse_results(html_path_list: list[Path], save_dir: Path = RAWDF_DIR, save_filename: str = "horse_results.csv") -> pd.DataFrame:
    """
    horseページのhtmlを読み込んで、馬の過去成績テーブルに加工する関数
    """
    dfs = {}
    for html_path in tqdm(html_path_list):
        with open(html_path, "rb") as f:
            try:
                horse_id = html_path.stem
                html = f.read().replace(b"<diary_snap_cut>", b"").replace(b"<diary_snap_cut>", b"")
                df = pd.read_html(html)[2]

                df.index = [horse_id]*len(df)
                dfs[horse_id] = df
            except IndexError as e:
                print(f"table not found at {horse_id}")
                continue
    concat_df = pd.concat(dfs.values())
    concat_df.index.name = "horse_id"
    concat_df.columns = concat_df.columns.str.replace(" ", "")
    save_dir.mkdir(parents=True, exist_ok=True)
    concat_df.to_csv(save_dir/save_filename, sep="\t")
    return concat_df


def create_race_info(
        html_path_list: list[Path],
        save_dir: Path = RAWDF_DIR,
        save_filename: str = "race_info.csv"
) -> pd.DataFrame:
    """
    raceの詳細ページのhtmlを読み込んで、レース情報テーブルに加工する関数
    """
    dfs = {}
    for html_path in tqdm(html_path_list):
        with open(html_path, "rb") as f:
            try:
                html = f.read()
                soup = BeautifulSoup(html, "lxml").find("div", class_="data_intro")
                info_dict = {}
                info_dict = {}
                info_dict["title"] = soup.find("h1").text
                p_list = soup.find_all("p")
                info_dict["info1"] = re.findall(
                    r"[\w:]+", p_list[0].text.replace(" ", "")
                )
                info_dict["info2"] = re.findall(r"\w+", p_list[1].text)
                df=pd.DataFrame().from_dict(info_dict, orient="index").T

                # ファイル名からrace_idを取得
                race_id = html_path.stem
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
    return concat_df.reset_index()
