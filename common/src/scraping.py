from urllib.request import urlopen, Request
from bs4 import BeautifulSoup
import pandas as pd
import re
import time
from tqdm.notebook import tqdm
from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import NoSuchWindowException
from selenium.webdriver.common.by import By
import traceback
from pathlib import Path
import pickle

HTML_DIR = Path("..", "data", "html",)
HTML_RACE_DIR = HTML_DIR/"race"
HTML_HORSE_DIR = HTML_DIR/"horse"


def scrape_kaisai_date(from_: str, to_: str) -> list[str]:
    """
    from_とto_をyyyy-mmの形で指定すると,開催日一覧を取得する関数
    """
    kaisai_date_list = []
    for date in tqdm(pd.date_range(from_, to_, freq="MS")):
        year = date.year
        month = date.month
        url = f'https://race.netkeiba.com/top/calendar.html?year={year}&month={month}'
        headers = {"User-Agent": "Mozilla/5.0"}
        request = Request(url, headers=headers)
        html = urlopen(request).read()  # スクレイピング
        time.sleep(1)
        soup = BeautifulSoup(html, "lxml")
        a_list = soup.find("table", class_="Calendar_Table").find_all("a")
        for a in a_list:
            kaisai_date = re.findall(r"kaisai_date=(\d{8})", a["href"])[0]
            kaisai_date_list.append(kaisai_date)

    return kaisai_date_list


def scrape_race_id_list(kaisai_date_list: list[str]) -> list[str]:
    """
    開催日 (yyyymmdd形式)をリストで入れると、レースidが返ってくる変数
    """
    options = Options()
    options.add_argument("--user-agent=Mozilla/5.0")
    options.add_argument("--headless")

    # 正しいバイナリファイルのパスを指定
    chromedriver_path = '/home/jawa/.wdm/drivers/chromedriver/linux64/131.0.6778.264/chromedriver-linux64/chromedriver'
    service = Service(chromedriver_path)
    race_id_list = []

    with webdriver.Chrome(service=service, options=options) as driver:
        for kaisai_date in tqdm(kaisai_date_list):
            url = f"https://race.netkeiba.com/top/race_list.html?kaisai_date={kaisai_date}"
            try:
                driver.get(url)
                time.sleep(1)
                li_list = driver.find_elements(By.CLASS_NAME, "RaceList_DataItem")
                for li in li_list:
                    href = li.find_element(By.TAG_NAME, "a").get_attribute("href")
                    race_id = re.findall(r"race_id=(\d{12})", href)[0]
                    # print(race_id)
                    race_id_list.append(race_id)
            except:
                print(f"stopped at {url}")
                print(traceback.format_exc())
                break
    return race_id_list


def scrape_html_race(race_id_list: list[str], save_dir: Path = HTML_RACE_DIR) -> list[Path]:
    """
    netkeiba.comのraceページのhtmlをスクレイピングして、save_dirに保存する関数
    すでにhtmlが存在される場合はスキップされて、新たに取得されたthtmlのpathだけが返ってくる
    """
    html_path_list = []
    save_dir.mkdir(parents=True, exist_ok=True)
    for race_id in tqdm(race_id_list):
        filepath = save_dir/f"{race_id}.bin"
        # binファイルがすでに存在する場合はスキップする
        if filepath.is_file():
            print(f"skipped:{race_id}")
        else:
            url = f"https://db.netkeiba.com/race/{race_id}"
            headers = {"User-Agent": "Mozilla/5.0"}
            request = Request(url, headers=headers)
            html = urlopen(request).read()
            time.sleep(1)
            with open(filepath, "wb") as f:
                f.write(html)
            html_path_list.append(filepath)
    return html_path_list


def scrape_html_horse(horse_id_list: list[str], save_dir: Path = HTML_HORSE_DIR, skip: bool = True) -> list[Path]:
    """
    netkeiba.comのhorseページのhtmlをスクレイピングして、save_dirに保存する関数
    すでにhtmlが存在するとき、skip=Trueの場合はスキップされて、新たに取得されたthtmlのpathだけが返ってくる
    逆にskip=Falseのとき、htmlが再度作られ、情報が上書き(更新)される
    """
    html_path_list = []
    save_dir.mkdir(parents=True, exist_ok=True)
    for horse_id in tqdm(horse_id_list):
        filepath = save_dir/f"{horse_id}.bin"
        # binファイルがすでに存在し、skip=Trueの場合はスキップする
        if filepath.is_file() and skip:
            print(f"skipped:{horse_id}")
        else:
            url = f"https://db.netkeiba.com/horse/{horse_id}"
            headers = {"User-Agent": "Mozilla/5.0"}
            request = Request(url, headers=headers)
            html = urlopen(request).read()
            time.sleep(1)
            with open(filepath, "wb") as f:
                f.write(html)
            html_path_list.append(filepath)
            #skipされた場合、htmlのlistが空になって表示される気がする。つまり、すべて取得済みだと返り値が空
    return html_path_list
