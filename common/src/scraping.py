from urllib.request import urlopen, Request
from urllib.error import HTTPError, URLError
from bs4 import BeautifulSoup
import pandas as pd
import re
import time
from tqdm.notebook import tqdm
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
import traceback
from pathlib import Path
import random

HTML_DIR = Path("..", "data", "html")
HTML_RACE_DIR = HTML_DIR / "race"
HTML_HORSE_DIR = HTML_DIR / "horse"
HTML_RACE_PREDICT_DIR=HTML_DIR/"predict_race"

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:115.0) Gecko/20100101 Firefox/115.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:115.0) Gecko/20100101 Firefox/115.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.2 Safari/605.1.15",
]

def _sleep(base=1.2, jitter=1.0):
    time.sleep(base + random.random() * jitter)

def _fetch(url: str, referer: str | None = None, timeout: int = 30, max_retry: int = 4) -> bytes:
    """
    urlopen版の「安全フェッチ」
    - UAローテ
    - referer
    - リトライ + バックオフ
    """
    backoff = 1.0
    for attempt in range(max_retry):
        headers = {"User-Agent": random.choice(USER_AGENTS)}
        if referer:
            headers["Referer"] = referer

        req = Request(url, headers=headers)
        try:
            _sleep(1.0, 1.2)  # request前に少し揺らす
            return urlopen(req, timeout=timeout).read()

        except HTTPError as e:
            code = getattr(e, "code", None)
            print(f"[HTTPError] {code} {url} (attempt {attempt+1}/{max_retry})")
            # 429/503/400系は一旦休む（増やす）
            _sleep(2.0 * backoff, 3.0 * backoff)
            backoff *= 1.8

        except URLError as e:
            print(f"[URLError] {e} {url} (attempt {attempt+1}/{max_retry})")
            _sleep(2.0 * backoff, 3.0 * backoff)
            backoff *= 1.8

        except Exception as e:
            print(f"[FAIL] {e} {url} (attempt {attempt+1}/{max_retry})")
            _sleep(2.0 * backoff, 3.0 * backoff)
            backoff *= 1.8

    raise RuntimeError(f"fetch failed after retries: {url}")


def scrape_kaisai_date(from_: str, to_: str) -> list[str]:
    kaisai_date_list = []
    for date in tqdm(pd.date_range(from_, to_, freq="MS")):
        year = date.year
        month = date.month
        url = f"https://race.netkeiba.com/top/calendar.html?year={year}&month={month}"
        html = _fetch(url, referer="https://race.netkeiba.com/top/")
        soup = BeautifulSoup(html, "lxml")
        a_list = soup.find("table", class_="Calendar_Table").find_all("a")
        for a in a_list:
            kaisai_date = re.findall(r"kaisai_date=(\d{8})", a["href"])[0]
            kaisai_date_list.append(kaisai_date)
        _sleep(1.0, 2.0)  # 月ごとにも少し休む
    return kaisai_date_list


def scrape_race_id_list(kaisai_date_list: list[str]) -> list[str]:
    """
    ここが一番BANリスク高いので、sleep強め & 例外でbreakしない
    """
    options = Options()
    options.add_argument("--headless=new")
    options.add_argument(f"--user-agent={random.choice(USER_AGENTS)}")

    chromedriver_path = "/home/jawa/.wdm/drivers/chromedriver/linux64/131.0.6778.264/chromedriver-linux64/chromedriver"
    service = Service(chromedriver_path)

    race_id_list = []
    with webdriver.Chrome(service=service, options=options) as driver:
        for kaisai_date in tqdm(kaisai_date_list):
            url = f"https://race.netkeiba.com/top/race_list.html?kaisai_date={kaisai_date}"
            try:
                driver.get(url)
                _sleep(1.5, 2.5)

                li_list = driver.find_elements(By.CLASS_NAME, "RaceList_DataItem")
                for li in li_list:
                    href = li.find_element(By.TAG_NAME, "a").get_attribute("href")
                    m = re.findall(r"race_id=(\d{12})", href)
                    if m:
                        race_id_list.append(m[0])

                _sleep(1.0, 2.0)

            except Exception:
                print(f"[WARN] stopped at {url}")
                print(traceback.format_exc())
                # breakじゃなくて次へ（大量欠損を防ぐ）
                _sleep(5.0, 10.0)
                continue

    return race_id_list


def scrape_html_race(race_id_list: list[str], save_dir: Path = HTML_RACE_DIR) -> list[Path]:
    html_path_list = []
    save_dir.mkdir(parents=True, exist_ok=True)

    for race_id in tqdm(race_id_list):
        filepath = save_dir / f"{race_id}.bin"

        if filepath.is_file():
            print(f"skipped:{race_id}")
            html_path_list.append(filepath)
            continue

        url = f"https://db.netkeiba.com/race/{race_id}"
        html = _fetch(url, referer="https://db.netkeiba.com/")
        with open(filepath, "wb") as f:
            f.write(html)
        html_path_list.append(filepath)

        _sleep(1.0, 2.5)

    return html_path_list

def scrape_html_predict_race(race_id_list: list[str], save_dir: Path = HTML_RACE_DIR) -> list[Path]:
    html_path_list = []
    save_dir.mkdir(parents=True, exist_ok=True)

    for race_id in tqdm(race_id_list):
        filepath = save_dir / f"{race_id}.bin"

        if filepath.is_file():
            print(f"skipped:{race_id}")
            html_path_list.append(filepath)
            continue

        url = f"https://race.netkeiba.com/race/shutuba.html?race_id={race_id}&rf=race_list"
        html = _fetch(url, referer="https://db.netkeiba.com/")
        with open(filepath, "wb") as f:
            f.write(html)
        html_path_list.append(filepath)

        _sleep(1.0, 2.5)

    return html_path_list


def scrape_html_horse(horse_id_list: list[str], save_dir: Path = HTML_HORSE_DIR, skip: bool = True) -> list[Path]:
    """
    返り値の一貫性のため
    - skip=Trueでも pathは返す（raceと同じ思想）
    """
    html_path_list = []
    save_dir.mkdir(parents=True, exist_ok=True)

    for horse_id in tqdm(horse_id_list):
        filepath = save_dir / f"{horse_id}.bin"

        if filepath.is_file() and skip:
            print(f"skipped:{horse_id}")
            html_path_list.append(filepath)
            continue

        url = f"https://db.netkeiba.com/horse/{horse_id}"
        html = _fetch(url, referer="https://db.netkeiba.com/")
        with open(filepath, "wb") as f:
            f.write(html)
        html_path_list.append(filepath)

        _sleep(1.2, 3.0)

    return html_path_list
