from urllib.request import urlopen
from bs4 import BeautifulSoup
import pandas as pd
import re
import time
from tqdm.notebook import tqdm


def scrape_kaisai_date(from_: str, to_: str):
    """
    from_toとto_をyyyy-mmの形で指定すると、間の開催日一覧を取得する関数
    """
    kaisai_date_list = []
    for date in tqdm(pd.date_range(from_, to_, freq="MS")):
        year = date.year
        month = date.month
        url = f'https://race.netkeiba.com/top/calendar.html?year={year}&month={month}'
        html = urlopen(url).read()  # スクレイピング
        time.sleep(1)
        soup = BeautifulSoup(html)
        a_list = soup.find("table", class_="Calendar_Table").find_all("a")
        for a in a_list:
            kaisai_date = re.findall(r"kaisai_date=(\d{8})", a["href"])[0]
            kaisai_date_list.append(kaisai_date)

    return kaisai_date_list
