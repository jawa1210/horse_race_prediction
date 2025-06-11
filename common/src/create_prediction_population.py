import re
from bs4 import BeautifulSoup
from urllib.request import urlopen, Request
import pandas as pd
import scraping
import time
from tqdm.notebook import tqdm
from pathlib import Path

POPULATION_DIR=Path("..","data","prediction_population")
POPULATION_DIR.mkdir(exist_ok=True,parents=True)

def scrape_horse_id_list(race_id: str)->list[str]:
    """
    レースidを指定すると、出走馬id一覧が返ってくる関数
    """
    url=f"https://race.netkeiba.com/race/shutuba.html?race_id={race_id}"
    headers={"User-Agent":"Mozilla/5.0"}
    request=Request(url, headers=headers)
    html = urlopen(request).read()
    soup=BeautifulSoup(html,"lxml")
    td_list=soup.find_all("td",class_="HorseInfo")
    horse_id_list=[]
    for td in td_list:
        horse_id=re.findall(r"\d{10}",td.find("a")["href"])[0]
        horse_id_list.append(horse_id)
    return horse_id_list

def create(
        kaisai_date:str,
        save_dir:Path=POPULATION_DIR,
        save_filename:str="population.csv"
)->pd.DataFrame:
    """
    開催日(yyymmdd形式)を指定すると、予測母集団である
    (date,race_id,horse_id)のDataFrameが返ってくる関数
    """
    print("scraping race_id_list...")
    race_id_list=scraping.scrape_race_id_list([kaisai_date])
    dfs={}
    print("scraping horse_id_list...")
    for race_id in tqdm(race_id_list):
        horse_id_list=scrape_horse_id_list(race_id)
        time.sleep(1)
        df=pd.DataFrame(
            {"date":kaisai_date,"race_id":race_id,"horse_id":horse_id_list}
        )
        dfs[race_id]=df
    concat_df=pd.concat(dfs.values())
    concat_df["date"]=pd.to_datetime(concat_df["date"])
    concat_df.to_csv(save_dir/save_filename,index=False,sep="\t")
    return concat_df
