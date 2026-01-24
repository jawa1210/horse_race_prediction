from __future__ import annotations

import re
import time
from pathlib import Path
from urllib.request import urlopen, Request

import pandas as pd
from bs4 import BeautifulSoup
from tqdm.notebook import tqdm

from scraping import scrape_race_id_list  # 既存のscrapingを使う

# 既存binの場所（あなたのツリーに合わせて）
HTML_RACE_DIR = Path("..", "data", "html", "race")  # common/src から見た相対なら調整してOK
POPULATION_DIR = Path("..", "data", "prediction_population")
POPULATION_DIR.mkdir(exist_ok=True, parents=True)

UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36"


def _load_existing_race_ids(bin_dir: Path) -> list[str]:
    if not bin_dir.exists():
        return []
    ids = [p.stem for p in bin_dir.glob("*.bin") if re.fullmatch(r"\d{12}", p.stem)]
    return sorted(ids)


def _filter_race_ids_by_kaisai_date(race_ids: list[str], kaisai_date: str) -> list[str]:
    """
    race_id は 12桁: YYYYPPKKDDRR
    kaisai_date は 8桁: YYYYPPKKDD と一致する（先頭8桁）
    """
    s = re.sub(r"\D", "", str(kaisai_date))
    if len(s) != 8:
        raise ValueError(f"kaisai_date must be 8 digits (YYYYMMDD): {kaisai_date}")
    return [rid for rid in race_ids if rid.startswith(s)]


def scrape_horse_id_list(race_id: str) -> list[str]:
    """
    shutuba.html から horse_id を拾う（落ちない版）
    """
    url = f"https://race.netkeiba.com/race/shutuba.html?race_id={race_id}"
    req = Request(url, headers={"User-Agent": UA})
    html = urlopen(req, timeout=20).read()
    soup = BeautifulSoup(html, "lxml")

    # HorseInfo が取れない場合もあるので、aタグから広めに拾う
    horse_ids = []
    for a in soup.find_all("a", href=True):
        href = a["href"]
        # horse/XXXX や /horse/XXXX を拾う（8-12桁まで許容）
        if "horse" in href:
            m = re.findall(r"\d{8,12}", href)
            if m:
                horse_ids.append(m[-1])

    # 重複除去（順序維持）
    seen = set()
    out = []
    for hid in horse_ids:
        if hid not in seen:
            seen.add(hid)
            out.append(hid)
    return out


def create(
    kaisai_date: str,
    save_dir: Path = POPULATION_DIR,
    save_filename: str = "population.csv",
    sleep_sec: float = 0.8,
    prefer_local_bin: bool = True,
) -> pd.DataFrame:
    """
    1) まずローカルbinから race_id を復元（prefer_local_bin=True）
    2) 無ければ scraping.scrape_race_id_list で取りに行く
       ※ただしあなたのscrape_race_id_listは「既存binをスキップ」するので、
         race_id収集目的なら local_bin 方式が正解
    """
    kaisai_date = re.sub(r"\D", "", str(kaisai_date))

    # --- race_id_list を作る ---
    race_id_list: list[str] = []
    if prefer_local_bin:
        all_ids = _load_existing_race_ids(HTML_RACE_DIR)
        race_id_list = _filter_race_ids_by_kaisai_date(all_ids, kaisai_date)

    if not race_id_list:
        # fallback（ネットから）: 既存binでもrace_idは必要なのでスキップしない
        race_id_list = scrape_race_id_list([kaisai_date], bin_dir=HTML_RACE_DIR, skip_existing=False)


    if not race_id_list:
        raise ValueError(f"race_id_list is empty for kaisai_date={kaisai_date}")

    # --- horse_id_list を集める ---
    dfs: list[pd.DataFrame] = []
    failed = []
    empty = []

    for rid in tqdm(race_id_list, desc="scrape horse_id_list"):
        try:
            horse_ids = scrape_horse_id_list(rid)
            if not horse_ids:
                empty.append(rid)
                continue

            df = pd.DataFrame({"date": [kaisai_date] * len(horse_ids), "race_id": rid, "horse_id": horse_ids})
            dfs.append(df)

        except Exception as e:
            failed.append((rid, repr(e)))

        time.sleep(sleep_sec)

    if not dfs:
        raise ValueError(
            f"No horse_id collected. kaisai_date={kaisai_date} "
            f"races={len(race_id_list)} empty={len(empty)} failed={len(failed)} first_failed={failed[:1]}"
        )

    pop = pd.concat(dfs, ignore_index=True)
    # date は後段が YYYY-MM-DD を期待するならここで変換してもOK
    pop["date"] = pd.to_datetime(pop["date"], format="%Y%m%d", errors="coerce")

    save_dir.mkdir(exist_ok=True, parents=True)
    out_path = save_dir / save_filename
    pop.to_csv(out_path, index=False, sep="\t")

    print("[create_prediction_population] saved:", out_path.resolve())
    if empty:
        print("[WARN] empty horse list races (first10):", empty[:10])
    if failed:
        print("[WARN] failed races (first5):", failed[:5])

    return pop
