# report_html.py
from pathlib import Path
import datetime as dt
import time

import pandas as pd
from tqdm.notebook import tqdm

from feature_engineering import PredictionFeatureCreator, RAW_DATA_DIR, DATA_DIR
from prediction import predict_win, predict_trifecta_topk

POPULATION_CSV = RAW_DATA_DIR / "prediction_population" / "population.csv"

OUT_DIR = DATA_DIR / "predictions_html"
OUT_DIR.mkdir(exist_ok=True, parents=True)

MODEL_DIR = Path("..", "data") / "03_train"


def _attach_names_if_exist(df: pd.DataFrame, feats: pd.DataFrame) -> pd.DataFrame:
    """
    featsに horse_name / jockey_name があれば df に付ける（無ければそのまま）
    keyは race_id + umaban
    """
    out = df.copy()
    name_cols = []
    if "horse_name" in feats.columns:
        name_cols.append("horse_name")
    if "jockey_name" in feats.columns:
        name_cols.append("jockey_name")
    if not name_cols:
        return out

    key_cols = [c for c in ["race_id", "umaban"] if c in out.columns and c in feats.columns]
    if not key_cols:
        return out

    tmp = feats[key_cols + name_cols].drop_duplicates()
    return out.merge(tmp, on=key_cols, how="left")


def build_html_report(target_date: str, topk_per_race: int = 18) -> Path:
    pop = pd.read_csv(POPULATION_CSV, sep="\t", dtype={"race_id": str})
    pop["date"] = pop["date"].astype(str).str.strip()

    race_ids = sorted(pop.loc[pop["date"] == target_date, "race_id"].astype(str).unique())
    if len(race_ids) == 0:
        raise ValueError(f"{target_date} の race_id が population.csv にありません")

    pfc = PredictionFeatureCreator()
    pfc.agg_horse_n_races()  # 重いので最初に1回

    now = dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    parts = []
    parts.append(f"""<!doctype html>
<html lang="ja">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>予想表 {target_date}</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Noto Sans JP", "Hiragino Sans", "Yu Gothic", sans-serif;
         margin: 24px; background: #fafafa; color: #111; }}
  h1 {{ margin: 0 0 8px 0; font-size: 24px; }}
  .meta {{ color: #444; margin-bottom: 18px; }}
  .toc {{ background: #fff; border-radius: 12px; padding: 14px 16px; box-shadow: 0 2px 10px rgba(0,0,0,0.06); margin-bottom: 18px; }}
  .toc a {{ text-decoration: none; margin-right: 10px; white-space: nowrap; display: inline-block; padding: 6px 10px; border-radius: 999px; background: #f1f3f5; color: #111; }}
  .race {{ background: #fff; border-radius: 14px; padding: 14px 16px; box-shadow: 0 2px 12px rgba(0,0,0,0.07); margin: 14px 0; }}
  .race h2 {{ margin: 0 0 8px 0; font-size: 18px; }}
  h3 {{ margin: 14px 0 8px 0; font-size: 16px; }}
  table {{ border-collapse: collapse; width: 100%; margin-bottom: 6px; }}
  th, td {{ border-bottom: 1px solid #eee; padding: 8px 8px; text-align: left; font-size: 14px; }}
  th {{ background: #fbfbfb; position: sticky; top: 0; z-index: 1; }}
  .note {{ color:#555; font-size: 13px; margin-top: 8px; }}
  .grid {{ display: grid; grid-template-columns: 1fr; gap: 12px; }}
  @media (min-width: 980px) {{
    .grid {{ grid-template-columns: 1fr 1fr; }}
  }}
</style>
</head>
<body>
<h1>予想表（{target_date}）</h1>
<div class="meta">生成: {now} | 上位 {topk_per_race} 頭 / レース | 単勝 + 3連系（ランキング + 3連単TopK）</div>
""")

    # 目次
    parts.append('<div class="toc"><div style="margin-bottom:8px; font-weight:600;">レース一覧</div>')
    for rid in race_ids:
        parts.append(f'<a href="#r{rid}">{rid}</a>')
    parts.append("</div>")

    errors = []

    for rid in tqdm(race_ids, desc=f"Build HTML {target_date}"):
        try:
            feats = pfc.create_features(race_id=rid, predict=True, skip_agg_horse=True)

            # ===== 1) 単勝モデル =====
            pred_win = predict_win(
                feats,
                model_filepath=MODEL_DIR/"model.pkl",
                config_filepath="config.yaml",
                category_map_path=MODEL_DIR/"category_map.pkl",
            )
            pred_win = _attach_names_if_exist(pred_win, feats)

            dfw = pred_win.copy()
            if "tansyo_odds" in dfw.columns:
                dfw["EV(単勝)"] = dfw["pred"] * dfw["tansyo_odds"]

            cols_w = ["umaban", "horse_name", "jockey_name", "tansyo_odds", "popularity", "agari", "pred", "EV(単勝)"]
            cols_w = [c for c in cols_w if c in dfw.columns]
            dfw = dfw[cols_w].sort_values("pred", ascending=False).head(topk_per_race).reset_index(drop=True)
            dfw = dfw.rename(columns={
                "umaban": "馬番", "horse_name": "馬名", "jockey_name": "騎手",
                "tansyo_odds": "単勝", "popularity": "人気", "agari": "上り",
                "pred": "単勝Pred"
            })

            if "単勝Pred" in dfw.columns:
                dfw["単勝Pred"] = dfw["単勝Pred"].map(lambda x: f"{float(x):.3f}" if pd.notna(x) else "")
            if "単勝" in dfw.columns:
                dfw["単勝"] = dfw["単勝"].map(lambda x: f"{float(x):.1f}" if pd.notna(x) else "")
            if "EV(単勝)" in dfw.columns:
                dfw["EV(単勝)"] = dfw["EV(単勝)"].map(lambda x: f"{float(x):.2f}" if pd.notna(x) else "")

            # ===== 2) 3連系（ランキング + 3連単TopK） =====
            pred_rank_df, tri_df = predict_trifecta_topk(
                feats,
                model_filepath=MODEL_DIR/"model_tri.pkl",
                config_filepath="config_tri.yaml",
                category_map_path=MODEL_DIR/"category_map.pkl",
                cand_n=8,
                topk=30,
                temperature=1.0,
            )
            pred_rank_df = _attach_names_if_exist(pred_rank_df, feats)

            dfr = pred_rank_df.copy()
            cols_r = ["pred_rank", "umaban", "horse_name", "jockey_name", "tansyo_odds", "popularity", "agari", "score"]
            cols_r = [c for c in cols_r if c in dfr.columns]
            dfr = dfr[cols_r].sort_values("pred_rank").head(topk_per_race).reset_index(drop=True)

            dfr = dfr.rename(columns={
                "pred_rank": "予測順位", "umaban": "馬番", "horse_name": "馬名", "jockey_name": "騎手",
                "tansyo_odds": "単勝", "popularity": "人気", "agari": "上り", "score": "Score"
            })
            if "Score" in dfr.columns:
                dfr["Score"] = dfr["Score"].map(lambda x: f"{float(x):.3f}" if pd.notna(x) else "")
            if "単勝" in dfr.columns:
                dfr["単勝"] = dfr["単勝"].map(lambda x: f"{float(x):.1f}" if pd.notna(x) else "")

            tri = tri_df.copy()
            if "race_id" in tri.columns:
                tri = tri[tri["race_id"].astype(str) == str(rid)].copy()

            if len(tri) > 0:
                tri = tri[["1着", "2着", "3着", "prob"]].copy()
                tri = tri.rename(columns={"prob": "確率(近似)"})
                tri["確率(近似)"] = tri["確率(近似)"].map(lambda x: f"{float(x):.4f}" if pd.notna(x) else "")
            else:
                tri = pd.DataFrame([{"1着": "", "2着": "", "3着": "", "確率(近似)": "（候補不足）"}])

            # ===== HTML =====
            parts.append(f'<div class="race" id="r{rid}"><h2>Race {rid}</h2>')

            parts.append('<div class="grid">')

            parts.append('<div>')
            parts.append('<h3>単勝モデル（分類）</h3>')
            parts.append(dfw.to_html(index=False, escape=False))
            parts.append('<div class="note">EV(単勝)=単勝Pred×単勝オッズ</div>')
            parts.append('</div>')

            parts.append('<div>')
            parts.append('<h3>3連系モデル（ランキング）</h3>')
            parts.append(dfr.to_html(index=False, escape=False))
            parts.append('</div>')

            parts.append('</div>')  # grid

            parts.append('<h3>3連単 Top30（近似確率）</h3>')
            parts.append(tri.to_html(index=False, escape=False))
            parts.append('<div class="note">※確率はランキングScoreからの近似（Plackett–Luce）。レース内の相対比較に使ってください。</div>')

            parts.append('</div>')  # race

        except Exception as e:
            errors.append((rid, repr(e)))

        time.sleep(1.0)

    if errors:
        parts.append('<div class="race"><h2>予測できなかったレース</h2>')
        parts.append(pd.DataFrame(errors, columns=["race_id", "error"]).to_html(index=False, escape=False))
        parts.append("</div>")

    parts.append("</body></html>")

    out_path = OUT_DIR / f"predictions_{target_date}.html"
    out_path.write_text("\n".join(parts), encoding="utf-8")
    return out_path
