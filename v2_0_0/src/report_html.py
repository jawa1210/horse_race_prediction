# report_html.py
from __future__ import annotations

from pathlib import Path
import datetime as dt
import time
import math
import re

import pandas as pd
from tqdm.notebook import tqdm

from feature_engineering import PredictionFeatureCreator, RAW_DATA_DIR, DATA_DIR
from prediction import predict_win, predict_trifecta_topk
import yaml

from eval_strategies import (
    eval_strategies_for_date,
    load_cumulative_summary,
    format_summary_df_for_html,
)

# ========= Paths =========
POPULATION_CSV_DIR = RAW_DATA_DIR / "prediction_population"

OUT_DIR = DATA_DIR / "predictions_html"
OUT_DIR.mkdir(exist_ok=True, parents=True)

MODEL_DIR = Path("..", "data") / "03_train"
CONF_CALIB_PATH = MODEL_DIR / "confidence_calib_tri.yaml"

RESULTS_DIR = DATA_DIR / "results_cache"
RESULTS_DIR.mkdir(exist_ok=True, parents=True)
RESULTS_FLAT = RESULTS_DIR / "results_flat.csv"
POPULATION_INPUT_DIR = RAW_DATA_DIR/"prediction_population"
INPUT_DIR = DATA_DIR/"01_preprocessed"

PAYBACK_FLAT = RESULTS_DIR / "payouts_flat.csv"

def _load_payback_map(date: str) -> dict[str, pd.DataFrame]:
    """race_id -> 払戻表DataFrame（payouts_flat.csv: date,race_id,bet_type,combo,pay100）"""
    if not PAYBACK_FLAT.exists() or PAYBACK_FLAT.stat().st_size == 0:
        return {}

    try:
        df = pd.read_csv(PAYBACK_FLAT, dtype={"race_id": str, "bet_type": str, "combo": str})
    except pd.errors.EmptyDataError:
        return {}

    # ★ここが重要：pay100 を見る
    need = {"date", "race_id", "bet_type", "combo", "pay100"}
    if not need.issubset(df.columns):
        return {}

    df = df[df["date"].astype(str) == str(date)].copy()
    if df.empty:
        return {}

    # 表示用の券種名
    bt_map = {
        "tansho": "単勝",
        "fukusho": "複勝",
        "umaren": "馬連",
        "umatan": "馬単",
        "wide": "ワイド",
        "sanrenpuku": "三連複",
        "sanrentan": "三連単",
    }
    df["式別"] = df["bet_type"].map(lambda x: bt_map.get(str(x), str(x)))
    df["払戻(円/100円)"] = pd.to_numeric(df["pay100"], errors="coerce").fillna(0).astype(int)
    df["組み合わせ"] = df["combo"].astype(str)

    df = df[df["払戻(円/100円)"] > 0].copy()

    # 表示順
    order = ["単勝", "複勝", "馬連", "馬単","ワイド", "三連複", "三連単"]
    df["__ord"] = df["式別"].map(lambda x: order.index(x) if x in order else 999)
    df = df.sort_values(["race_id", "__ord", "払戻(円/100円)"], ascending=[True, True, False]).drop(columns="__ord")

    out: dict[str, pd.DataFrame] = {}
    for rid, g in df.groupby("race_id", sort=False):
        out[str(rid)] = g[["式別", "組み合わせ", "払戻(円/100円)"]].reset_index(drop=True)

    return out


# ========= Small helpers =========
def _attach_names_if_exist(df: pd.DataFrame, feats: pd.DataFrame) -> pd.DataFrame:
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


def _format_odds_html(x) -> str:
    if x is None or (isinstance(x, float) and pd.isna(x)) or x == "":
        return ""
    try:
        v = float(x)
    except Exception:
        return str(x)

    if v < 10:
        return f'<span class="odds1">{v:.1f}</span>'
    if v >= 100:
        return f'<span class="odds3">{v:.1f}</span>'
    return f"{v:.1f}"


def _format_pop_html(x) -> str:
    if x is None or (isinstance(x, float) and pd.isna(x)) or x == "":
        return ""
    try:
        v = int(float(x))
    except Exception:
        return str(x)

    if v <= 3:
        return f'<span class="pop1">{v:d}</span>'
    if v >= 10:
        return f'<span class="pop10">{v:d}</span>'
    return f"{v:d}"


def _load_results_map(date: str) -> dict[tuple[str, int], int]:
    if not RESULTS_FLAT.exists() or RESULTS_FLAT.stat().st_size == 0:
        return {}
    try:
        df = pd.read_csv(RESULTS_FLAT, dtype={"race_id": str})
    except pd.errors.EmptyDataError:
        return {}
    need = {"date", "race_id", "umaban", "rank"}
    if not need.issubset(df.columns):
        return {}
    df = df[df["date"].astype(str) == str(date)].copy()
    if df.empty:
        return {}
    df["race_id"] = df["race_id"].astype(str)
    df["umaban"] = pd.to_numeric(df["umaban"], errors="coerce")
    df["rank"] = pd.to_numeric(df["rank"], errors="coerce")
    df = df.dropna(subset=["race_id", "umaban", "rank"])

    out: dict[tuple[str, int], int] = {}
    for r in df.itertuples(index=False):
        try:
            out[(str(r.race_id), int(r.umaban))] = int(r.rank)
        except Exception:
            pass
    return out


def _result_class(rank: int | None) -> str | None:
    if rank is None:
        return None
    if rank == 1:
        return "hit-1"
    if rank <= 3:
        return "hit-3"
    if rank <= 5:
        return "hit-5"
    return None


def _circled_number(n: int) -> str:
    circled = {
        1: "①", 2: "②", 3: "③", 4: "④", 5: "⑤", 6: "⑥", 7: "⑦", 8: "⑧", 9: "⑨", 10: "⑩",
        11: "⑪", 12: "⑫", 13: "⑬", 14: "⑭", 15: "⑮", 16: "⑯", 17: "⑰", 18: "⑱", 19: "⑲", 20: "⑳"
    }
    try:
        n = int(n)
    except Exception:
        return ""
    return circled.get(n, str(n))


def _rank_icon(rank: int | None) -> str:
    if rank is None:
        return ""
    if rank == 1:
        return "🥇"
    if rank == 2:
        return "🥈"
    if rank == 3:
        return "🥉"
    return _circled_number(rank)


def _render_table_with_results(
    df: pd.DataFrame,
    rid: str,
    results_map: dict[tuple[str, int], int],
    extra_row_class_map: dict[int, str] | None = None,
) -> str:
    df = df.copy()
    extra_row_class_map = extra_row_class_map or {}

    ranks = []
    for _, row in df.iterrows():
        try:
            um = int(str(row.get("馬番")).split()[0])
        except Exception:
            um = None
        rk = results_map.get((str(rid), um)) if um is not None else None
        ranks.append("" if rk is None else str(rk))
    df.insert(0, "結果", ranks)

    cols = list(df.columns)
    ths = "".join([f"<th>{c}</th>" for c in cols])

    rows_html = []
    for _, row in df.iterrows():
        rk = row.get("結果")
        rk_i = int(rk) if str(rk).isdigit() else None
        cls1 = _result_class(rk_i)

        try:
            um = int(str(row.get("馬番")).split()[0])
        except Exception:
            um = None
        cls2 = extra_row_class_map.get(um)

        classes = " ".join([c for c in [cls1, cls2] if c])
        cls_attr = f' class="{classes}"' if classes else ""

        row_dict = row.to_dict()
        pred_tag = ""
        if cls2 == "win-strong":
            pred_tag = '<span class="predtag win">勝ち濃厚</span>'
        elif cls2 == "third-strong":
            pred_tag = '<span class="predtag third">3着濃厚</span>'

        if um is not None:
            icon = f' <span class="rankicon">{_rank_icon(rk_i)}</span>' if rk_i is not None else ""
            row_dict["馬番"] = f'{um}{icon}{pred_tag}'


        tds = "".join([f"<td>{row_dict.get(c, '')}</td>" for c in cols])
        rows_html.append(f"<tr{cls_attr}>{tds}</tr>")

    tbody = "\n".join(rows_html)
    return f"<table><thead><tr>{ths}</tr></thead><tbody>{tbody}</tbody></table>"


def _chaos_entropy_from_scores(scores, temperature: float = 1.0) -> float:
    s = pd.to_numeric(pd.Series(scores), errors="coerce").dropna().astype(float).values
    if len(s) == 0:
        return 0.0
    s = s - s.max()
    s = s / max(temperature, 1e-9)
    exps = [math.exp(float(x)) for x in s]
    denom = sum(exps)
    if denom <= 0:
        return 0.0
    p = [x / denom for x in exps]
    h = -sum([pi * math.log(pi + 1e-12) for pi in p])
    return float(h)


def _is_chaotic_race_by_triscore(pred_rank_df: pd.DataFrame, topn: int = 8, entropy_th: float = 1.80) -> tuple[bool, float]:
    if "score" not in pred_rank_df.columns:
        return False, 0.0
    s = pred_rank_df.sort_values("score", ascending=False)["score"].head(topn)
    h = _chaos_entropy_from_scores(s, temperature=1.0)
    return (h >= entropy_th), h


def _load_conf_calib() -> dict | None:
    if not CONF_CALIB_PATH.exists():
        return None
    try:
        with open(CONF_CALIB_PATH, "r") as f:
            return yaml.safe_load(f)
    except Exception:
        return None


def _conf_bucket_from_score(conf_score: float, calib: dict) -> int:
    th = calib.get("thresholds", {})
    q20 = float(th.get("q20", 0.0))
    q40 = float(th.get("q40", 0.0))
    q60 = float(th.get("q60", 0.0))
    q80 = float(th.get("q80", 0.0))
    if conf_score < q20:
        return 1
    if conf_score < q40:
        return 2
    if conf_score < q60:
        return 3
    if conf_score < q80:
        return 4
    return 5


def _conf_score_from_tri(entropy: float, marg: dict, entropy_th: float = 1.80) -> float:
    top5 = float(marg.get("top5_mass", 0.0))
    top1 = float(marg.get("top1_prob", 0.0))
    p1max = float(marg.get("p1_max", 0.0))
    p1gap = float(marg.get("p1_gap", 0.0))
    e_norm = min(max(entropy / max(entropy_th, 1e-9), 0.0), 2.0)
    score = (0.45 * top5 + 0.20 * min(top1 * 2.0, 1.0) + 0.25 * p1max + 0.20 * min(p1gap * 2.0, 1.0)) - (0.55 * min(e_norm / 1.2, 1.0))
    return float(score)


def _marginals_from_tri_df(tri_df: pd.DataFrame) -> dict:
    if tri_df is None or len(tri_df) == 0:
        return {"p1_map": {}, "p3_map": {}, "p1_max": 0.0, "p1_gap": 0.0, "top1_prob": 0.0, "top5_mass": 0.0}
    x = tri_df.copy()
    x["prob"] = pd.to_numeric(x["prob"], errors="coerce").fillna(0.0)
    mass = float(x["prob"].sum())
    if mass <= 0:
        return {"p1_map": {}, "p3_map": {}, "p1_max": 0.0, "p1_gap": 0.0, "top1_prob": 0.0, "top5_mass": 0.0}
    x["p"] = x["prob"] / mass
    p1 = x.groupby("1着")["p"].sum().sort_values(ascending=False)
    p3 = x.groupby("3着")["p"].sum().sort_values(ascending=False)
    p1_max = float(p1.iloc[0]) if len(p1) >= 1 else 0.0
    p1_2nd = float(p1.iloc[1]) if len(p1) >= 2 else 0.0
    p1_gap = p1_max - p1_2nd
    top1_prob = float(x["p"].sort_values(ascending=False).head(1).sum())
    top5_mass = float(x["p"].sort_values(ascending=False).head(5).sum())
    return {
        "p1_map": p1.to_dict(), "p3_map": p3.to_dict(),
        "p1_max": p1_max, "p1_gap": p1_gap,
        "top1_prob": top1_prob, "top5_mass": top5_mass,
    }


def _pick_win_and_third_by_marginals(marg: dict) -> tuple[int | None, int | None]:
    p1 = marg.get("p1_map", {})
    p3 = marg.get("p3_map", {})

    def top2(d):
        items = sorted(d.items(), key=lambda kv: kv[1], reverse=True)
        if len(items) == 0:
            return (None, 0.0, None, 0.0)
        if len(items) == 1:
            return (items[0][0], float(items[0][1]), None, 0.0)
        return (items[0][0], float(items[0][1]), items[1][0], float(items[1][1]))

    w1, wv1, w2, wv2 = top2(p1)
    t1, tv1, t2, tv2 = top2(p3)

    win_um = None
    third_um = None

    if w1 is not None and (wv1 >= 0.35) and ((wv1 - wv2) >= 0.12):
        try:
            win_um = int(float(w1))
        except Exception:
            win_um = None

    if t1 is not None and (tv1 >= 0.22) and ((tv1 - tv2) >= 0.07):
        try:
            third_um = int(float(t1))
        except Exception:
            third_um = None

    return win_um, third_um


def _legend_html() -> str:
    return """
    <div class="legend">
      <h2 style="margin:0 0 8px 0;">凡例（表示項目の説明）</h2>
      <div class="legend-grid">
        <div class="legend-item"><div class="legend-title">的中率/回収率（戦略サマリ）</div>
          <div class="legend-body">
            的中率(レース): その戦略で「そのレースで何か1つでも当たった」割合。<br>
            的中率(券): 当たった券枚数 / 買った券枚数。<br>
            回収率(ROI): 総払戻 / 総投資 × 100。払戻は <b>100円単位</b>。
          </div>
        </div>
        <div class="legend-item"><div class="legend-title">単勝Pred / EV(単勝)</div>
          <div class="legend-body">
            単勝Pred: 単勝モデルの1着確率推定。<br>
            EV(単勝)=単勝Pred×単勝オッズ（目安）。
          </div>
        </div>
        <div class="legend-item"><div class="legend-title">荒れ指数（Entropy）</div>
          <div class="legend-body">
            3連系スコア上位をsoftmaxした分布のエントロピー。大きいほど団子で荒れやすい。
          </div>
        </div>
        <div class="legend-item"><div class="legend-title">自信度（1〜5）</div>
          <div class="legend-body">
            TopKの集中度＋勝ち馬の固さ−荒れ度から conf_score を作り、較正ファイルの分位で1〜5表示（無い場合は暫定）。
          </div>
        </div>
        <div class="legend-item"><div class="legend-title">金/水色行（勝ち濃厚 / 3着濃厚）</div>
          <div class="legend-body">
            3連単TopKから擬似周辺確率を作り「勝ち濃厚」「3着濃厚」を強調。
          </div>
        </div>
        <div class="legend-item"><div class="legend-title">🥇🥈🥉 / ④⑤⑥…</div>
          <div class="legend-body">
            results_flat（実着順）に基づいて馬番横に実順位を表示。
          </div>
        </div>
      </div>
    </div>
    """


# ========= Main =========

def build_html_report(
    target_date: str,
    read_file_name_csv: str = "population.csv",
    read_horse_name_csv: str = "horse_results_prediction.csv",
    topk_per_race: int = 18,
    stake_per_ticket: int = 100,
    roi_mode: str = "both",
    skip_agg_horse: bool=True
) -> Path:
    population_csv = Path(read_file_name_csv)
    horse_csv = Path(read_horse_name_csv)

    payback_by_race = _load_payback_map(target_date)


    # build_html_report 内でも「入力DIR補完」したいならここでやる
    if population_csv.parent == Path("."):
        population_csv = POPULATION_INPUT_DIR / population_csv
    population_csv = population_csv.resolve()

    if not population_csv.exists():
        raise FileNotFoundError(f"population_csv not found: {population_csv}")

    # horse_csv も population と同じく「絶対化 + 存在チェック」
    horse_csv = Path(read_horse_name_csv)
    if horse_csv.parent == Path("."):
        horse_csv = INPUT_DIR / horse_csv
    horse_csv = horse_csv.resolve()

    if not horse_csv.exists():
        raise FileNotFoundError(f"horse_csv not found: {horse_csv}")

    pop = pd.read_csv(population_csv, sep="\t", dtype={"race_id": str})
    pop["date"] = pop["date"].astype(str).str.strip()

    race_ids_all = sorted(pop.loc[pop["date"] == target_date, "race_id"].astype(str).unique())
    if len(race_ids_all) == 0:
        raise ValueError(f"{target_date} の race_id が {population_csv} にありません")

    pfc = PredictionFeatureCreator(population_file_name=population_csv,
                                   horse_results_prediction_feile_name=horse_csv)
    pfc.agg_horse_n_races()

    now = dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    results_map = _load_results_map(target_date)
    has_results = len(results_map) > 0
    conf_calib = _load_conf_calib()

    # 暫定（結果が1頭でも入ってるレースだけ）
    race_ids_done = sorted(set([rid for (rid, _um) in results_map.keys()]) & set(race_ids_all))

    # ★戦略評価用に溜める
    pred_order_by_race: dict[str, list[int]] = {}
    odds_map_by_race: dict[str, dict[int, float]] = {}

    parts: list[str] = []
    parts.append(f"""<!doctype html>
<html lang="ja"><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>予想表 {target_date}</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Noto Sans JP", "Hiragino Sans", "Yu Gothic", sans-serif;
         margin: 24px; background: #fafafa; color: #111; }}
  h1 {{ margin: 0 0 8px 0; font-size: 24px; }}
  .meta {{ color: #444; margin-bottom: 10px; }}

  .toc {{ background: #fff; border-radius: 12px; padding: 14px 16px; box-shadow: 0 2px 10px rgba(0,0,0,0.06); margin-bottom: 18px; }}
  .toc a {{ text-decoration: none; margin-right: 10px; white-space: nowrap; display: inline-block; padding: 6px 10px; border-radius: 999px; background: #f1f3f5; color: #111; }}

  .race {{ background: #fff; border-radius: 14px; padding: 14px 16px; box-shadow: 0 2px 12px rgba(0,0,0,0.07); margin: 14px 0; }}
  .race h2 {{ margin: 0 0 8px 0; font-size: 18px; }}
  .race.race-chaos h2 {{ color: #d60000; background: #fff0f0; border-left: 6px solid #d60000; padding-left: 10px; }}

  h3 {{ margin: 14px 0 8px 0; font-size: 16px; }}
  table {{ border-collapse: collapse; width: 100%; margin-bottom: 6px; }}
  th, td {{ border-bottom: 1px solid #eee; padding: 8px 8px; text-align: left; font-size: 14px; }}
  th {{ background: #fbfbfb; position: sticky; top: 0; z-index: 1; }}

  .note {{ color:#555; font-size: 13px; margin-top: 8px; }}
  .grid {{ display: grid; grid-template-columns: 1fr; gap: 12px; }}
  @media (min-width: 980px) {{ .grid {{ grid-template-columns: 1fr 1fr; }} }}

  .odds1 {{ color:#d60000; font-weight:600; }}
  .odds3 {{ color:#0057d8; font-weight:600; }}

  tr.hit-1 {{ background:#d9f7e6; font-weight:700; }}
  tr.hit-3 {{ background:#e7f0ff; }}
  tr.hit-5 {{ background:#f3f4ff; }}

  .pop1 {{ color:#d60000; font-weight:600; }}
  .pop10{{ color:#0057d8; font-weight:600; }}

  .rankicon {{ margin-left:6px; }}

  .badge {{ display:inline-block; padding:4px 10px; border-radius:999px; font-weight:700; font-size:12px; vertical-align:middle; margin-left:8px; }}
  .badge.c1 {{ background:#f1f3f5; color:#444; }}
  .badge.c2 {{ background:#e7f0ff; color:#1f4dd8; }}
  .badge.c3 {{ background:#e8fff2; color:#0a7a3a; }}
  .badge.c4 {{ background:#fff1db; color:#a85a00; }}
  .badge.c5 {{ background:#ffe0e0; color:#c40000; }}

  .legend {{ background:#fff; border-radius: 14px; padding: 14px 16px; box-shadow: 0 2px 12px rgba(0,0,0,0.07); margin: 14px 0 18px 0; }}
  .legend-grid {{ display:grid; grid-template-columns: 1fr; gap: 10px; }}
  @media (min-width: 980px) {{ .legend-grid {{ grid-template-columns: 1fr 1fr; }} }}
  .legend-item {{ border:1px solid #f0f0f0; border-radius: 12px; padding: 10px 12px; background:#fcfcfc; }}
  .legend-title {{ font-weight: 800; margin-bottom: 6px; }}
  .legend-body {{ font-size: 13px; color:#333; line-height: 1.6; }}

  .summary {{ background:#fff; border-radius: 14px; padding: 14px 16px; box-shadow: 0 2px 12px rgba(0,0,0,0.07); margin: 14px 0; }}
  .summary h2 {{ margin: 0 0 8px 0; font-size: 18px; }}

  /* 暫定/最終切替UI */
  .modebar {{ background:#fff; border-radius:12px; padding:10px 12px; box-shadow:0 2px 10px rgba(0,0,0,0.06); margin: 14px 0; display:flex; gap:8px; align-items:center; flex-wrap:wrap; }}
  .modebtn {{ border:1px solid #e5e5e5; background:#f7f7f7; padding:6px 10px; border-radius:999px; cursor:pointer; font-weight:700; font-size:13px; }}
  .modebtn.active {{ background:#111; color:#fff; border-color:#111; }}
  .modebox {{ display:none; }}
  .modebox.active {{ display:block; }}

/* 予想強調は「左端セルのボーダー＋ラベル」にする（結果背景と混ざらない） */
tr.win-strong td:first-child {{ border-left: 6px solid #e0b800; }}
tr.third-strong td:first-child {{ border-left: 6px solid #00a3d8; }}

/* 予想ラベル */
.predtag{{
  display:inline-block;
  padding:2px 8px;
  border-radius:999px;
  font-size:12px;
  font-weight:800;
  margin-left:8px;
  border:1px solid transparent;
}}
.predtag.win {{ background:#fff6cc; color:#6b5500; border-color:#e9d98b; }}
.predtag.third {{ background:#e9f7ff; color:#004a63; border-color:#a9dff1; }}

</style>
</head><body>
<h1>予想表（{target_date}）</h1>
<div class="meta">生成: {now} | 上位 {topk_per_race} 頭 / レース</div>
<div class="meta">対象population: {read_file_name_csv} / 結果反映: {"あり" if has_results else "なし"}（results_flat.csv）</div>
<div class="meta">暫定の確定済みレース: {len(race_ids_done)} / {len(race_ids_all)}（結果が入ってるレースのみ集計可能）</div>
""")

    parts.append(_legend_html())

    # ---- まずサマリ枠（あとで中身を差し込む） ----
    parts.append("""
<div class="modebar">
  <span style="font-weight:800;">サマリ表示:</span>
  <button class="modebtn active" data-mode="provisional">暫定（確定済みのみ）</button>
  <button class="modebtn" data-mode="final">最終（当日全体）</button>
  <span class="meta" id="modeNote"></span>
</div>

<div id="box-provisional" class="modebox active"></div>
<div id="box-final" class="modebox"></div>
<div id="box-cumulative" class="modebox active"></div>

<script>
(function(){
  const btns = document.querySelectorAll(".modebtn");
  const boxP = document.getElementById("box-provisional");
  const boxF = document.getElementById("box-final");
  const note = document.getElementById("modeNote");

  function setMode(mode){
    btns.forEach(b => b.classList.toggle("active", b.dataset.mode===mode));
    boxP.classList.toggle("active", mode==="provisional");
    boxF.classList.toggle("active", mode==="final");
    if(mode==="provisional") note.textContent = "（結果・払戻が揃っているレースだけで計算）";
    if(mode==="final") note.textContent = "（当日全体。未確定があると参考値）";
  }
  btns.forEach(b => b.addEventListener("click", () => setMode(b.dataset.mode)));
  setMode("provisional");
})();
</script>
""")

    # 目次
    parts.append('<div class="toc"><div style="margin-bottom:8px; font-weight:600;">レース一覧</div>')
    for rid in race_ids_all:
        parts.append(f'<a href="#r{rid}">{rid}</a>')
    parts.append("</div>")

    errors = []

    for rid in tqdm(race_ids_all, desc=f"Build HTML {target_date}"):
        try:
            feats = pfc.create_features(race_id=rid, predict=True, skip_agg_horse=skip_agg_horse)

            # --- 単勝
            pred_win = predict_win(
                feats,
                model_filepath=MODEL_DIR / "model.pkl",
                config_filepath="config.yaml",
                category_map_path=MODEL_DIR / "category_map.pkl",
            )
            pred_win = _attach_names_if_exist(pred_win, feats)

            # odds_map_by_race（戦略用）
            odds_map: dict[int, float] = {}
            if "tansyo_odds" in pred_win.columns and "umaban" in pred_win.columns:
                tmp = pred_win[["umaban", "tansyo_odds"]].copy()
                tmp["umaban"] = pd.to_numeric(tmp["umaban"], errors="coerce")
                tmp["tansyo_odds"] = pd.to_numeric(tmp["tansyo_odds"], errors="coerce")
                tmp = tmp.dropna()
                for r in tmp.itertuples(index=False):
                    odds_map[int(r.umaban)] = float(r.tansyo_odds)
            odds_map_by_race[str(rid)] = odds_map

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
                dfw["単勝"] = dfw["単勝"].map(_format_odds_html)
            if "人気" in dfw.columns:
                dfw["人気"] = dfw["人気"].map(_format_pop_html)
            if "EV(単勝)" in dfw.columns:
                dfw["EV(単勝)"] = dfw["EV(単勝)"].map(lambda x: f"{float(x):.2f}" if pd.notna(x) else "")

            # --- 3連系（ランキング + TopK）
            pred_rank_df, tri_df = predict_trifecta_topk(
                feats,
                model_filepath=MODEL_DIR / "model_tri.pkl",
                config_filepath="config_tri.yaml",
                category_map_path=MODEL_DIR / "category_map.pkl",
                cand_n=8,
                topk=30,
                temperature=1.0,
            )
            pred_rank_df = _attach_names_if_exist(pred_rank_df, feats)

            is_chaos, chaos_h = _is_chaotic_race_by_triscore(pred_rank_df, topn=8, entropy_th=1.80)
            marg = _marginals_from_tri_df(tri_df)
            win_um, third_um = _pick_win_and_third_by_marginals(marg)

            row_class_map: dict[int, str] = {}
            if win_um is not None:
                row_class_map[int(win_um)] = "win-strong"
            if third_um is not None:
                row_class_map.setdefault(int(third_um), "third-strong")

            if conf_calib is not None:
                entropy_th = float(conf_calib.get("entropy_th", 1.80))
                conf_score = _conf_score_from_tri(entropy=chaos_h, marg=marg, entropy_th=entropy_th)
                conf = _conf_bucket_from_score(conf_score, conf_calib)
                hr = conf_calib.get("hit_rate", {}).get(conf, None)
                exp_hit = (f"{float(hr)*100:.1f}%" if hr is not None else "")
            else:
                conf, exp_hit = 3, ""

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
                dfr["単勝"] = dfr["単勝"].map(_format_odds_html)
            if "人気" in dfr.columns:
                dfr["人気"] = dfr["人気"].map(_format_pop_html)

            # ★戦略用：予測順位の馬番リスト
            pred_order: list[int] = []
            if "馬番" in dfr.columns:
                for x in dfr["馬番"].tolist():
                    try:
                        pred_order.append(int(str(x).split()[0]))
                    except Exception:
                        pass
            pred_order_by_race[str(rid)] = pred_order

            tri = tri_df.copy()
            if "race_id" in tri.columns:
                tri = tri[tri["race_id"].astype(str) == str(rid)].copy()
            if len(tri) > 0:
                tri = tri[["1着", "2着", "3着", "prob"]].copy()
                tri = tri.rename(columns={"prob": "確率(近似)"})
                tri["確率(近似)"] = tri["確率(近似)"].map(lambda x: f"{float(x):.4f}" if pd.notna(x) else "")
            else:
                tri = pd.DataFrame([{"1着": "", "2着": "", "3着": "", "確率(近似)": "（候補不足）"}])

            race_div_class = "race race-chaos" if is_chaos else "race"
            badge_text = f'自信度 {conf}/5' + (f'（hit@K {exp_hit}）' if exp_hit else '')
            badge = f'<span class="badge c{conf}">{badge_text}</span>'

            parts.append(f'<div class="{race_div_class}" id="r{rid}"><h2>Race {rid}{badge}</h2>')
            parts.append(f'<div class="note">荒れ指数(Entropy, top8 score): {chaos_h:.3f}（閾値 1.80 以上で赤）</div>')

            parts.append('<div class="grid">')

            parts.append('<div>')
            parts.append('<h3>単勝モデル（分類）</h3>')
            parts.append(_render_table_with_results(dfw, rid, results_map) if has_results else dfw.to_html(index=False, escape=False))
            parts.append('<div class="note">EV(単勝)=単勝Pred×単勝オッズ</div>')
            parts.append('</div>')

            parts.append('<div>')
            parts.append('<h3>3連系モデル（ランキング）</h3>')
            parts.append(_render_table_with_results(dfr, rid, results_map, extra_row_class_map=row_class_map)
                         if has_results else _render_table_with_results(dfr, rid, {}, extra_row_class_map=row_class_map))
            parts.append('</div>')

            parts.append('</div>')  # grid

            parts.append('<h3>3連単 Top30（近似確率）</h3>')
            parts.append(tri.to_html(index=False, escape=False))
            parts.append('<div class="note">※確率はランキングScoreからの近似（Plackett–Luce）。レース内の相対比較に使ってください。</div>')

            # --- 払戻（結果が確定しているときだけ表示）---
            pay_df = payback_by_race.get(str(rid))
            if pay_df is not None and len(pay_df) > 0:
                parts.append('<h3>払い戻し（確定）</h3>')
                parts.append(pay_df.to_html(index=False, escape=False))
            else:
                # 出さないなら何もしない、出すなら薄く「未確定」を出す
                parts.append('<div class="note">払い戻し：未確定</div>')

            parts.append('</div>')  # race


        except Exception as e:
            errors.append((rid, repr(e)))

        time.sleep(0.2)

    # ---- サマリ生成（暫定/最終 切替） ----
    def _summary_block(title: str, df: pd.DataFrame | None) -> str:
        if df is None or len(df) == 0:
            return f"<div class='summary'><h2>{title}</h2><div class='note'>（データなし）</div></div>"
        return f"<div class='summary'><h2>{title}</h2>{df.to_html(index=False, escape=False)}</div>"

    provisional_html = "<div class='summary'><h2>暫定サマリ</h2><div class='note'>未計算</div></div>"
    final_html = "<div class='summary'><h2>最終サマリ</h2><div class='note'>未計算</div></div>"
    cumulative_html = ""

    try:
        # 暫定：結果が入ってるレースだけ（途中経過）
        if roi_mode in ["both", "provisional"]:
            pr_eval, pr_venue, pr_total = eval_strategies_for_date(
                target_date=target_date,
                race_ids=race_ids_done,
                pred_order_by_race=pred_order_by_race,
                odds_map_by_race=odds_map_by_race,
                stake_per_ticket=stake_per_ticket,
            )
            pr_total_disp = format_summary_df_for_html(pr_total)
            pr_venue_disp = format_summary_df_for_html(pr_venue)
            provisional_html = (
                _summary_block(f"暫定（確定済み {len(race_ids_done)}/{len(race_ids_all)} レース）当日サマリ（戦略別）", pr_total_disp)
                + _summary_block("暫定：開催場所別サマリ（戦略別）", pr_venue_disp)
            )

        # 最終：当日全レース（未確定があると参考値）
        if roi_mode in ["both", "final"]:
            fn_eval, fn_venue, fn_total = eval_strategies_for_date(
                target_date=target_date,
                race_ids=race_ids_all,
                pred_order_by_race=pred_order_by_race,
                odds_map_by_race=odds_map_by_race,
                stake_per_ticket=stake_per_ticket,
            )
            fn_total_disp = format_summary_df_for_html(fn_total)
            fn_venue_disp = format_summary_df_for_html(fn_venue)
            final_html = (
                _summary_block(f"最終（当日全 {len(race_ids_all)} レース）当日サマリ（戦略別）", fn_total_disp)
                + _summary_block("最終：開催場所別サマリ（戦略別）", fn_venue_disp)
            )

        # 累計（常に表示）
        cum = load_cumulative_summary()
        cum_disp = format_summary_df_for_html(cum) if cum is not None else None
        if cum_disp is not None and len(cum_disp):
            cumulative_html = _summary_block("累計サマリ（戦略別）", cum_disp)

    except Exception as e:
        provisional_html = f"<div class='summary'><h2>暫定サマリ</h2><div class='note'>サマリ生成に失敗: {repr(e)}</div></div>"
        final_html = f"<div class='summary'><h2>最終サマリ</h2><div class='note'>サマリ生成に失敗: {repr(e)}</div></div>"

    # 先頭の modebox を中身で埋める（文字列置換）
    html_all = "\n".join(parts)
    html_all = html_all.replace('<div id="box-provisional" class="modebox active"></div>',
                                f'<div id="box-provisional" class="modebox active">{provisional_html}</div>')
    html_all = html_all.replace('<div id="box-final" class="modebox"></div>',
                                f'<div id="box-final" class="modebox">{final_html}</div>')
    # 累計は常時表示でよいので、暫定ボックスの下に置く（modebox active）
    html_all = html_all.replace('<div id="box-cumulative" class="modebox active"></div>',
                                f'<div id="box-cumulative" class="modebox active">{cumulative_html}</div>')

    # エラー表
    if errors:
        html_all += '<div class="race"><h2>予測できなかったレース</h2>'
        html_all += pd.DataFrame(errors, columns=["race_id", "error"]).to_html(index=False, escape=False)
        html_all += "</div>"

    html_all += "</body></html>"

    out_path = OUT_DIR / f"predictions_{target_date}.html"
    out_path.write_text(html_all, encoding="utf-8")
    return out_path
