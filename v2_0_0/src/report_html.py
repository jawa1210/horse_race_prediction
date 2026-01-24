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

from strategies import build_strategy_catalog
from eval_bets import (
    actual_outcomes_from_results_map,
    payout_map_for_race,
    eval_one_race_tickets,
)

# feature_engineering.py に mapping がある前提（なければ同等の辞書をここに用意）
from feature_engineering import (
    race_course_mapping,   # 例: {"中山": 5, ...}
    race_type_mapping,     # 例: {"芝": 0, "ダ": 1, ...}
    ground_state_mapping,  # 例: {"良": 0, "稍重": 1, ...}
)

_INV_PLACE  = {v: k for k, v in race_course_mapping.items()}
_INV_RTYPE  = {v: k for k, v in race_type_mapping.items()}
_INV_GROUND = {v: k for k, v in ground_state_mapping.items()}

# --- netkeiba race_id 用（JRA）: PP=01..10 ---
_NETKEIBA_JRA_PLACE = {
    1: "札幌",
    2: "函館",
    3: "福島",
    4: "新潟",
    5: "東京",
    6: "中山",
    7: "中京",
    8: "京都",
    9: "阪神",
    10: "小倉",
}

def _place_from_race_id_netkeiba(rid: str) -> str:
    """
    netkeiba race_id(12桁想定): YYYYPPKKDDRR
      PP = 開催場コード(01..10)
    """
    s = re.sub(r"\D", "", str(rid))
    if len(s) >= 12:
        # netkeiba の基本形: YYYYPPKKDDRR
        try:
            pp = int(s[4:6])
            return _NETKEIBA_JRA_PLACE.get(pp, f"場{pp:02d}")
        except Exception:
            return ""
    # フォールバック（もし別形式が混ざった時用）
    if len(s) >= 4:
        try:
            pp = int(s[-4:-2])
            return _NETKEIBA_JRA_PLACE.get(pp, f"場{pp:02d}")
        except Exception:
            return ""
    return ""




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

def _strategy_label_ja(name: str) -> str:
    """
    入力例:
      sanrentan_1-234-234     -> 三連単①-②③④-②③④
      sanrentan_1-23-2345     -> 三連単①-②③-②③④⑤
      sanrenpuku_box4         -> 三連複BOX4
      umatan_nagashi4         -> 馬単流4
      wide_nagashi5           -> ワイド流5
      tansho_p1               -> 単勝①
    """
    s = str(name)

    # bet type 部分を判定（先頭の種別）
    bet_map = {
        "sanrentan": "三連単",
        "sanrenpuku": "三連複",
        "umatan": "馬単",
        "umaren": "馬連",
        "wide": "ワイド",
        "tansho": "単勝",
        "fukusho": "複勝",
    }

    bet = ""
    rest = s
    for k, v in bet_map.items():
        if s.startswith(k + "_"):
            bet = v
            rest = s[len(k) + 1:]  # 先頭 "k_" を落とす
            break
        if s == k:
            bet = v
            rest = ""
            break

    if bet == "":
        # 変換できないものはそのまま（保険）
        return s

    # パターン1: box4 / box5
    m = re.fullmatch(r"box(\d+)", rest)
    if m:
        return f"{bet}BOX{m.group(1)}"

    # パターン2: nagashi4 / nagashi5
    m = re.fullmatch(r"nagashi(\d+)", rest)
    if m:
        return f"{bet}流{m.group(1)}"

    # パターン3: p1 (単勝1点とか)
    m = re.fullmatch(r"p(\d+)", rest)
    if m:
        n = int(m.group(1))
        return f"{bet}{_circled_number(n)}"

    # パターン4: フォーメーション 1-234-234 / 1-23-2345 など
    # rest が "1-234-234" の形式なら、数字列を「②③④」みたいに展開
    def _expand_group(g: str) -> str:
        # "234" -> "②③④" / "12" -> "①②" / "8" -> "⑧"
        out = []
        for ch in g:
            if ch.isdigit():
                out.append(_circled_number(int(ch)))
        return "".join(out) if out else g

    if re.fullmatch(r"[0-9]+(-[0-9]+)+", rest):
        parts = rest.split("-")
        parts_ja = [_expand_group(p) for p in parts]
        return bet + ("-".join(parts_ja))

    # それ以外（念のため）
    # 例: "box4_extra" みたいなのが来たら詰めて返す
    rest = rest.replace("nagashi", "流").replace("box", "BOX").replace("_", "")
    return bet + rest


def _roi_to_mult_text(roi_percent: float) -> str:
    """ROI% -> +倍率（+39.6倍）"""
    try:
        roi = float(roi_percent)
    except Exception:
        return ""
    mult = roi / 100.0
    sign = "+" if mult >= 0 else ""
    return f"{sign}{mult:.1f}倍"

def _mult_class_from_roi(roi_percent: float) -> str:
    """
    ROI% -> 倍率クラス（濃さ）
    目安:
      <2倍: m0
      <5倍: m1
      <10倍: m2
      <20倍: m3
      >=20倍: m4
    """
    try:
        mult = float(roi_percent) / 100.0
    except Exception:
        return "m0"
    if mult < 2.0:
        return "m0"
    if mult < 5.0:
        return "m1"
    if mult < 10.0:
        return "m2"
    if mult < 20.0:
        return "m3"
    return "m4"


def _roi_to_mult_html(roi_percent: float) -> str:
    """ROI% -> <span class="mult mX">+39.6倍</span>"""
    txt = _roi_to_mult_text(roi_percent)
    cls = _mult_class_from_roi(roi_percent)
    if txt == "":
        return ""
    return f'<span class="mult {cls}">{txt}</span>'



def _race_condition_text(feats: pd.DataFrame) -> str:
    """feats(1レース全馬)から見出し用の条件文字列を作る: '中山 芝1600m 良'"""
    if feats is None or len(feats) == 0:
        return ""

    r0 = feats.iloc[0]

    def _get_int(name: str) -> int | None:
        if name not in feats.columns:
            return None
        v = r0.get(name)
        if pd.isna(v):
            return None
        try:
            return int(v)
        except Exception:
            return None

    place_code  = _get_int("place")
    rtype_code  = _get_int("race_type")
    ground_code = _get_int("ground_state")

    # course_len は int じゃない時があるので別処理
    dist = None
    if "course_len" in feats.columns:
        v = r0.get("course_len")
        if pd.notna(v):
            try:
                dist = int(float(v))
            except Exception:
                dist = None

    place  = _INV_PLACE.get(place_code, "")
    rtype  = _INV_RTYPE.get(rtype_code, "")
    ground = _INV_GROUND.get(ground_code, "")

    s = " ".join([x for x in [place, f"{rtype}{dist}m" if (rtype or dist) else "", ground] if x])
    return s

def _toc_label_from_race_id(rid: str) -> str:
    rid = str(rid)
    place = _place_from_race_id_netkeiba(rid)

    rnum = ""
    try:
        rnum = f"{int(rid[-2:])}R"
    except Exception:
        pass

    return " ".join([x for x in [place, rnum] if x]) or rid


def add_agari_badge(df: pd.DataFrame, col="上り") -> pd.DataFrame:
    df = df.copy()
    if col not in df.columns:
        return df

    a = pd.to_numeric(df[col], errors="coerce")

    # 小さいほど速い → dense rank（同着は同じ順位）
    r = a.rank(method="dense", ascending=True)

    def _badge(v, rk):
        if pd.isna(v):
            return ""
        v = float(v)
        if pd.isna(rk):
            return f"{v:.1f}"
        rk = int(rk)
        if rk == 1:
            return f'<span class="agari g">{v:.1f}</span>'  # gold
        if rk == 2:
            return f'<span class="agari s">{v:.1f}</span>'  # silver
        if rk == 3:
            return f'<span class="agari b">{v:.1f}</span>'  # bronze
        return f"{v:.1f}"

    df[col] = [
        _badge(v, rk) for v, rk in zip(a.tolist(), r.tolist())
    ]
    return df


def _hit_badge_html_for_race(
    target_date: str,
    rid: str,
    pred_order: list[int],
    odds_map: dict[int, float],
    results_map: dict[tuple[str, int], int],
    payouts_df: pd.DataFrame,
    stake_per_ticket: int,
) -> str:
    outcomes = actual_outcomes_from_results_map(results_map, rid)
    if not outcomes:
        return ""

    pay_map = payout_map_for_race(payouts_df, target_date, rid)
    if not pay_map:
        return ""

    catalog = build_strategy_catalog()

    hit_items: list[tuple[str, float, dict]] = []
    for strat_name, gen in catalog:
        tickets = gen(pred_order, odds_map)
        ev = eval_one_race_tickets(
            rid=rid,
            tickets=tickets,
            outcomes=outcomes,
            pay_map=pay_map,
            stake_per_ticket=stake_per_ticket,
        )
        ret = float(ev.get("return", 0.0))
        stake = float(ev.get("stake", 0.0))
        roi = float(ev.get("roi", (ret / stake * 100.0) if stake > 0 else 0.0))
        if ret > 0:
            hit_items.append((strat_name, roi, ev))

    if not hit_items:
        return ""

    hit_items.sort(key=lambda x: x[1], reverse=True)

    # ---- 表示短縮：上位3つ + 他+N ----
    show_k = 3
    shown = hit_items[:show_k]
    rest_n = max(0, len(hit_items) - len(shown))

    inside_parts = []
    for name, roi, _ev in shown:
        inside_parts.append(f"{_strategy_label_ja(name)}｜{_roi_to_mult_html(roi)}")

    inside = " / ".join(inside_parts)
    if rest_n > 0:
        inside += f" / 他+{rest_n}"

    detail_id = f"hitdetail-{rid}"
    badge_html = (
        f'<span class="hitbadge" onclick="toggleHitDetail(\'{detail_id}\')">'
        f'的中🎯({inside})'
        f'</span>'
    )

    # ---- 折りたたみ内訳：全部出す（同じ表記・色付き）----
    detail_lines = []
    for strat_name, roi, _ev in hit_items:
        ja = _strategy_label_ja(strat_name)
        detail_lines.append(f"<li><b>{ja}</b>｜{_roi_to_mult_html(roi)}</li>")

    detail_html = (
        f'<div id="{detail_id}" class="hitdetail">'
        f'<b>的中内訳</b>'
        f'<ul>{"".join(detail_lines)}</ul>'
        f'</div>'
    )

    return badge_html + detail_html


def _stars(conf: int) -> str:
    conf = int(conf) if conf is not None else 3
    conf = max(1, min(5, conf))
    return "★" * conf + "☆" * (5 - conf)

def _chaos_label(entropy: float) -> str:
    """
    entropy から荒れ度ラベル
    目安は調整してOK（とりあえず分かりやすい閾値）
    """
    try:
        h = float(entropy)
    except Exception:
        return ""
    if h >= 2.10:
        return "大荒"
    if h >= 1.90:
        return "中荒"
    if h >= 1.75:
        return "小荒"
    return ""

def _fmt_roi_red(x) -> str:
    """
    回収率の色分け（HTML）
      - >=200%: 濃い赤
      - >=100%: 薄い赤
      - <100% : 通常
    """
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return ""
    s = str(x).strip()
    try:
        v = float(s.replace("%", ""))
    except Exception:
        return s

    # 表示文字
    txt = f"{v:.1f}%"

    if v >= 200.0:
        # 濃赤（視認性重視）
        return f'<span style="color:#b00000; font-weight:900;">{txt}</span>'
    if v >= 100.0:
        # 薄赤
        return f'<span style="color:#e05a5a; font-weight:900;">{txt}</span>'
    return txt



def _format_summary_table(df: pd.DataFrame, mode: str) -> pd.DataFrame:
    """
    mode:
      - "total": 暫定/最終（戦略別）
      - "venue": 暫定/最終（開催場所別）
      - "cum": 累計（戦略別）
    """
    if df is None or len(df) == 0:
        return df

    out = df.copy()

    # 1) 列名の日本語化（最低限）
    col_rename = {
        "races": "レース数",
        "tickets": "券数",
        "hit_rate_race": "的中率(レース)",
        "hit_rate_ticket": "的中率(券)",
        "stake": "投資(円)",
        "return": "払戻(円)",
        "roi": "回収率",
        "venue": "競馬場",
        "strategy": "買い目",
    }
    out = out.rename(columns={k: v for k, v in col_rename.items() if k in out.columns})

    # 2) strategy を日本語化
    if "買い目" in out.columns:
        # "(sanrentan_box4,)" みたいなのが来る場合があるので掃除
        def _clean_strategy(s):
            t = str(s).strip()

            # "(sanrenpuku_box4,)" 形式も "'sanrenpuku_box4'" 形式も両対応
            t = t.strip()                 # whitespace
            t = t.strip("()")             # remove outer parentheses
            t = t.replace(",", " ").strip()
            t = t.replace("'", "").replace('"', "")  # ★ クォート除去
            t = t.split()[0] if t else t             # 余計なゴミが残ったら先頭だけ
            return t
        out["買い目"] = out["買い目"].map(lambda x: _strategy_label_ja(_clean_strategy(x)))

    # 3) 回収率を赤く（HTML）
    if "回収率" in out.columns:
        out["回収率"] = out["回収率"].map(_fmt_roi_red)

    # 4) 金額列は見やすく（整数カンマ）
    for c in ["投資(円)", "払戻(円)"]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0).astype(int).map(lambda v: f"{v:,}")

    # 5) 並び替え（要望通り）
    if mode == "total":
        # 1つ目：買い目 → 競馬場 → …
        pref = ["買い目", "競馬場", "レース数", "券数", "的中率(レース)", "的中率(券)", "投資(円)", "払戻(円)", "回収率"]
    elif mode == "venue":
        # 2つ目：競馬場 → 買い目 → …
        pref = ["競馬場", "買い目", "レース数", "券数", "的中率(レース)", "的中率(券)", "投資(円)", "払戻(円)", "回収率"]
    else:  # "cum"
        pref = ["買い目", "レース数", "券数", "的中率(レース)", "的中率(券)", "投資(円)", "払戻(円)", "回収率"]

    cols = [c for c in pref if c in out.columns] + [c for c in out.columns if c not in pref]
    out = out[cols]

    return out




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

def _attach_display_if_exist(df: pd.DataFrame, feats: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    key_cols = [c for c in ["race_id", "umaban"] if c in out.columns and c in feats.columns]
    if not key_cols:
        return out

    add_cols = []
    for c in ["horse_name", "jockey_name", "sex_age_disp", "wakuban", "impost"]:
        if c in feats.columns:
            add_cols.append(c)

    if not add_cols:
        return out

    tmp = feats[key_cols + add_cols].drop_duplicates()
    return out.merge(tmp, on=key_cols, how="left")

_attach_names_if_exist = _attach_display_if_exist

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


def _load_results_maps(date: str) -> tuple[dict[tuple[str, int], int], dict[tuple[str, int], float]]:
    """
    results_flat.csv から
      rank_map[(race_id, umaban)] = rank
      agari_map[(race_id, umaban)] = agari
    を作って返す
    """
    if not RESULTS_FLAT.exists() or RESULTS_FLAT.stat().st_size == 0:
        return {}, {}

    try:
        df = pd.read_csv(RESULTS_FLAT, dtype={"race_id": str})
    except pd.errors.EmptyDataError:
        return {}, {}

    need = {"date", "race_id", "umaban", "rank"}
    if not need.issubset(df.columns):
        return {}, {}

    df = df[df["date"].astype(str) == str(date)].copy()
    if df.empty:
        return {}, {}

    df["race_id"] = df["race_id"].astype(str)
    df["umaban"] = pd.to_numeric(df["umaban"], errors="coerce")
    df["rank"] = pd.to_numeric(df["rank"], errors="coerce")
    if "agari" in df.columns:
        df["agari"] = pd.to_numeric(df["agari"], errors="coerce")
    else:
        df["agari"] = pd.NA

    df = df.dropna(subset=["race_id", "umaban", "rank"]).copy()
    df["umaban"] = df["umaban"].astype(int)
    df["rank"] = df["rank"].astype(int)

    rank_map: dict[tuple[str, int], int] = {}
    agari_map: dict[tuple[str, int], float] = {}

    for r in df.itertuples(index=False):
        key = (str(r.race_id), int(r.umaban))
        rank_map[key] = int(r.rank)
        try:
            if pd.notna(r.agari):
                agari_map[key] = float(r.agari)
        except Exception:
            pass

    return rank_map, agari_map



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
    agari_map: dict[tuple[str, int], float] | None = None,
    extra_row_class_map: dict[int, str] | None = None,
) -> str:
    agari_map = agari_map or {}
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

    # results_flat の agari を「上り」に差し込む（列名は "上り" を想定）
    if "上り" in df.columns:
        new_agari = []
        for _, row in df.iterrows():
            try:
                um = int(str(row.get("馬番")).split()[0])
            except Exception:
                um = None
            a = agari_map.get((str(rid), um)) if um is not None else None
            new_agari.append(a if a is not None else row.get("上り"))
        df["上り"] = new_agari

        # ここで badge 化（表に出ている馬の中で順位付け）
        df = add_agari_badge(df, col="上り")


    cols = [c for c in df.columns if c != "__wakuban"]
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

            # 枠番（帽子色）は __wakuban から読む
            waku_cls = ""
            w = row.get("__wakuban", None)
            try:
                if w is not None and pd.notna(w):
                    w_i = int(float(w))
                    if 1 <= w_i <= 8:
                        waku_cls = f"w{w_i}"
            except Exception:
                pass

            badge = f'<span class="umabadge {waku_cls}">{um}</span>' if waku_cls else str(um)
            row_dict["馬番"] = f'{badge}{icon}{pred_tag}'

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
    results_map, agari_map = _load_results_maps(target_date)
    has_results = len(results_map) > 0
    conf_calib = _load_conf_calib()

    payouts_df_all = None
    if PAYBACK_FLAT.exists() and PAYBACK_FLAT.stat().st_size > 0:
        try:
            payouts_df_all = pd.read_csv(PAYBACK_FLAT, dtype={"race_id": str, "bet_type": str, "combo": str})
        except pd.errors.EmptyDataError:
            payouts_df_all = None


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
tr.hit-5 {{ background:#ededed; }}



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
  padding:2px 10px;          /* ← 少し横に余裕 */
  border-radius:999px;       /* ← 丸バッジ */
  font-size:12px;
  font-weight:800;
  margin-left:14px;          /* ← 8→14 にして離す */
  border:1px solid transparent;
  line-height:1.2;
  vertical-align:middle;
}}

.predtag.win{{
  background:#fff2a8;        /* 少し濃いめ */
  color:#4a3a00;
  border-color:#d9be3a;
  box-shadow:0 1px 0 rgba(0,0,0,0.06);
}}
.predtag.third{{
  background:#dff4ff;
  color:#003b52;
  border-color:#67c4e6;
  box-shadow:0 1px 0 rgba(0,0,0,0.06);
}}


/* 的中詳細（折りたたみ） */
.hitdetail {{
  display:none;
  margin-top:8px;
  padding:10px 12px;
  background:#fff7f7;
  border:1px solid #f2caca;
  border-radius:10px;
  font-size:13px;
}}
.hitdetail ul {{
  margin:6px 0 0 18px;
}}
.hitdetail li {{
  margin:4px 0;
}}
.hitbadge {{
  cursor:pointer;
  user-select:none;
}}
.racecond{{
  display:inline-block;
  margin-left:10px;
  padding:3px 10px;
  border-radius:999px;
  background:#f1f3f5;
  border:1px solid #e5e5e5;
  color:#333;
  font-weight:800;
  font-size:12px;
  vertical-align:middle;
}}
.umabadge{{
  display:inline-block;
  min-width: 28px;
  padding:2px 8px;
  border-radius:999px;
  font-weight:900;
  text-align:center;
  border:1px solid rgba(0,0,0,0.15);
}}
.w1{{ background:#ffffff; color:#111; }}
.w2{{ background:#111111; color:#fff; }}
.w3{{ background:#d60000; color:#fff; }}
.w4{{ background:#0057d8; color:#fff; }}
.w5{{ background:#ffd200; color:#111; }}
.w6{{ background:#18a000; color:#fff; }}
.w7{{ background:#ff8a00; color:#111; }}
.w8{{ background:#ff4fc3; color:#111; }}  /* ピンク */

.agari{{ font-weight:900; padding:1px 6px; border-radius:8px; }}
.agari.g{{ background:#ffe08a; }}
.agari.s{{
  background:#0b5ed7;  /* 濃い青 */
  color:#fff;
}}

.agari.b{{ background:#ffd1b0; }}

/* 的中したレースのタイトルを赤く */
.race-title.hit {{
  color: #d60000;
}}

/* 倍率バッジ（+xx.x倍）を色分け */
.mult {{
  display: inline-block;
  padding: 1px 8px;
  border-radius: 999px;
  font-weight: 900;
  font-size: 12px;
  border: 1px solid rgba(0,0,0,0.10);
  margin-left: 4px;
  vertical-align: middle;
}}
.mult.m0 {{ background: #f1f3f5; color:#444; }}  /* 〜+2倍程度 */
.mult.m1 {{ background: #e7f0ff; color:#1f4dd8; }}
.mult.m2 {{ background: #e8fff2; color:#0a7a3a; }}
.mult.m3 {{ background: #fff1db; color:#a85a00; }}
.mult.m4 {{ background: #ffe0e0; color:#c40000; }} /* 爆高 */


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
                 <script>
function toggleHitDetail(id){
  const el = document.getElementById(id);
  if(!el) return;
  el.style.display = (el.style.display === "none" || el.style.display === "")
    ? "block" : "none";
}
</script>
""")

    # 目次
    parts.append('<div class="toc"><div style="margin-bottom:8px; font-weight:600;">レース一覧</div>')
    for rid in race_ids_all:
        label = _toc_label_from_race_id(rid)
        parts.append(f'<a href="#r{rid}">{label}</a>')
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

            pred_win = _attach_display_if_exist(pred_win, feats)


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

            # まず rename_map を先に用意
            rename_map = {
                "umaban": "馬番",
                "horse_name": "馬名",
                "sex_age_disp": "性齢",
                "jockey_name": "騎手",
                "impost": "斤量",
                "tansyo_odds": "単勝",
                "popularity": "人気",
                "agari": "上り",
                "pred": "単勝Pred",
            }

            # 色付け用の隠し列（wakubanを残す）
            dfw["__wakuban"] = dfw.get("wakuban")

            cols_w = [
                "umaban",
                "__wakuban",   # ← これを残す
                "horse_name",
                "sex_age_disp",
                "jockey_name",
                "impost",
                "tansyo_odds",
                "popularity",
                "agari",
                "pred",
                "EV(単勝)",
            ]
            cols_w = [c for c in cols_w if c in dfw.columns]

            dfw = (
                dfw[cols_w]
                .sort_values("pred", ascending=False)
                .head(topk_per_race)
                .reset_index(drop=True)
            )

            # 表示名に変える（__wakuban はそのまま残す）
            dfw = dfw.rename(columns=rename_map)



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
            pred_rank_df = _attach_display_if_exist(pred_rank_df, feats)

            is_chaos, chaos_h = _is_chaotic_race_by_triscore(pred_rank_df, topn=8, entropy_th=1.80)
            chaos_tag = _chaos_label(chaos_h)  # "小荒"/"中荒"/"大荒"/""
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

            # 色付け用の隠し列
            dfr["__wakuban"] = dfr.get("wakuban")

            cols_r = [
                "pred_rank",
                "umaban",
                "__wakuban",   # ← これを残す
                "horse_name",
                "sex_age_disp",
                "jockey_name",
                "impost",
                "tansyo_odds",
                "popularity",
                "agari",
                "score",
            ]
            cols_r = [c for c in cols_r if c in dfr.columns]

            dfr = (
                dfr[cols_r]
                .sort_values("pred_rank")
                .head(topk_per_race)
                .reset_index(drop=True)
            )

            rename_map_r = {
                "pred_rank": "予測順位",
                "umaban": "馬番",
                "horse_name": "馬名",
                "sex_age_disp": "性齢",
                "jockey_name": "騎手",
                "impost": "斤量",
                "tansyo_odds": "単勝",
                "popularity": "人気",
                "agari": "上り",
                "score": "Score",
            }

            dfr = dfr.rename(columns=rename_map_r)


            # 表示フォーマット
            if "Score" in dfr.columns:
                dfr["Score"] = dfr["Score"].map(lambda x: f"{float(x):.3f}" if pd.notna(x) else "")
            if "単勝" in dfr.columns:
                dfr["単勝"] = dfr["単勝"].map(_format_odds_html)
            if "人気" in dfr.columns:
                dfr["人気"] = dfr["人気"].map(_format_pop_html)
            if "斤量" in dfr.columns:
                dfr["斤量"] = dfr["斤量"].map(lambda x: "" if pd.isna(x) else f"{float(x):.0f}")


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

            race_div_class = "race"
            # badge_text = f'自信度 {conf}/5' + (f'（hit@K {exp_hit}）' if exp_hit else '')
            hit_txt = f"Hit{exp_hit}" if exp_hit else ""
            chaos_txt = chaos_tag  # 空なら何も出さない

            tail = "｜".join([x for x in [hit_txt, chaos_txt] if x])
            badge = f'<span class="racecond">自信{_stars(conf)}{"｜"+tail if tail else ""}</span>'


            hit_badge = ""
            if has_results and payouts_df_all is not None:
                hit_badge = _hit_badge_html_for_race(
                    target_date=str(target_date),
                    rid=str(rid),
                    pred_order=pred_order,
                    odds_map=odds_map,
                    results_map=results_map,
                    payouts_df=payouts_df_all,
                    stake_per_ticket=stake_per_ticket,
                )

            is_hit = (hit_badge != "")


            cond = _race_condition_text(feats)
            cond_html = f' <span class="racecond">{cond}</span>' if cond else ""
            # --- 見出し用レース情報 ---
            race_name = ""
            if "race_name" in feats.columns:
                v = feats.iloc[0]["race_name"]
                if pd.notna(v):
                    race_name = str(v)

            # 開催（中山など）
            # 開催（race_id 由来: netkeiba PP）
            place = _place_from_race_id_netkeiba(rid)

            # 芝/ダ + 距離 + 馬場
            cond = _race_condition_text(feats)   # 例: "中山 芝1600m 良"

            # cond から「芝1600m 良」だけ抜く
            cond_tail = cond
            if place and cond.startswith(place):
                cond_tail = cond[len(place):].strip()

            # R番号（race_id の末尾2桁）
            try:
                rnum = int(str(rid)[-2:])
                rnum = f"{rnum}R"
            except Exception:
                rnum = ""

            title_main = " ".join([x for x in [place, rnum, race_name] if x])
            title_sub  = cond_tail
            title_cls = "race-title hit" if is_hit else "race-title"
            parts.append(
                f'<div class="{race_div_class}" id="r{rid}">'
                f'<h2><span class="{title_cls}">{title_main}</span> <span class="racecond">{title_sub}</span>{badge}{hit_badge}</h2>'
            )

            parts.append(f'<div class="note">荒れ指数(Entropy, top8 score): {chaos_h:.3f}</div>')


            parts.append('<div class="grid">')

            parts.append('<div>')
            parts.append('<h3>単勝モデル（分類）</h3>')
            parts.append(
                _render_table_with_results(dfw, rid, results_map, agari_map)
                if has_results
                else dfw.to_html(index=False, escape=False)
            )


            parts.append('<div class="note">EV(単勝)=単勝Pred×単勝オッズ</div>')
            parts.append('</div>')

            parts.append('<div>')
            parts.append('<h3>3連系モデル（ランキング）</h3>')
            parts.append(_render_table_with_results(dfr, rid, results_map, agari_map, extra_row_class_map=row_class_map)
             if has_results else _render_table_with_results(dfr, rid, {}, {}, extra_row_class_map=row_class_map))

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
            pr_total_disp = _format_summary_table(
                format_summary_df_for_html(pr_total),
                mode="total"
            )

            pr_venue_disp = _format_summary_table(
                format_summary_df_for_html(pr_venue),
                mode="venue"
            )


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
            fn_total_disp = _format_summary_table(
                format_summary_df_for_html(fn_total),
                mode="total"
            )

            fn_venue_disp = _format_summary_table(
                format_summary_df_for_html(fn_venue),
                mode="venue"
            )

            final_html = (
                _summary_block(f"最終（当日全 {len(race_ids_all)} レース）当日サマリ（戦略別）", fn_total_disp)
                + _summary_block("最終：開催場所別サマリ（戦略別）", fn_venue_disp)
            )

        # 累計（常に表示）
        cum = load_cumulative_summary()
        cum_disp = _format_summary_table(format_summary_df_for_html(cum), mode="cum") if cum is not None else None
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
