# v2_0_0/src/recommend_strategy.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math
import pickle

import numpy as np
import pandas as pd

from strategies import build_strategy_catalog
from eval_bets import (
    actual_outcomes_from_results_map,
    payout_map_for_race,
    eval_one_race_tickets,
)

DATA_DIR = Path("..", "data")
MODEL_DIR = DATA_DIR / "03_train"
RESULTS_DIR = DATA_DIR / "results_cache"
RESULTS_FLAT = RESULTS_DIR / "results_flat.csv"
PAYBACK_FLAT = RESULTS_DIR / "payouts_flat.csv"

OUT_MODEL = MODEL_DIR / "strategy_selector.pkl"
OUT_FEATURES = MODEL_DIR / "strategy_train_features.tsv"


# -------------------------
# tri_df から「形状特徴」を作る
# -------------------------
def _safe_float(x, default=0.0) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)

def _softmax(x: np.ndarray) -> np.ndarray:
    x = x.astype(float)
    x = x - np.max(x) if len(x) else x
    ex = np.exp(x)
    s = ex.sum()
    if s <= 0:
        return np.zeros_like(ex)
    return ex / s

def entropy_from_top_scores(scores: list[float], temperature: float = 1.0) -> float:
    s = np.array([_safe_float(v) for v in scores], dtype=float)
    if s.size == 0:
        return 0.0
    s = (s - s.max()) / max(temperature, 1e-9)
    p = _softmax(s)
    return float(-(p * np.log(p + 1e-12)).sum())

def tri_marginals(tri_df: pd.DataFrame) -> dict:
    """
    tri_df: columns ["1着","2着","3着","prob"] を想定
    周辺確率 p(1着=umaban) / p(3着=umaban) と集中度指標を返す
    """
    if tri_df is None or len(tri_df) == 0:
        return {
            "p1_map": {}, "p3_map": {},
            "mass": 0.0,
            "p1_max": 0.0, "p1_gap": 0.0,
            "p3_max": 0.0, "p3_gap": 0.0,
            "top1_prob": 0.0, "top5_mass": 0.0,
            "effn": 0.0,
        }

    x = tri_df.copy()
    x["prob"] = pd.to_numeric(x["prob"], errors="coerce").fillna(0.0)
    mass = float(x["prob"].sum())
    if mass <= 0:
        return {
            "p1_map": {}, "p3_map": {},
            "mass": 0.0,
            "p1_max": 0.0, "p1_gap": 0.0,
            "p3_max": 0.0, "p3_gap": 0.0,
            "top1_prob": 0.0, "top5_mass": 0.0,
            "effn": 0.0,
        }

    x["p"] = x["prob"] / mass

    # 1着/3着の周辺
    p1 = x.groupby("1着")["p"].sum().sort_values(ascending=False)
    p3 = x.groupby("3着")["p"].sum().sort_values(ascending=False)

    def _top2(series: pd.Series):
        if len(series) == 0:
            return (None, 0.0, None, 0.0)
        if len(series) == 1:
            return (series.index[0], float(series.iloc[0]), None, 0.0)
        return (series.index[0], float(series.iloc[0]), series.index[1], float(series.iloc[1]))

    a1, av1, a2, av2 = _top2(p1)
    b1, bv1, b2, bv2 = _top2(p3)

    # topK内の確率集中度
    p_sorted = x["p"].sort_values(ascending=False)
    top1_prob = float(p_sorted.head(1).sum())
    top5_mass = float(p_sorted.head(5).sum())

    # 有効候補数（集中してるほど小さい）
    # effn = 1 / sum(p_i^2)
    effn = float(1.0 / max(float((x["p"] ** 2).sum()), 1e-12))

    return {
        "p1_map": p1.to_dict(),
        "p3_map": p3.to_dict(),
        "mass": mass,
        "p1_max": av1,
        "p1_gap": (av1 - av2),
        "p3_max": bv1,
        "p3_gap": (bv1 - bv2),
        "top1_prob": top1_prob,
        "top5_mass": top5_mass,
        "effn": effn,
    }

def extract_race_features_from_predictions(pred_rank_df_race, tri_df_race, score_topn_for_entropy=8):
    feats = {}

    # --- 馬単位 score ---
    scores = pd.to_numeric(pred_rank_df_race.get("score", pd.Series([])), errors="coerce").dropna().values
    if len(scores):
        feats["score_max"] = float(np.max(scores))
        feats["score_std"] = float(np.std(scores))
        topn = sorted(scores, reverse=True)[:score_topn_for_entropy]
        feats["entropy"] = float(entropy_from_top_scores(topn, temperature=1.0))  # ★train_strategy_selectorで使う
    else:
        feats["score_max"] = 0.0
        feats["score_std"] = 0.0
        feats["entropy"] = 0.0

    # --- 3連系が無ければダミー ---
    required_cols = {"1着", "2着", "3着", "prob"}
    if tri_df_race is None or tri_df_race.empty or not required_cols.issubset(tri_df_race.columns):
        feats["tri_mass"] = 0.0
        feats["conf_score"] = 0.0
        feats["p1_max"] = 0.0
        feats["p1_gap"] = 0.0
        feats["p3_max"] = 0.0
        feats["p3_gap"] = 0.0
        feats["top1_prob"] = 0.0
        feats["top5_mass"] = 0.0
        feats["effn"] = 0.0
        return feats

    marg = tri_marginals(tri_df_race)

    # ★dict は入れない（p1_map/p3_mapは捨てる）
    feats["tri_mass"]  = float(marg.get("mass", 0.0))
    feats["p1_max"]    = float(marg.get("p1_max", 0.0))
    feats["p1_gap"]    = float(marg.get("p1_gap", 0.0))
    feats["p3_max"]    = float(marg.get("p3_max", 0.0))
    feats["p3_gap"]    = float(marg.get("p3_gap", 0.0))
    feats["top1_prob"] = float(marg.get("top1_prob", 0.0))
    feats["top5_mass"] = float(marg.get("top5_mass", 0.0))
    feats["effn"]      = float(marg.get("effn", 0.0))

    # conf_score: 「固いほど高い」みたいな雑指標（好きに調整OK）
    # effnが小さいほど固い → conf_score を 1/effn で
    feats["conf_score"] = float(1.0 / max(feats["effn"], 1e-9))

    return feats



# -------------------------
# 結果/払戻のロード
# -------------------------
def load_results_map(date: str) -> dict[tuple[str, int], int]:
    """
    results_flat.csv -> rank_map[(race_id, umaban)] = rank
    """
    if not RESULTS_FLAT.exists() or RESULTS_FLAT.stat().st_size == 0:
        return {}
    df = pd.read_csv(RESULTS_FLAT, dtype={"race_id": str})
    need = {"date", "race_id", "umaban", "rank"}
    if not need.issubset(df.columns):
        return {}
    df = df[df["date"].astype(str) == str(date)].copy()
    if df.empty:
        return {}
    df["race_id"] = (
        df["race_id"].astype(str)
        .str.replace(r"\.0$", "", regex=True)
        .str.zfill(12)
    )
    df["umaban"] = pd.to_numeric(df["umaban"], errors="coerce")
    df["rank"] = pd.to_numeric(df["rank"], errors="coerce")
    df = df.dropna(subset=["race_id", "umaban", "rank"]).copy()
    df["umaban"] = df["umaban"].astype(int)
    df["rank"] = df["rank"].astype(int)
    return {(r.race_id, int(r.umaban)): int(r.rank) for r in df.itertuples(index=False)}

def load_payouts_df(date: str) -> pd.DataFrame:
    if not PAYBACK_FLAT.exists() or PAYBACK_FLAT.stat().st_size == 0:
        return pd.DataFrame()
    df = pd.read_csv(PAYBACK_FLAT, dtype={"race_id": str, "bet_type": str, "combo": str})
    need = {"date", "race_id", "bet_type", "combo", "pay100"}
    if not need.issubset(df.columns):
        return pd.DataFrame()
    df = df[df["date"].astype(str) == str(date)].copy()
    df["race_id"] = (
        df["race_id"].astype(str)
          .str.replace(r"\.0$", "", regex=True)
          .str.zfill(12)
    )
    return df


# -------------------------
# 戦略の「正解」作成（教師ラベル）
# -------------------------
@dataclass
class LabelConfig:
    stake_per_ticket: int = 100
    alpha_ticket_penalty: float = 0.15   # 券数が多い戦略に不利を与える
    require_hit: bool = False            # Trueなら「当たった戦略の中で選ぶ」

def evaluate_strategies_for_one_race(
    rid: str,
    pred_order: list[int],
    odds_map: dict[int, float],
    results_map: dict[tuple[str, int], int],
    payouts_df: pd.DataFrame,
    stake_per_ticket: int,
) -> list[dict]:
    """
    build_strategy_catalog() の全戦略を1レースで評価し、行リストを返す
    """
    outcomes = actual_outcomes_from_results_map(results_map, rid)
    if not outcomes:
        return []

    pay_map = payout_map_for_race(payouts_df, str(payouts_df["date"].iloc[0]) if "date" in payouts_df.columns and len(payouts_df) else "", rid)
    # payout_map_for_race は (payouts_df, date, rid) なので date が必要
    # ただこの関数だけで使うなら date を外から渡す方が良いので、下でラッパーで渡す

    raise RuntimeError("This function expects wrapper that provides pay_map with date. Use evaluate_strategies_for_one_race_with_paymap().")

def evaluate_strategies_for_one_race_with_paymap(
    rid: str,
    pred_order: list[int],
    odds_map: dict[int, float],
    outcomes: dict,
    pay_map: dict,
    stake_per_ticket: int,
) -> list[dict]:
    catalog = build_strategy_catalog()
    rows = []
    for strat_name, gen in catalog:
        tickets = gen(pred_order, odds_map)
        ev = eval_one_race_tickets(
            rid=rid,
            tickets=tickets,
            outcomes=outcomes,
            pay_map=pay_map,
            stake_per_ticket=stake_per_ticket,
        )
        stake = float(ev.get("stake", 0.0))
        ret = float(ev.get("return", 0.0))
        roi = float(ev.get("roi", (ret / stake * 100.0) if stake > 0 else 0.0))
        rows.append({
            "strategy": str(strat_name),
            "stake": stake,
            "return": ret,
            "roi": roi,
            "tickets": int(ev.get("tickets", 0)) if "tickets" in ev else int(stake / max(stake_per_ticket, 1)),
            "hit": 1 if ret > 0 else 0,
        })
    return rows

def choose_best_strategy(rows: list[dict], cfg: LabelConfig) -> str | None:
    """
    教師ラベルを作る：
      utility = (roi/100) - alpha * log1p(tickets)
    require_hit=Trueなら hit==1 の中で選ぶ
    """
    if not rows:
        return None

    cand = rows
    if cfg.require_hit:
        cand = [r for r in rows if int(r.get("hit", 0)) == 1]
        if not cand:
            return None

    def utility(r):
        roi = float(r.get("roi", 0.0)) / 100.0
        t = float(r.get("tickets", 0))
        return roi - cfg.alpha_ticket_penalty * math.log1p(t)

    cand = sorted(cand, key=utility, reverse=True)
    return str(cand[0]["strategy"])


# -------------------------
# 学習データ生成
# -------------------------
def build_training_table(
    date_list: list[str],
    pred_rank_df_all: pd.DataFrame,
    tri_df_all: pd.DataFrame,
    pred_order_by_race: dict[str, list[int]],
    odds_map_by_race: dict[str, dict[int, float]],
    label_cfg: LabelConfig = LabelConfig(),
) -> pd.DataFrame:
    """
    date_list に含まれる日付のレースについて、
    特徴量 + best_strategy を作って返す
    """
    out_rows = []

    for date in date_list:
        results_map = load_results_map(date)
        payouts_df = load_payouts_df(date)
        if payouts_df is None or len(payouts_df) == 0:
            continue

        # payouts_df は load_payouts_df(date) で既に date で絞られてる前提
        rids_from_payouts = set(payouts_df["race_id"].astype(str).unique())

        # results_map は load_results_map(date) で既に date で絞られてる前提
        rids_from_results = set([rid for (rid, _) in results_map.keys()])

        # 予測が作れてる race_id（features → pred_order_by_race）
        rids_from_pred = set(map(str, pred_order_by_race.keys()))

        # ✅ その日に「払戻あり」「結果あり」「予測あり」の race_id だけ
        rids = sorted(rids_from_payouts & rids_from_results & rids_from_pred)

        print("[DEBUG]", date,
            "payouts", len(rids_from_payouts),
            "results", len(rids_from_results),
            "pred", len(rids_from_pred),
            "USE", len(rids))


        for rid in rids:
            rid = str(rid)

            # outcomes/pay_map を作れるレースだけ
            outcomes = actual_outcomes_from_results_map(results_map, rid)
            if not outcomes:
                print("[SKIP outcomes]", date, rid)
                continue
            pay_map = payout_map_for_race(payouts_df, date, rid)
            if not pay_map:
                print("[SKIP pay_map]", date, rid)
                continue

            # 予測がある前提
            pred_order = pred_order_by_race.get(rid, [])
            odds_map = odds_map_by_race.get(rid, {})

            g_rank = pred_rank_df_all[pred_rank_df_all["race_id"].astype(str) == rid].copy()
            g_tri = tri_df_all[tri_df_all["race_id"].astype(str) == rid].copy() if "race_id" in tri_df_all.columns else pd.DataFrame()

            feats = extract_race_features_from_predictions(g_rank, g_tri)

            # 全戦略を1レース評価 → best_strategy
            rows = evaluate_strategies_for_one_race_with_paymap(
                rid=rid,
                pred_order=pred_order,
                odds_map=odds_map,
                outcomes=outcomes,
                pay_map=pay_map,
                stake_per_ticket=label_cfg.stake_per_ticket,
            )
            best = choose_best_strategy(rows, label_cfg)
            if best is None:
                continue

            out_rows.append({
                "date": date,
                "race_id": rid,
                **feats,
                "best_strategy": best,
            })

    return pd.DataFrame(out_rows)


# -------------------------
# 学習（軽量版：sklearn）
# -------------------------
def train_strategy_selector(df: pd.DataFrame):
    """
    いったん軽量で堅い：RandomForest 相当でも良いけど、
    ここでは sklearn の GradientBoosting で十分。
    ※ LightGBM でももちろんOK（好みで差し替え）
    """
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder
    from sklearn.metrics import accuracy_score
    from sklearn.ensemble import RandomForestClassifier

    feature_cols = [
        "entropy", "conf_score",
        "p1_max", "p1_gap", "p3_max", "p3_gap",
        "top1_prob", "top5_mass", "effn", "tri_mass",
    ]

    df2 = df.dropna(subset=["best_strategy"]).copy()
    for c in feature_cols:
        df2[c] = pd.to_numeric(df2[c], errors="coerce").fillna(0.0)

    X = df2[feature_cols].values
    le = LabelEncoder()
    y = le.fit_transform(df2["best_strategy"].astype(str).values)

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y if len(set(y)) > 1 else None)

    clf = RandomForestClassifier(
        n_estimators=500,
        random_state=42,
        class_weight="balanced_subsample",
        min_samples_leaf=10,
    )
    clf.fit(Xtr, ytr)

    pred = clf.predict(Xte) if len(Xte) else clf.predict(Xtr)
    acc = accuracy_score(yte, pred) if len(Xte) else 1.0

    payload = {
        "feature_cols": feature_cols,
        "label_classes": le.classes_.tolist(),
        "model": clf,
        "label_encoder": le,
        "meta": {"valid_acc": float(acc), "n_train": int(len(Xtr)), "n_total": int(len(df2))},
    }

    OUT_MODEL.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_MODEL, "wb") as f:
        pickle.dump(payload, f)

    return payload


def load_strategy_selector():
    if not OUT_MODEL.exists():
        return None
    with open(OUT_MODEL, "rb") as f:
        return pickle.load(f)

def predict_best_strategy_for_race(pred_rank_df_race: pd.DataFrame, tri_df_race: pd.DataFrame) -> tuple[str, dict]:
    payload = load_strategy_selector()
    if payload is None:
        return ("", {"reason": "model_not_found"})

    feature_cols = payload["feature_cols"]
    le = payload["label_encoder"]
    model = payload["model"]

    feats = extract_race_features_from_predictions(pred_rank_df_race, tri_df_race)
    x = np.array([[float(feats.get(c, 0.0)) for c in feature_cols]], dtype=float)

    y = model.predict(x)[0]
    name = str(le.inverse_transform([y])[0])

    pred_proba = 0.0
    top3 = None
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(x)[0]
        pred_proba = float(probs[y])  # ← 予測した戦略の確信度
        order = np.argsort(-probs)[:3]
        top3 = [(str(le.inverse_transform([i])[0]), float(probs[i])) for i in order]

    dbg = {"features": feats, "top3": top3, "pred_proba": pred_proba, "meta": payload.get("meta", {})}
    return (name, dbg)
