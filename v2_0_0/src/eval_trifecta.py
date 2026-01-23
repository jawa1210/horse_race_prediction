# eval_trifecta.py
from __future__ import annotations

from itertools import combinations, permutations
import pandas as pd
# eval_importance.py
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def _ensure_cols(df: pd.DataFrame, cols: list[str], name: str):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{name} is missing columns: {missing}")


def build_results_df_from_test(test_df: pd.DataFrame) -> pd.DataFrame:
    """
    test_df から正解着順（rank）を作る
    必要列: race_id, umaban, rank
    """
    _ensure_cols(test_df, ["race_id", "umaban", "rank"], "test_df")
    out = test_df[["race_id", "umaban", "rank"]].copy()
    out["race_id"] = out["race_id"].astype(str)
    out["rank"] = pd.to_numeric(out["rank"], errors="coerce")
    out = out.dropna(subset=["rank"])
    return out


def build_pred_rank_df(evaluation_df_tri: pd.DataFrame) -> pd.DataFrame:
    """
    evaluation_df_tri から予測順位DFを作る
    必要列: race_id, umaban, pred_rank
    （scoreがあれば残してOK）
    """
    _ensure_cols(evaluation_df_tri, ["race_id", "umaban"], "evaluation_df_tri")

    if "pred_rank" not in evaluation_df_tri.columns:
        # もし score はあるけど pred_rank が無い場合は作る
        if "score" not in evaluation_df_tri.columns:
            raise ValueError("evaluation_df_tri needs 'pred_rank' or 'score'")
        df = evaluation_df_tri.copy()
        df["race_id"] = df["race_id"].astype(str)
        df["pred_rank"] = (
            df.groupby("race_id")["score"]
            .rank(ascending=False, method="first")
            .astype(int)
        )
        return df[["race_id", "umaban", "pred_rank", "score"]].copy()

    df = evaluation_df_tri.copy()
    df["race_id"] = df["race_id"].astype(str)
    return df[[c for c in ["race_id", "umaban", "pred_rank", "score"] if c in df.columns]].copy()


def _get_true_top3(results_df: pd.DataFrame, race_id: str):
    g = results_df[results_df["race_id"] == race_id].sort_values("rank")
    if len(g) < 3:
        raise ValueError("not enough finishers")
    a = int(g.iloc[0]["umaban"])
    b = int(g.iloc[1]["umaban"])
    c = int(g.iloc[2]["umaban"])
    return a, b, c


def eval_box_hit_rate(pred_rank_df: pd.DataFrame, results_df: pd.DataFrame, N: int, bet_type: str) -> float:
    """
    上位N頭BOXの的中率（レース単位）
    bet_type: umaren / umatan / wide / sanrenpuku
    """
    _ensure_cols(pred_rank_df, ["race_id", "umaban", "pred_rank"], "pred_rank_df")
    _ensure_cols(results_df, ["race_id", "umaban", "rank"], "results_df")

    hit = 0
    total = 0

    for rid, g in pred_rank_df.groupby("race_id"):
        try:
            a, b, c = _get_true_top3(results_df, rid)
        except Exception:
            continue

        topN = g.sort_values("pred_rank").head(N)["umaban"].astype(int).tolist()

        if bet_type == "umaren":
            # 1-2着(順不同)
            target = {a, b}
            hit_flag = target in [set(x) for x in combinations(topN, 2)]

        elif bet_type == "umatan":
            # 1-2着(順序)
            hit_flag = (a, b) in list(permutations(topN, 2))

        elif bet_type == "wide":
            # 1-2 / 1-3 / 2-3 のどれか当たればOK
            pairs = [set(x) for x in combinations(topN, 2)]
            hit_flag = ({a, b} in pairs) or ({a, c} in pairs) or ({b, c} in pairs)

        elif bet_type == "sanrenpuku":
            # 1-2-3(順不同)
            target = {a, b, c}
            hit_flag = target in [set(x) for x in combinations(topN, 3)]

        else:
            raise ValueError(f"unknown bet_type: {bet_type}")

        hit += int(hit_flag)
        total += 1

    return hit / total if total else 0.0


def eval_nagashi_hit_rate(pred_rank_df: pd.DataFrame, results_df: pd.DataFrame, N: int, bet_type: str) -> float:
    """
    上位N頭のうち「1位固定」流しの的中率（レース単位）
    bet_type: umaren / umatan / sanrenpuku / sanrentan
    """
    _ensure_cols(pred_rank_df, ["race_id", "umaban", "pred_rank"], "pred_rank_df")
    _ensure_cols(results_df, ["race_id", "umaban", "rank"], "results_df")

    hit = 0
    total = 0

    for rid, g in pred_rank_df.groupby("race_id"):
        try:
            a, b, c = _get_true_top3(results_df, rid)
        except Exception:
            continue

        topN = g.sort_values("pred_rank").head(N)["umaban"].astype(int).tolist()
        if len(topN) < 2:
            continue

        key = topN[0]
        others = topN[1:]

        if bet_type == "umaren":
            # key - others（順不同）
            bets = [set([key, x]) for x in others]
            hit_flag = set([a, b]) in bets

        elif bet_type == "umatan":
            # key -> others（順序）
            bets = [(key, x) for x in others]
            hit_flag = (a, b) in bets

        elif bet_type == "sanrenpuku":
            # key + othersから2頭（順不同）
            bets = [set([key, x, y]) for x, y in combinations(others, 2)]
            hit_flag = set([a, b, c]) in bets

        elif bet_type == "sanrentan":
            # key -> (x,y)（順序）
            bets = [(key, x, y) for x, y in permutations(others, 2)]
            hit_flag = (a, b, c) in bets

        else:
            raise ValueError(f"unknown bet_type: {bet_type}")

        hit += int(hit_flag)
        total += 1

    return hit / total if total else 0.0


def evaluate_strategies(pred_rank_df: pd.DataFrame, results_df: pd.DataFrame) -> pd.DataFrame:
    """
    あなたが欲しい戦略（上位4/5、BOXと1頭流し）を全部まとめて返す
    """
    rows = []

    for N in [4, 5]:
        # BOX
        for bt in ["umaren", "umatan", "wide", "sanrenpuku"]:
            rows.append({
                "topN": N,
                "style": "BOX",
                "bet_type": bt,
                "hit_rate": eval_box_hit_rate(pred_rank_df, results_df, N, bt),
            })

        # 1頭流し
        for bt in ["umaren", "umatan", "sanrenpuku", "sanrentan"]:
            rows.append({
                "topN": N,
                "style": "NAGASHI(1st)",
                "bet_type": bt,
                "hit_rate": eval_nagashi_hit_rate(pred_rank_df, results_df, N, bt),
            })

    return pd.DataFrame(rows).sort_values(["topN", "style", "bet_type"]).reset_index(drop=True)

def _ensure_cols(df: pd.DataFrame, cols: list[str], name: str):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{name} is missing columns: {missing}")

def _get_true_top3(results_df: pd.DataFrame, race_id: str):
    g = results_df[results_df["race_id"] == race_id].sort_values("rank")
    if len(g) < 3:
        raise ValueError("not enough finishers")
    a = int(g.iloc[0]["umaban"])
    b = int(g.iloc[1]["umaban"])
    c = int(g.iloc[2]["umaban"])
    return a, b, c

def _get_pop_map(results_df: pd.DataFrame, race_id: str) -> dict[int, float]:
    """
    raceごとの umaban -> popularity を辞書化
    popularity が無い場合は空（平均人気はNaNになる）
    """
    if "popularity" not in results_df.columns:
        return {}
    g = results_df[results_df["race_id"] == race_id][["umaban", "popularity"]].copy()
    g["umaban"] = g["umaban"].astype(int)
    g["popularity"] = pd.to_numeric(g["popularity"], errors="coerce")
    return dict(zip(g["umaban"].tolist(), g["popularity"].tolist()))

def _mean_pop(pop_map: dict[int, float], umabans: list[int]) -> float:
    vals = [pop_map.get(u, np.nan) for u in umabans]
    vals = [v for v in vals if pd.notna(v)]
    return float(np.mean(vals)) if len(vals) else np.nan


def eval_box_stats(pred_rank_df: pd.DataFrame, results_df: pd.DataFrame, N: int, bet_type: str):
    """
    戻り値: (num_races, hits, hit_rate, avg_popularity_when_hit)
    avg_popularity_when_hit は「的中した時の的中馬(=正解側)の平均人気」を平均したもの
    """
    _ensure_cols(pred_rank_df, ["race_id", "umaban", "pred_rank"], "pred_rank_df")
    _ensure_cols(results_df, ["race_id", "umaban", "rank"], "results_df")

    hit = 0
    total = 0
    hit_pops = []

    for rid, g in pred_rank_df.groupby("race_id"):
        try:
            a, b, c = _get_true_top3(results_df, rid)
        except Exception:
            continue

        topN = g.sort_values("pred_rank").head(N)["umaban"].astype(int).tolist()
        pop_map = _get_pop_map(results_df, rid)

        hit_flag = False
        pop_val = np.nan

        if bet_type == "umaren":
            target = {a, b}
            hit_flag = target in [set(x) for x in combinations(topN, 2)]
            if hit_flag:
                pop_val = _mean_pop(pop_map, [a, b])

        elif bet_type == "umatan":
            hit_flag = (a, b) in list(permutations(topN, 2))
            if hit_flag:
                pop_val = _mean_pop(pop_map, [a, b])

        elif bet_type == "wide":
            pairs = [set(x) for x in combinations(topN, 2)]
            # どのペアで当たったかを決める（優先順は 1-2, 1-3, 2-3）
            if {a, b} in pairs:
                hit_flag = True
                pop_val = _mean_pop(pop_map, [a, b])
            elif {a, c} in pairs:
                hit_flag = True
                pop_val = _mean_pop(pop_map, [a, c])
            elif {b, c} in pairs:
                hit_flag = True
                pop_val = _mean_pop(pop_map, [b, c])

        elif bet_type == "sanrenpuku":
            target = {a, b, c}
            hit_flag = target in [set(x) for x in combinations(topN, 3)]
            if hit_flag:
                pop_val = _mean_pop(pop_map, [a, b, c])

        else:
            raise ValueError(f"unknown bet_type: {bet_type}")

        hit += int(hit_flag)
        total += 1
        if hit_flag:
            hit_pops.append(pop_val)

    hit_rate = hit / total if total else 0.0
    avg_pop = float(np.nanmean(hit_pops)) if len(hit_pops) else np.nan
    return total, hit, hit_rate, avg_pop


def eval_nagashi_stats(pred_rank_df: pd.DataFrame, results_df: pd.DataFrame, N: int, bet_type: str):
    """
    上位N頭のうち「1位固定」流しの統計
    戻り値: (num_races, hits, hit_rate, avg_popularity_when_hit)
    """
    _ensure_cols(pred_rank_df, ["race_id", "umaban", "pred_rank"], "pred_rank_df")
    _ensure_cols(results_df, ["race_id", "umaban", "rank"], "results_df")

    hit = 0
    total = 0
    hit_pops = []

    for rid, g in pred_rank_df.groupby("race_id"):
        try:
            a, b, c = _get_true_top3(results_df, rid)
        except Exception:
            continue

        topN = g.sort_values("pred_rank").head(N)["umaban"].astype(int).tolist()
        if len(topN) < 2:
            continue

        key = topN[0]
        others = topN[1:]
        pop_map = _get_pop_map(results_df, rid)

        hit_flag = False
        pop_val = np.nan

        if bet_type == "umaren":
            bets = [set([key, x]) for x in others]
            hit_flag = set([a, b]) in bets
            if hit_flag:
                pop_val = _mean_pop(pop_map, [a, b])

        elif bet_type == "umatan":
            bets = [(key, x) for x in others]
            hit_flag = (a, b) in bets
            if hit_flag:
                pop_val = _mean_pop(pop_map, [a, b])

        elif bet_type == "sanrenpuku":
            bets = [set([key, x, y]) for x, y in combinations(others, 2)]
            hit_flag = set([a, b, c]) in bets
            if hit_flag:
                pop_val = _mean_pop(pop_map, [a, b, c])

        elif bet_type == "sanrentan":
            bets = [(key, x, y) for x, y in permutations(others, 2)]
            hit_flag = (a, b, c) in bets
            if hit_flag:
                pop_val = _mean_pop(pop_map, [a, b, c])

        else:
            raise ValueError(f"unknown bet_type: {bet_type}")

        hit += int(hit_flag)
        total += 1
        if hit_flag:
            hit_pops.append(pop_val)

    hit_rate = hit / total if total else 0.0
    avg_pop = float(np.nanmean(hit_pops)) if len(hit_pops) else np.nan
    return total, hit, hit_rate, avg_pop


def evaluate_strategies_with_stats(pred_rank_df: pd.DataFrame, results_df: pd.DataFrame) -> pd.DataFrame:
    """
    上位4/5で
      - BOX: 馬連/馬単/ワイド/三連複
      - 1頭流し: 馬連/馬単/三連複/三連単
    について
      - num_races
      - hits
      - hit_rate
      - avg_popularity_when_hit
    を返す
    """
    rows = []

    for N in [4, 5]:
        for bt in ["umaren", "umatan", "wide", "sanrenpuku"]:
            total, hit, hr, ap = eval_box_stats(pred_rank_df, results_df, N, bt)
            rows.append({
                "topN": N,
                "style": "BOX",
                "bet_type": bt,
                "num_races": total,
                "hits": hit,
                "hit_rate": hr,
                "avg_popularity_when_hit": ap,
            })

        for bt in ["umaren", "umatan", "sanrenpuku", "sanrentan"]:
            total, hit, hr, ap = eval_nagashi_stats(pred_rank_df, results_df, N, bt)
            rows.append({
                "topN": N,
                "style": "NAGASHI(1st)",
                "bet_type": bt,
                "num_races": total,
                "hits": hit,
                "hit_rate": hr,
                "avg_popularity_when_hit": ap,
            })

    df = pd.DataFrame(rows).sort_values(["topN", "style", "bet_type"]).reset_index(drop=True)
    return df



def get_feature_importance(
    model,
    feature_cols: list[str],
    importance_type: str = "gain",
    topk: int = 40,
) -> pd.DataFrame:
    """
    LightGBMモデルから特徴量重要度を取得
    """
    imp = model.feature_importance(importance_type=importance_type)

    df = pd.DataFrame({
        "feature": feature_cols,
        "importance": imp,
    }).sort_values("importance", ascending=False)

    return df.head(topk).reset_index(drop=True)


def plot_feature_importance(
    imp_df: pd.DataFrame,
    title: str = "Feature Importance (gain)",
):
    """
    横棒グラフで可視化
    """
    plt.figure(figsize=(8, max(6, len(imp_df) * 0.25)))
    sns.barplot(
        data=imp_df,
        y="feature",
        x="importance",
        orient="h",
        color="steelblue",
    )
    plt.title(title)
    plt.tight_layout()
    plt.show()
