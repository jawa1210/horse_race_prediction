#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Build empirical confidence calibration (5-level) from test evaluation results.

Inputs:
  - evaluation_df (test) with columns: race_id, horse_id, umaban, rank, score
  - You need trifecta topK generation functions (same as training)

Outputs:
  ../data/03_train/confidence_calib_tri.yaml
"""

from __future__ import annotations
import math
from itertools import permutations
from pathlib import Path
import pandas as pd
import numpy as np
import yaml

MODEL_DIR = Path("..", "data") / "03_train"
DEFAULT_EVAL_PATH = MODEL_DIR / "evaluation.csv"   # ★ファイルを指定
OUT_PATH = MODEL_DIR / "confidence_calib_tri.yaml"



# ---- same PL trifecta generator as yours ----
def plackett_luce_trifecta_topk(
    race_df: pd.DataFrame,
    score_col: str = "score",
    horse_col: str = "horse_id",
    umaban_col: str = "umaban",
    cand_n: int = 8,
    topk: int = 20,
    temperature: float = 1.0,
) -> pd.DataFrame:
    df = race_df.dropna(subset=[score_col]).copy()
    df = df.sort_values(score_col, ascending=False).head(cand_n).copy()
    if len(df) < 3:
        return pd.DataFrame()

    s = df[score_col].astype(float).values
    s = (s - s.max()) / max(temperature, 1e-9)
    w = np.exp(s)

    horses = df[horse_col].tolist()
    umabans = df[umaban_col].tolist() if umaban_col in df.columns else [None] * len(df)
    idxs = list(range(len(horses)))

    rows = []
    denom1 = float(w.sum())
    for i, j, k in permutations(idxs, 3):
        p1 = w[i] / denom1
        denom2 = denom1 - w[i]
        if denom2 <= 0:
            continue
        p2 = w[j] / denom2
        denom3 = denom2 - w[j]
        if denom3 <= 0:
            continue
        p3 = w[k] / denom3
        rows.append({
            "1着_horse_id": horses[i],
            "2着_horse_id": horses[j],
            "3着_horse_id": horses[k],
            "1着": umabans[i],
            "2着": umabans[j],
            "3着": umabans[k],
            "prob": float(p1 * p2 * p3),
        })

    out = (pd.DataFrame(rows)
           .sort_values("prob", ascending=False)
           .head(topk)
           .reset_index(drop=True))
    return out

def trifecta_hit_in_topk(race_df: pd.DataFrame, tri_df: pd.DataFrame) -> int:
    true = race_df.sort_values("rank").head(3)
    if len(true) < 3:
        return 0
    t = tuple(true["horse_id"].tolist())
    cand = list(zip(tri_df["1着_horse_id"], tri_df["2着_horse_id"], tri_df["3着_horse_id"]))
    return 1 if t in cand else 0

# ---- features for confidence ----
def chaos_entropy_from_scores(scores, temperature: float = 1.0) -> float:
    s = pd.to_numeric(pd.Series(scores), errors="coerce").dropna().astype(float).values
    if len(s) == 0:
        return 0.0
    s = s - s.max()
    s = s / max(temperature, 1e-9)
    exps = np.exp(s)
    denom = float(exps.sum())
    if denom <= 0:
        return 0.0
    p = exps / denom
    h = -float(np.sum(p * np.log(p + 1e-12)))
    return h

def marginals_from_tri(tri_df: pd.DataFrame) -> dict:
    if tri_df is None or len(tri_df) == 0:
        return {"p1_max": 0.0, "p1_gap": 0.0, "top1_prob": 0.0, "top5_mass": 0.0}
    x = tri_df.copy()
    x["prob"] = pd.to_numeric(x["prob"], errors="coerce").fillna(0.0)
    mass = float(x["prob"].sum())
    if mass <= 0:
        return {"p1_max": 0.0, "p1_gap": 0.0, "top1_prob": 0.0, "top5_mass": 0.0}

    x["p"] = x["prob"] / mass
    p1 = x.groupby("1着")["p"].sum().sort_values(ascending=False)

    p1_max = float(p1.iloc[0]) if len(p1) >= 1 else 0.0
    p1_2nd = float(p1.iloc[1]) if len(p1) >= 2 else 0.0
    p1_gap = p1_max - p1_2nd

    top1_prob = float(x["p"].sort_values(ascending=False).head(1).sum())
    top5_mass = float(x["p"].sort_values(ascending=False).head(5).sum())

    return {"p1_max": p1_max, "p1_gap": p1_gap, "top1_prob": top1_prob, "top5_mass": top5_mass}

def build_conf_score(entropy: float, marg: dict, entropy_th: float = 1.80) -> float:
    # 経験値化のための“まとめ指標”（後で見直しやすい形）
    # 集中度が高いほど +、団子(entropy)ほど -
    top5 = float(marg.get("top5_mass", 0.0))
    top1 = float(marg.get("top1_prob", 0.0))
    p1max = float(marg.get("p1_max", 0.0))
    p1gap = float(marg.get("p1_gap", 0.0))

    e_norm = min(max(entropy / max(entropy_th, 1e-9), 0.0), 2.0)  # 0..2
    # 係数は“経験値作り”の初期値。最終的にはこのスコアで分位切りして実測率にするので頑丈
    score = (0.45 * top5 + 0.20 * min(top1 * 2.0, 1.0) + 0.25 * p1max + 0.20 * min(p1gap * 2.0, 1.0)) - (0.55 * min(e_norm / 1.2, 1.0))
    return float(score)

def main(
    evaluation_path: Path = DEFAULT_EVAL_PATH,  # ★ここが重要
    cand_n=8, topk=20, temperature=1.0,
    entropy_topn=8, entropy_th=1.80
):
    evaluation_path = Path(evaluation_path)

    if evaluation_path.is_dir():
        raise ValueError(f"evaluation_path must be a file path, got directory: {evaluation_path}")

    if not evaluation_path.exists():
        raise FileNotFoundError(f"evaluation file not found: {evaluation_path}")

    df = pd.read_csv(evaluation_path, sep="\t", dtype={"race_id": str})
    df["rank"] = pd.to_numeric(df["rank"], errors="coerce")
    df["score"] = pd.to_numeric(df["score"], errors="coerce")
    df["umaban"] = pd.to_numeric(df["umaban"], errors="coerce")
    df = df.dropna(subset=["rank", "score", "umaban"])

    rows = []
    for rid, g in df.groupby("race_id"):
        # trifecta
        tri = plackett_luce_trifecta_topk(g, cand_n=cand_n, topk=topk, temperature=temperature)
        hit = trifecta_hit_in_topk(g, tri) if len(tri) else 0

        # entropy from top entropy_topn scores
        top_scores = g.sort_values("score", ascending=False)["score"].head(entropy_topn)
        ent = chaos_entropy_from_scores(top_scores, temperature=1.0)

        marg = marginals_from_tri(tri)
        cs = build_conf_score(ent, marg, entropy_th=entropy_th)

        rows.append({
            "race_id": rid,
            "hit": hit,
            "entropy": ent,
            "conf_score": cs,
            **marg,
        })

    feat = pd.DataFrame(rows)
    if feat.empty:
        raise RuntimeError("No races found for calibration.")

    # 5段階：conf_scoreの分位で切る（経験値ベース）
    # 5が一番当たりやすい
    feat["bucket"] = pd.qcut(feat["conf_score"], 5, labels=[1,2,3,4,5]).astype(int)

    # 実測hit率（この値をHTMLで表示）
    summary = (feat.groupby("bucket")["hit"]
               .agg(["mean","count"])
               .reset_index()
               .sort_values("bucket"))

    # しきい値（境界）を保存（score -> bucket）
    qs = feat["conf_score"].quantile([0.2, 0.4, 0.6, 0.8]).to_dict()
    thresholds = {
        "q20": float(qs[0.2]),
        "q40": float(qs[0.4]),
        "q60": float(qs[0.6]),
        "q80": float(qs[0.8]),
    }

    # bucketごとの経験hit率
    hit_rate = {int(r["bucket"]): float(r["mean"]) for _, r in summary.iterrows()}
    count = {int(r["bucket"]): int(r["count"]) for _, r in summary.iterrows()}

    out = {
        "version": 1,
        "target": f"hit@{topk}",
        "cand_n": int(cand_n),
        "topk": int(topk),
        "temperature": float(temperature),
        "entropy_topn": int(entropy_topn),
        "entropy_th": float(entropy_th),
        "thresholds": thresholds,
        "hit_rate": hit_rate,
        "count": count,
        # 参考：開発時に見たいなら残す
        "conf_score_desc": feat["conf_score"].describe().to_dict(),
    }

    OUT_PATH.parent.mkdir(exist_ok=True, parents=True)
    with open(OUT_PATH, "w") as f:
        yaml.safe_dump(out, f, allow_unicode=True, sort_keys=False)

    print("[OK] wrote:", OUT_PATH)
    print(summary)

if __name__ == "__main__":
    # 例：train_rank_trifecta.py が出した evaluation.tsv を指定
    # ../data/03_train/evaluation.csv でもOK（あなたの命名に合わせて）
    main(evaluation_path=MODEL_DIR / "evaluation.csv")
