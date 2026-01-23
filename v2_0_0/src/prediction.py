# prediction.py
import math
import pickle
from itertools import permutations
from pathlib import Path

import pandas as pd
import yaml

DATA_DIR = Path("..", "data")
MODEL_DIR = DATA_DIR / "03_train"


# ---------------------------
# 共通: config/model/category
# ---------------------------
def _load_feature_cols(config_filepath: Path) -> list[str]:
    with open(config_filepath, "r") as f:
        return yaml.safe_load(f)["features"]


def _load_model(model_filepath: Path):
    with open(model_filepath, "rb") as f:
        return pickle.load(f)


def _load_category_map(category_map_path: Path) -> dict:
    with open(category_map_path, "rb") as f:
        return pickle.load(f)


def _apply_categories(features: pd.DataFrame, feature_cols: list[str], category_map: dict) -> pd.DataFrame:
    """
    学習時と同じカテゴリに揃える（horse_id/jockey_id/trainer_id）
    category_mapに存在しないキーはスキップして安全に動かす。
    """
    feats = features.copy()
    ID_COLS = ["horse_id", "jockey_id", "trainer_id"]

    for c in ID_COLS:
        if c in feature_cols and c in feats.columns:
            cats = category_map.get(c)
            if cats is None:
                continue
            feats[c] = (
                feats[c]
                .astype("string")
                .astype("category")
                .cat.set_categories(cats)
            )
    return feats


# ---------------------------
# 単勝モデル（分類）
# ---------------------------
def predict_win(
    features: pd.DataFrame,
    model_filepath: Path = MODEL_DIR / "model.pkl",
    config_filepath: Path = Path("config.yaml"),
    category_map_path: Path = MODEL_DIR / "category_map_win.pkl",
) -> pd.DataFrame:
    """
    返り値: race_idごとに pred 降順で並んだ DataFrame
      - pred: 単勝モデルの出力（勝率っぽいスコア）
    """
    feature_cols = _load_feature_cols(Path(config_filepath))
    model = _load_model(Path(model_filepath))
    category_map = _load_category_map(Path(category_map_path))

    feats = _apply_categories(features, feature_cols, category_map)

    # 予測
    X = feats[feature_cols]
    out_cols = [c for c in ["race_id", "umaban", "tansyo_odds", "popularity", "agari"] if c in feats.columns]
    pred_df = feats[out_cols].copy()
    pred_df["pred"] = model.predict(X)

    # race内に寄せて整形（任意）
    pred_df = pred_df.sort_values(["race_id", "pred"], ascending=[True, False]).reset_index(drop=True)
    return pred_df


# ---------------------------
# 3連系モデル（ランキング）
# ---------------------------
def predict_rank(
    features: pd.DataFrame,
    model_filepath: Path = MODEL_DIR / "model_tri.pkl",
    config_filepath: Path = Path("config_tri.yaml"),
    category_map_path: Path = MODEL_DIR / "category_map_tri.pkl",
) -> pd.DataFrame:
    """
    返り値: race_idごとに pred_rank 昇順で並んだ DataFrame
      - score: ランキングスコア（大きいほど上位）
      - pred_rank: レース内順位（1が本命）
    """
    feature_cols = _load_feature_cols(Path(config_filepath))
    model = _load_model(Path(model_filepath))
    category_map = _load_category_map(Path(category_map_path))

    feats = _apply_categories(features, feature_cols, category_map)

    X = feats[feature_cols]
    out_cols = [c for c in ["race_id", "umaban", "tansyo_odds", "popularity", "agari"] if c in feats.columns]
    pred_df = feats[out_cols].copy()

    pred_df["score"] = model.predict(X, num_iteration=getattr(model, "best_iteration", None))
    pred_df["pred_rank"] = (
        pred_df.groupby("race_id")["score"]
        .rank(ascending=False, method="first")
        .astype(int)
    )

    pred_df = pred_df.sort_values(["race_id", "pred_rank"]).reset_index(drop=True)
    return pred_df


def trifecta_pl_topk_for_race(
    race_df: pd.DataFrame,
    cand_n: int = 8,
    topk: int = 30,
    temperature: float = 1.0,
) -> pd.DataFrame:
    """
    1レース分の pred_rank_df（score付き）から3連単TopKを生成（Plackett-Luce近似）
    """
    g = race_df.dropna(subset=["score"]).copy()
    g = g.sort_values("score", ascending=False).head(cand_n).copy()
    if len(g) < 3:
        return pd.DataFrame()

    scores = g["score"].astype(float).values
    scores = (scores - scores.max()) / max(temperature, 1e-9)
    weights = [math.exp(s) for s in scores]

    umabans = g["umaban"].tolist() if "umaban" in g.columns else list(range(len(g)))
    idxs = list(range(len(umabans)))

    rows = []
    sumw = sum(weights)

    for i, j, k in permutations(idxs, 3):
        denom1 = sumw
        p1 = weights[i] / denom1

        denom2 = denom1 - weights[i]
        if denom2 <= 0:
            continue
        p2 = weights[j] / denom2

        denom3 = denom2 - weights[j]
        if denom3 <= 0:
            continue
        p3 = weights[k] / denom3

        p = p1 * p2 * p3
        rows.append({"1着": umabans[i], "2着": umabans[j], "3着": umabans[k], "prob": p})

    return pd.DataFrame(rows).sort_values("prob", ascending=False).head(topk).reset_index(drop=True)


def predict_trifecta_topk(
    features: pd.DataFrame,
    model_filepath: Path = MODEL_DIR / "model_tri.pkl",
    config_filepath: Path = Path("config_tri.yaml"),
    category_map_path: Path = MODEL_DIR / "category_map_tri.pkl",
    cand_n: int = 8,
    topk: int = 30,
    temperature: float = 1.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    返り値:
      (pred_rank_df, tri_df)
      - pred_rank_df: 馬単位（score/pred_rank）
      - tri_df: レースごとの3連単TopK（race_id付き）
    """
    pred_rank_df = predict_rank(
        features=features,
        model_filepath=model_filepath,
        config_filepath=config_filepath,
        category_map_path=category_map_path,
    )

    tri_rows = []
    for rid, g in pred_rank_df.groupby("race_id"):
        t = trifecta_pl_topk_for_race(g, cand_n=cand_n, topk=topk, temperature=temperature)
        if t.empty:
            continue
        t.insert(0, "race_id", str(rid))
        tri_rows.append(t)

    tri_df = pd.concat(tri_rows, ignore_index=True) if tri_rows else pd.DataFrame()
    return pred_rank_df, tri_df
