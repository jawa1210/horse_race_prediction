#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Ranking-based (LambdaRank) model for horse racing + Trifecta (3連単) ticket generation.

Input:
  ../data/02_features/features.csv (tab-separated)

Output:
  ../data/03_train/
    - model_rank.pkl
    - category_map.pkl
    - importance_rank.png / .csv
    - eval_horse_level.tsv      (馬単位の予測スコア)
    - eval_trifecta_topk.tsv    (3連単候補TopKと確率)
    - metrics.txt               (TopK 的中率)

Usage:
  python3 train_rank_trifecta.py --test_start_date 2024-01-01 --topk 20 --cand_n 8
"""

import argparse
import math
import pickle
from itertools import permutations
from pathlib import Path

import lightgbm as lgb
import matplotlib.pyplot as plt
import pandas as pd
import yaml

DATA_DIR = Path("..", "data")
INPUT_DIR = DATA_DIR / "02_features"
OUTPUT_DIR = DATA_DIR / "03_train"
CONFIG_FILE = Path("config_tri.yaml")


# ----------------------------
# Config YAML generator
# ----------------------------
def create_yaml(delete_col=None):
    if delete_col is None:
        delete_col = ["race_id", "date", "rank", "target", "popularity"]  # popularity入れない運用の例

    features = pd.read_csv(INPUT_DIR / "features.csv", sep="\t")

    feature_cols = [c for c in features.columns if c not in delete_col]

    params = {
        # ranking
        "objective": "lambdarank",
        "metric": "ndcg",
        "ndcg_eval_at": [1, 3, 5],
        # training stability
        "learning_rate": 0.03,
        "num_leaves": 63,
        "min_data_in_leaf": 40,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "lambda_l1": 0.0,
        "lambda_l2": 1.0,
        "max_depth": -1,
        "verbosity": -1,
        "seed": 100,
    }

    config = {
        "features": feature_cols,
        "params": params,
    }

    with open(CONFIG_FILE, "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    print(f"[OK] YAML created: {CONFIG_FILE}")
    print(f"[OK] num_features: {len(feature_cols)}")


# ----------------------------
# Helpers: safety + ranking label + group
# ----------------------------
def _to_numeric_safe(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """
    Ensure numeric columns are numeric; coerce errors to NaN.
    """
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def _make_ranking_label(df: pd.DataFrame) -> pd.Series:
    """
    relevance: 1着=3, 2着=2, 3着=1, その他=0
    """
    r = pd.to_numeric(df["rank"], errors="coerce")
    label = (4 - r).clip(lower=0, upper=3).fillna(0).astype(int)
    return label


def _make_group(df: pd.DataFrame) -> list[int]:
    """
    group sizes per race_id, in the order of df rows grouped by race_id.
    Note: We sort by race_id to align group with dataset order.
    """
    return df.groupby("race_id").size().tolist()


# ----------------------------
# Trifecta probability (Plackett-Luce approx)
# ----------------------------
def plackett_luce_trifecta_topk(
    race_df: pd.DataFrame,
    score_col: str = "score",
    horse_col: str = "horse_id",
    umaban_col: str = "umaban",
    cand_n: int = 8,
    topk: int = 20,
    temperature: float = 1.0,
) -> pd.DataFrame:
    """
    Given horses in ONE race with a score, generate trifecta (i,j,k) probabilities
    using Plackett-Luce:
      w_i = exp(score_i / T)
      P(i,j,k) = w_i/sum(w) * w_j/sum(w\i) * w_k/sum(w\i,j)

    We restrict to top cand_n horses by score to keep it fast.
    """
    df = race_df.copy()
    df = df.dropna(subset=[score_col])
    df = df.sort_values(score_col, ascending=False).head(cand_n).copy()
    if len(df) < 3:
        return pd.DataFrame()

    # weights
    # subtract max for numerical stability
    s = df[score_col].astype(float).values
    s = (s - s.max()) / max(temperature, 1e-9)
    w = [math.exp(x) for x in s]

    horses = df[horse_col].tolist()
    umabans = df[umaban_col].tolist() if umaban_col in df.columns else [None] * len(df)

    # map index -> weight
    idxs = list(range(len(horses)))

    rows = []
    for i, j, k in permutations(idxs, 3):
        # P(i first)
        denom1 = sum(w)
        p1 = w[i] / denom1

        # P(j second | i first)
        denom2 = denom1 - w[i]
        if denom2 <= 0:
            continue
        p2 = w[j] / denom2

        # P(k third | i first, j second)
        denom3 = denom2 - w[j]
        if denom3 <= 0:
            continue
        p3 = w[k] / denom3

        p = p1 * p2 * p3
        rows.append(
            {
                "first_horse_id": horses[i],
                "second_horse_id": horses[j],
                "third_horse_id": horses[k],
                "first_umaban": umabans[i],
                "second_umaban": umabans[j],
                "third_umaban": umabans[k],
                "prob": p,
            }
        )

    out = pd.DataFrame(rows).sort_values("prob", ascending=False).head(topk).reset_index(drop=True)
    return out


def trifecta_hit_in_topk(race_df: pd.DataFrame, trifecta_df: pd.DataFrame) -> int:
    """
    race_df includes true rank per horse.
    trifecta_df: candidate trifectas.
    Returns 1 if the true (1st,2nd,3rd) horse_id tuple appears in trifecta_df else 0.
    """
    true = race_df.sort_values("rank").head(3)
    if len(true) < 3:
        return 0
    t = tuple(true["horse_id"].tolist())
    cand = list(
        zip(
            trifecta_df["first_horse_id"].tolist(),
            trifecta_df["second_horse_id"].tolist(),
            trifecta_df["third_horse_id"].tolist(),
        )
    )
    return 1 if t in cand else 0


# ----------------------------
# Trainer
# ----------------------------
class Trainer:
    def __init__(
        self,
        features_filepath: Path = INPUT_DIR / "features.csv",
        config_filepath: Path = "config.yaml",
        output_dir: Path = OUTPUT_DIR,
    ):
        self.features = pd.read_csv(features_filepath, sep="\t")
        self.features["date"] = pd.to_datetime(self.features["date"])

        with open(config_filepath, "r") as file:
            config = yaml.safe_load(file)
            self.feature_cols = config["features"]
            self.params = config["params"]

        output_dir.mkdir(exist_ok=True, parents=True)
        self.output_dir = output_dir

        self.ID_COLS = ["horse_id", "jockey_id", "trainer_id"]

        # 🔑 ここでランキング用に強制上書き
        self.params.update({
            "objective": "lambdarank",
            "metric": "ndcg",
            "ndcg_eval_at": [1, 3, 5],
            "learning_rate": 0.03,
            "num_leaves": 63,
            "min_data_in_leaf": 40,
            "verbosity": -1,
            "seed": 100,
        })

    def create_dataset(self, test_start_date: str):
        test_start = pd.to_datetime(test_start_date)

        self.features["rank"] = pd.to_numeric(self.features["rank"], errors="coerce")
        self.features = self.features.dropna(subset=["rank"])

        self.train_df = self.features.query("date < @test_start").copy()
        self.test_df  = self.features.query("date >= @test_start").copy()

        # race_id 単位で group が正しくなるように並べ替え
        self.train_df = self.train_df.sort_values(["race_id", "umaban"]).reset_index(drop=True)
        self.test_df  = self.test_df.sort_values(["race_id", "umaban"]).reset_index(drop=True)

    def _make_label(self, df: pd.DataFrame) -> pd.Series:
        # 1着3,2着2,3着1,その他0
        return (4 - df["rank"]).clip(0, 3).astype(int)

    def _make_group(self, df: pd.DataFrame):
        return df.groupby("race_id").size().tolist()

    def _apply_category(self, train_df, test_df):
        cat_cols = [c for c in self.ID_COLS if c in self.feature_cols]

        category_map = {}
        for c in cat_cols:
            s = train_df[c].astype("string")
            category_map[c] = sorted(s.dropna().unique())
            train_df[c] = s.astype("category").cat.set_categories(category_map[c])
            test_df[c]  = test_df[c].astype("string").astype("category").cat.set_categories(category_map[c])

        with open(self.output_dir / "category_map_tri.pkl", "wb") as f:
            pickle.dump(category_map, f)

        return cat_cols

    def train(self, train_df, test_df, model_filename, importance_filename):
        train_df = train_df.copy()
        test_df  = test_df.copy()

        train_df["label"] = self._make_label(train_df)
        test_df["label"]  = self._make_label(test_df)

        group_train = self._make_group(train_df)
        group_valid = self._make_group(test_df)

        cat_cols = self._apply_category(train_df, test_df)

        lgb_train = lgb.Dataset(
            train_df[self.feature_cols],
            label=train_df["label"],
            group=group_train,
            categorical_feature=cat_cols,
            free_raw_data=False,
        )
        lgb_valid = lgb.Dataset(
            test_df[self.feature_cols],
            label=test_df["label"],
            group=group_valid,
            reference=lgb_train,
            categorical_feature=cat_cols,
            free_raw_data=False,
        )

        model = lgb.train(
            self.params,
            lgb_train,
            num_boost_round=15000,
            valid_sets=[lgb_valid],
            callbacks=[
                lgb.log_evaluation(100),
                lgb.early_stopping(300),
            ],
        )

        with open(self.output_dir / model_filename, "wb") as f:
            pickle.dump(model, f)

        lgb.plot_importance(model, importance_type="gain", figsize=(30, 15))
        plt.savefig(self.output_dir / f"{importance_filename}.png")
        plt.close()

        evaluation_df = test_df[
            ["race_id", "horse_id", "rank", "umaban", "tansyo_odds", "popularity"]
        ].copy()
        evaluation_df["score"] = model.predict(
            test_df[self.feature_cols],
            num_iteration=model.best_iteration
        )
        evaluation_df["pred_rank"] = (
            evaluation_df.groupby("race_id")["score"]
            .rank(ascending=False, method="first")
        )

        self.model = model
        return evaluation_df

    def run(
        self,
        test_start_date: str,
        importance_filename="importance",
        model_filename="model_tri.pkl",
        evaluation_filename="evaluation.csv",
    ):
        self.create_dataset(test_start_date)
        evaluation_df = self.train(
            self.train_df,
            self.test_df,
            model_filename,
            importance_filename,
        )
        evaluation_df.to_csv(self.output_dir / evaluation_filename, sep="\t", index=False)
        return evaluation_df
