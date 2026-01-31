# v2_0_0/src/train_strategy_selector.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import pickle

import numpy as np
import pandas as pd
from tqdm.notebook import tqdm

import lightgbm as lgb  # model.pkl が lightgbm Booster の想定

from feature_engineering import RAW_DATA_DIR, DATA_DIR
from recommend_strategy import (
    build_training_table,          # 特徴量 + best_strategy を作る（あなたの既存）
    train_strategy_selector,       # sklearnで学習して strategy_selector.pkl を保存（あなたの既存）
    LabelConfig,
    OUT_MODEL,
    OUT_FEATURES,
)

# =========================================================
# Paths
# =========================================================
RESULTS_DIR = DATA_DIR / "results_cache"
RESULTS_FLAT = RESULTS_DIR / "results_flat.csv"
PAYOUTS_FLAT = RESULTS_DIR / "payouts_flat.csv"

FEATURES_PATH = DATA_DIR / "02_features" / "features.csv"
MODEL_TRI_PATH = DATA_DIR / "03_train" / "model_tri.pkl"  # 3連系rankingモデル（例）
RACE_INFO_PATH = RAW_DATA_DIR / "rawdf" / "race_info.csv"


# =========================================================
# Utils
# =========================================================


CATMAP_PATH = DATA_DIR / "03_train" / "category_map_tri.pkl"   # 実パスに合わせて

def _load_catmap():
    with open(CATMAP_PATH, "rb") as f:
        return pickle.load(f)

def _encode_cats_for_predict(df: pd.DataFrame, feature_cols: list[str], catmap: dict) -> np.ndarray:
    X = df[feature_cols].copy()

    # catmap にある列は "学習時カテゴリ順" で index 化
    for c, cats in catmap.items():
        if c not in X.columns:
            continue
        idx = {str(v): i for i, v in enumerate(cats)}
        X[c] = X[c].astype("string").map(lambda v: idx.get(str(v), -1)).astype("int32")

    # それ以外は数値化
    for c in X.columns:
        if c in catmap:
            continue
        X[c] = pd.to_numeric(X[c], errors="coerce")

    X = X.fillna(0.0)
    return X.values

def _normalize_tri_df(tri_df: pd.DataFrame) -> pd.DataFrame:
    if tri_df is None or tri_df.empty:
        return pd.DataFrame()
    df = tri_df.copy()

    # どっち形式でも受ける
    rename_map = {}
    if "first_umaban" in df.columns:  rename_map["first_umaban"] = "1着"
    if "second_umaban" in df.columns: rename_map["second_umaban"] = "2着"
    if "third_umaban" in df.columns:  rename_map["third_umaban"] = "3着"
    df = df.rename(columns=rename_map)

    # 必須列が揃ってなければ空返し
    need = {"1着","2着","3着","prob"}
    if not need.issubset(set(df.columns)):
        return pd.DataFrame()

    return df[["race_id","1着","2着","3着","prob"]] if "race_id" in df.columns else df[["1着","2着","3着","prob"]]

def _ensure_exists(p: Path, name: str):
    if not p.exists():
        raise FileNotFoundError(f"{name} not found: {p}")


def _read_tab_or_csv(path: Path, dtype: dict | None = None) -> pd.DataFrame:
    """
    sep が \t の可能性もあるので自動推定する。
    """
    text = path.read_text(encoding="utf-8", errors="ignore")
    sep = "\t" if "\t" in text.splitlines()[0] else ","
    return pd.read_csv(path, sep=sep, dtype=dtype)


def _parse_date_col(df: pd.DataFrame, col: str = "date") -> pd.DataFrame:
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


def _date_str(d) -> str:
    return pd.to_datetime(d).strftime("%Y-%m-%d")


def _collect_trainable_dates(
    train_start_date: str | None = None,
) -> list[str]:
    """
    results_flat と payouts_flat の両方に存在する日付だけを返す
    ※ population.csv を使わない
    """
    if not RESULTS_FLAT.exists() or not PAYOUTS_FLAT.exists():
        return []

    df_r = _read_tab_or_csv(RESULTS_FLAT, dtype={"race_id": str})
    df_p = _read_tab_or_csv(PAYOUTS_FLAT, dtype={"race_id": str})

    if "date" not in df_r.columns or "date" not in df_p.columns:
        raise RuntimeError("results_flat / payouts_flat に date 列が必要です（update_* を回して作ってください）")

    df_r = _parse_date_col(df_r, "date")
    df_p = _parse_date_col(df_p, "date")

    dates = sorted(set(df_r["date"].dropna().dt.strftime("%Y-%m-%d")) & set(df_p["date"].dropna().dt.strftime("%Y-%m-%d")))

    if train_start_date is not None:
        d0 = pd.to_datetime(train_start_date)
        dates = [d for d in dates if pd.to_datetime(d) >= d0]

    return dates


def _load_race_info() -> pd.DataFrame:
    """
    common/data/raw_df/race_info.csv を読み、
    少なくとも race_id, date を持つ DataFrame を返す
    """
    _ensure_exists(RACE_INFO_PATH, "race_info.csv")

    df = _read_tab_or_csv(RACE_INFO_PATH, dtype={"race_id": str})

    # race_info.csv に date が無いケース（あなたの貼った例は無い可能性あり）に備えて、
    # info2 から "2023年7月22日" を抽出する保険を入れる
    if "date" not in df.columns:
        # info2 が list文字列のように入ってる想定
        # 例: "['2023年7月22日', '1回札幌1日目', ...]"
        if "info2" not in df.columns:
            raise RuntimeError("race_info.csv に date も info2 もありません。date が作れません。")

        s = df["info2"].astype(str)
        m = s.str.extract(r"(\d{4})年(\d{1,2})月(\d{1,2})日")
        df["date"] = pd.to_datetime(
            m[0].fillna("1900") + "-" + m[1].fillna("1").str.zfill(2) + "-" + m[2].fillna("1").str.zfill(2),
            errors="coerce",
        )

    df = _parse_date_col(df, "date")
    df["race_id"] = df["race_id"].astype(str)

    # date が取れなかった行は落とす
    df = df.dropna(subset=["race_id", "date"]).copy()
    df["date_str"] = df["date"].dt.strftime("%Y-%m-%d")
    return df


def _load_features() -> pd.DataFrame:
    _ensure_exists(FEATURES_PATH, "features.csv")
    df = _read_tab_or_csv(FEATURES_PATH, dtype={"race_id": str, "horse_id": str})
    df = _parse_date_col(df, "date")

    df = _normalize_race_id_col(df, "race_id")   # ★追加（最重要）

    for c in ["umaban", "tansyo_odds", "popularity", "rank"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df["race_id"] = df["race_id"].astype(str)
    df["horse_id"] = df["horse_id"].astype(str)
    return df


def _normalize_race_id_col(df: pd.DataFrame, col: str = "race_id") -> pd.DataFrame:
    if col in df.columns:
        df[col] = (
            df[col].astype(str)
                 .str.replace(r"\.0$", "", regex=True)  # 202504010101.0 → 202504010101
                 .str.zfill(12)                         # 念のため 12桁
        )
    return df



def _load_model_tri():
    _ensure_exists(MODEL_TRI_PATH, "model_tri.pkl")
    with open(MODEL_TRI_PATH, "rb") as f:
        model = pickle.load(f)
    return model


def _predict_score(model, X):
    """
    LightGBM Booster.predict に pandas.DataFrame を渡すと
    pandas categorical の整合性チェックで落ちることがある。
    → 内部で numpy に落として回避する。
    ついでに Booster.feature_name() があれば列順も揃える。
    """
    import numpy as np
    import pandas as pd

    if not hasattr(model, "predict"):
        raise TypeError(f"unsupported model type: {type(model)}")

    # DataFrame なら feature 順を model に合わせてから numpy 化
    if isinstance(X, pd.DataFrame):
        try:
            feat_names = model.feature_name()
            if feat_names:
                missing = [c for c in feat_names if c not in X.columns]
                if missing:
                    raise ValueError(f"missing features for predict: {missing[:10]}")
                # 余計な列があってもOK。必要列だけ、正しい順で。
                X = X[feat_names]
        except Exception:
            # feature_name が取れないなどの場合は、今の列順で通す（一致前提）
            pass

        return np.asarray(model.predict(X.to_numpy()))

    # ndarray 等はそのまま
    return np.asarray(model.predict(X))




# =========================================================
# Trifecta generation (Plackett-Luce approx) - 既存ロジック踏襲
# =========================================================
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
    1レース内の馬スコアから 3連単(i,j,k)の確率を Plackett-Luce で近似し TopK を返す
    """
    import math
    from itertools import permutations

    df = race_df.copy()
    df = df.dropna(subset=[score_col, umaban_col])
    df = df.sort_values(score_col, ascending=False).head(cand_n).copy()
    if len(df) < 3:
        return pd.DataFrame()

    s = df[score_col].astype(float).values
    s = (s - s.max()) / max(temperature, 1e-9)
    w = np.exp(s)

    horses = df[horse_col].tolist()
    umabans = df[umaban_col].astype(int).tolist()

    idxs = list(range(len(horses)))
    rows = []
    denom1 = float(w.sum())

    for i, j, k in permutations(idxs, 3):
        p1 = float(w[i] / denom1)
        denom2 = denom1 - float(w[i])
        if denom2 <= 0:
            continue
        p2 = float(w[j] / denom2)
        denom3 = denom2 - float(w[j])
        if denom3 <= 0:
            continue
        p3 = float(w[k] / denom3)

        rows.append(
            {
                "race_id": str(df["race_id"].iloc[0]),
                "first_horse_id": horses[i],
                "second_horse_id": horses[j],
                "third_horse_id": horses[k],
                "first_umaban": umabans[i],
                "second_umaban": umabans[j],
                "third_umaban": umabans[k],
                "prob": p1 * p2 * p3,
            }
        )

    out = pd.DataFrame(rows)
    out = out.sort_values("prob", ascending=False).head(topk).reset_index(drop=True)
    return out


# =========================================================
# Offline cache builder (NO population.csv, NO API)
# =========================================================
@dataclass
class CacheConfig:
    cand_n: int = 8
    topk: int = 20
    temperature: float = 1.0


def build_prediction_cache_for_date_offline(
    date: str,
    features_all: pd.DataFrame,
    model_tri,
    feature_cols: list[str] | None = None,
    cfg: CacheConfig = CacheConfig(),
):
    """
    features.csv（過去データ） + model_tri から、
    recommend_strategy が欲しがる “キャッシュ形式” を作る。

    Returns:
      pred_rank_all: DataFrame（全レース全馬）
      tri_all      : DataFrame（全レース 3連単候補 TopK）
      pred_order_by_race: dict[race_id] -> list[int umaban]（スコア順）
      odds_map_by_race  : dict[race_id] -> dict[int umaban] -> float(単勝オッズ)
      meta             : dict（デバッグ用）
    """
    d = pd.to_datetime(date)
    df = features_all.loc[features_all["date"] == d].copy()
    if df.empty:
        return (
            pd.DataFrame(),
            pd.DataFrame(),
            {},
            {},
            {"ok": False, "reason": f"no features for date={date}"},
        )

    # 必須列チェック
    need_cols = ["race_id", "horse_id", "umaban"]
    for c in need_cols:
        if c not in df.columns:
            raise RuntimeError(f"features.csv に必須列 {c} がありません")

    # feature_cols: 指定が無ければ推定
    if feature_cols is None:
        delete = set(["race_id", "date", "rank", "target", "popularity"])
        feature_cols = [c for c in df.columns if c not in delete]

    # ---- 重要：ここでカテゴリ列問題を潰す（最小修正） ----
    # model_tri が LightGBM Booster で「学習時にカテゴリを持っていた」場合、
    # pandasのカテゴリが合わず落ちることがある → predictは numpy に落としてしまう
    X = df[feature_cols].copy()
    for c in X.columns:
        X[c] = pd.to_numeric(X[c], errors="coerce")
    X = X.fillna(0.0)

    # predict
    score = _predict_score(model_tri, X)  # DataFrame のまま渡す
    df["score"] = score

    # pred_rank_all
    pred_rank_all = df[["race_id", "horse_id", "umaban"]].copy()
    pred_rank_all["score"] = df["score"].astype(float)
    if "tansyo_odds" in df.columns:
        pred_rank_all["tansyo_odds"] = df["tansyo_odds"]
    if "popularity" in df.columns:
        pred_rank_all["popularity"] = df["popularity"]

    pred_rank_all["pred_rank"] = (
        pred_rank_all.groupby("race_id")["score"]
        .rank(ascending=False, method="first")
    )

    # pred_order_by_race
    pred_order_by_race: dict[str, list[int]] = {}
    for rid, g in pred_rank_all.groupby("race_id"):
        g2 = g.dropna(subset=["umaban", "score"]).sort_values("score", ascending=False)
        pred_order_by_race[str(rid)] = g2["umaban"].astype(int).tolist()

    # odds_map_by_race（umaban -> float odds）
    odds_map_by_race: dict[str, dict[int, float]] = {}
    for rid, g in df.groupby("race_id"):
        m: dict[int, float] = {}
        if "tansyo_odds" in g.columns:
            for u, o in zip(g["umaban"], g["tansyo_odds"]):
                if pd.isna(u) or pd.isna(o):
                    continue
                try:
                    m[int(u)] = float(o)
                except Exception:
                    continue
        odds_map_by_race[str(rid)] = m

    # tri_all（レースごとに作って結合）
    tri_rows = []
    for rid, g in pred_rank_all.groupby("race_id"):
        race_df = g.copy()
        race_df["race_id"] = str(rid)

        tri = plackett_luce_trifecta_topk(
            race_df=race_df,
            score_col="score",
            horse_col="horse_id",
            umaban_col="umaban",
            cand_n=cfg.cand_n,
            topk=cfg.topk,
            temperature=cfg.temperature,
        )
        if len(tri):
            # ★ここが超重要：recommend_strategy が欲しい列名に揃える
            tri = tri.rename(columns={
                "first_umaban": "1着",
                "second_umaban": "2着",
                "third_umaban": "3着",
            })
            tri["race_id"] = str(rid)
            tri_rows.append(tri)

    tri_all = pd.concat(tri_rows, ignore_index=True) if tri_rows else pd.DataFrame()

    meta = {
        "ok": True,
        "date": date,
        "n_races": int(pred_rank_all["race_id"].nunique()),
        "n_rows": int(len(pred_rank_all)),
        "n_tri_rows": int(len(tri_all)),
        "feature_cols": feature_cols,
    }

    return pred_rank_all, tri_all, pred_order_by_race, odds_map_by_race, meta



# =========================================================
# Trainer
# =========================================================
class TrainerStrategySelector:
    """
    population.csv を使わない版:
      - date/race_id は race_info.csv 起点
      - 予測キャッシュは features.csv + model_tri から再構成（API叩かない）
      - results_flat / payouts_flat がある日だけ学習・評価
    """
    def __init__(
        self,
        label_cfg: LabelConfig = LabelConfig(stake_per_ticket=100, alpha_ticket_penalty=0.18, require_hit=False),
        cache_cfg: CacheConfig = CacheConfig(),
        feature_cols: list[str] | None = None,
    ):
        self.label_cfg = label_cfg
        self.cache_cfg = cache_cfg
        self.feature_cols = feature_cols  # Noneなら自動推定

        self.model_payload = None

        # heavy loads once
        _ensure_exists(RESULTS_FLAT, "results_flat.csv")
        _ensure_exists(PAYOUTS_FLAT, "payouts_flat.csv")
        _ensure_exists(FEATURES_PATH, "features.csv")
        _ensure_exists(MODEL_TRI_PATH, "model_tri.pkl")
        _ensure_exists(RACE_INFO_PATH, "race_info.csv")

        self.race_info = _load_race_info()
        self.features_all = _load_features()
        self.model_tri = _load_model_tri()

        # 学習可能日（results_flat と payouts_flat の共通集合）
        dates_rp = _collect_trainable_dates(train_start_date=None)

        # ★features が存在する日だけに絞る
        feat_dates = set(self.features_all["date"].dropna().dt.strftime("%Y-%m-%d").tolist())

        self.trainable_dates_all = sorted(set(dates_rp) & feat_dates)

        print("[INFO] trainable_dates (results&payouts):", len(dates_rp))
        print("[INFO] trainable_dates (+features):", len(self.trainable_dates_all))


    def create_dataset(
        self,
        train_start_date: str | None = None,
        test_start_date: str = "2026-01-01",
        skip_agg_horse: bool = True,  # 互換のため残す（ここでは使わない）
    ):
        """
        - train_start_date より前は捨てる（任意）
        - test_start_date 以降は test
        """
        test_start = pd.to_datetime(test_start_date)

        dates = self.trainable_dates_all
        if train_start_date is not None:
            d0 = pd.to_datetime(train_start_date)
            dates = [d for d in dates if pd.to_datetime(d) >= d0]

        # split
        train_dates = [d for d in dates if pd.to_datetime(d) < test_start]
        test_dates  = [d for d in dates if pd.to_datetime(d) >= test_start]

        self.train_dates = train_dates
        self.test_dates = test_dates

        self.train_table = self._build_table_for_dates(train_dates)
        self.test_table  = self._build_table_for_dates(test_dates)

    def _build_table_for_dates(self, date_list: list[str]) -> pd.DataFrame:
        all_rows = []

        pbar = tqdm(date_list, desc="Building strategy training data")
        for date in pbar:
            pbar.set_postfix_str(str(date))

            # その日に race_info が存在しないならスキップ（でも results/payouts はあるはず）
            # race_id list を使いたい時にここで取れる（将来拡張用）
            # race_ids = self.race_info.loc[self.race_info["date_str"] == date, "race_id"].tolist()

            pred_rank_all, tri_all, pred_order_by_race, odds_map_by_race, meta = build_prediction_cache_for_date_offline(
                date=date,
                features_all=self.features_all,
                model_tri=self.model_tri,
                feature_cols=self.feature_cols,
                cfg=self.cache_cfg,
            )
            if not meta.get("ok", False):
                print("[SKIP]", date, meta.get("reason"))
                continue

            tri_all = _normalize_tri_df(tri_all)
            if tri_all.empty:
                print("[WARN] tri_all empty:", date, "races:", meta.get("n_races"))

            if not meta.get("ok", False) or pred_rank_all.empty:
                continue

            tri_all = _normalize_tri_df(tri_all)

            df = build_training_table(
                date_list=[date],
                pred_rank_df_all=pred_rank_all,
                tri_df_all=tri_all,
                pred_order_by_race=pred_order_by_race,
                odds_map_by_race=odds_map_by_race,
                label_cfg=self.label_cfg,
            )

            if len(df):
                all_rows.append(df)

        return pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()

    def train(self):
        if self.train_table is None or self.train_table.empty:
            raise RuntimeError("train_table is empty (trainable_dates or features/model/results/payouts を確認)")

        OUT_FEATURES.parent.mkdir(parents=True, exist_ok=True)
        self.train_table.to_csv(OUT_FEATURES, sep="\t", index=False)

        payload = train_strategy_selector(self.train_table)
        self.model_payload = payload
        return payload

    def evaluate(self) -> dict:
        if self.test_table is None or self.test_table.empty:
            return {"ok": False, "reason": "test_table empty"}

        if self.model_payload is None:
            if not OUT_MODEL.exists():
                return {"ok": False, "reason": "model not trained"}
            with open(OUT_MODEL, "rb") as f:
                self.model_payload = pickle.load(f)

        feature_cols = self.model_payload["feature_cols"]
        le = self.model_payload["label_encoder"]
        model = self.model_payload["model"]

        df = self.test_table.copy()
        for c in feature_cols:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

        X = df[feature_cols].values
        y_true = le.transform(df["best_strategy"].astype(str).values)
        y_pred = model.predict(X)
        acc = float((y_true == y_pred).mean())

        return {
            "ok": True,
            "acc": acc,
            "n_test": int(len(df)),
            "n_test_dates": int(len(getattr(self, "test_dates", []))),
        }

    def run(
        self,
        test_start_date: str,
        train_start_date: str | None = None,
    ):
        self.create_dataset(train_start_date=train_start_date, test_start_date=test_start_date)
        payload = self.train()
        metrics = self.evaluate()
        return payload, metrics


# =========================================================
# Convenience main
# =========================================================
def main(
    train_start_date: str | None = None,
    test_start_date: str = "2026-01-01",
    cand_n: int = 8,
    topk: int = 20,
    temperature: float = 1.0,
):
    tss = TrainerStrategySelector(
        label_cfg=LabelConfig(stake_per_ticket=100, alpha_ticket_penalty=0.18, require_hit=False),
        cache_cfg=CacheConfig(cand_n=cand_n, topk=topk, temperature=temperature),
    )
    payload, metrics = tss.run(
        test_start_date=test_start_date,
        train_start_date=train_start_date,
    )
    print("[OK] trained strategy selector")
    print("metrics:", metrics)
    return payload, metrics
