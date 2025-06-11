import pickle
from pathlib import Path

import lightgbm as lgb
import matplotlib.pyplot as plt
import pandas as pd
import yaml
from sklearn.metrics import log_loss

DATA_DIR = Path("..", "data")
INPUT_DIR = DATA_DIR / "02_features"
OUTPUT_DIR = DATA_DIR / "03_train"
CONFIG_FILE = Path("config.yaml")

def create_yaml(delete_col=["race_id", "date", "rank", "target","owner_id"] ):
    features = pd.read_csv(INPUT_DIR / "features.csv", sep="\t")

    # 使用する特徴量を定義
    feature_cols = [
        col for col in features.columns
        if col not in  delete_col# 除外するカラムを指定
    ]

    # パラメータの設定
    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "random_state": 100,
    }

    # YAML ファイルの内容を定義
    config = {
        "features": feature_cols,
        "params": params,
    }

    # YAML ファイルに書き出し
    with open(CONFIG_FILE, "w") as file:
        yaml.dump(config, file, default_flow_style=False)

    print(f"YAML ファイルを作成しました: {CONFIG_FILE}")


class Trainer:
    def __init__(
        self,
        features_filepath: Path = INPUT_DIR / "features.csv",
        config_filepath: Path = "config.yaml",
        output_dir: Path = OUTPUT_DIR,
    ):
        self.features = pd.read_csv(features_filepath, sep="\t")
        with open(config_filepath, "r") as file:
            config = yaml.safe_load(file)
            self.feature_cols = config["features"]
            self.params = config["params"]
        output_dir.mkdir(exist_ok=True, parents=True)
        self.output_dir = output_dir

    def create_dataset(self, test_start_date: str):
        # 目的変数
        self.features["target"] = (self.features["rank"] == 1).astype(int)
        # 学習データとテストデータに分割
        self.train_df = self.features.query("date < @test_start_date")
        self.test_df = self.features.query("date >= @test_start_date")

    def train(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        model_filename: str,
        importance_filename: str,
    ) -> pd.DataFrame:
        # 学習データと検証データを作成
        lgb_train = lgb.Dataset(train_df[self.feature_cols], train_df["target"])
        lgb_valid = lgb.Dataset(
            test_df[self.feature_cols], test_df["target"], reference=lgb_train
        )

        # 学習の実行
        model = lgb.train(
            params=self.params,
            train_set=lgb_train,
            num_boost_round=10000,
            valid_sets=[lgb_valid],  # evalua
            valid_names=["valid"],  # 検証データの名前
            callbacks=[
                lgb.log_evaluation(100),
                lgb.early_stopping(stopping_rounds=100),
            ],
        )

        # モデルの保存
        self.best_params = model.params
        with open(self.output_dir / model_filename, "wb") as f:
            pickle.dump(model, f)

        # 特徴量重要度の可視化
        lgb.plot_importance(
            model, importance_type="gain", figsize=(30, 15), max_num_features=50
        )
        plt.savefig(self.output_dir / f"{importance_filename}.png")
        plt.close()

        # 特徴量重要度を保存
        importance_df = pd.DataFrame(
            {
                "feature": model.feature_name(),
                "importance": model.feature_importance(importance_type="gain"),
            }
        ).sort_values("importance", ascending=False)
        importance_df.to_csv(
            self.output_dir / f"{importance_filename}.csv", index=False, sep="\t"
        )

        # テストデータに対してスコアリング
        evaluation_df = test_df[
            [
                "race_id",
                "horse_id",
                "target",
                "rank",
                "tansyo_odds",
                "popularity",
                "umaban",
            ]
        ].copy()
        evaluation_df["pred"] = model.predict(
            test_df[self.feature_cols], num_iteration=model.best_iteration
        )
        logloss = log_loss(evaluation_df["target"], evaluation_df["pred"])
        print("-" * 20 + "result" + "-" * 20)
        print(f"test_df's binary_logloss: {logloss}")
        return evaluation_df


    def run(
        self,
        test_start_date: str,
        importance_filename: str = "importance",
        model_filename: str = "model.pkl",
        evaluation_filename: str = "evaluation.csv",
    ):
        """
        学習処理を実行する。test_start_dateをYYYY-MM-DD形式で指定すると、
        その日付以降のデータをテストデータに、
        それより前のデータを学習データに分割する
        """
        self.create_dataset(test_start_date)
        print(self.train_df.shape, self.test_df.shape)
        evaluation_df = self.train(
            self.train_df,
            self.test_df,
            model_filename,
            importance_filename,
        )
        evaluation_df.to_csv(
            self.output_dir / evaluation_filename, sep="\t", index=False
        )
        return evaluation_df