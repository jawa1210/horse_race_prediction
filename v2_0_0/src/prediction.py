import pickle
from pathlib import Path
import pandas as pd
import yaml

DATA_DIR=Path("..","data")
MODEL_DIR=DATA_DIR/"03_train"

def predict(
        features: pd.DataFrame,
        model_filepath: Path=MODEL_DIR/"model.pkl",
        config_filepath: Path="config.yaml"
):
    with open(config_filepath,"r") as f:
        feature_cols=yaml.safe_load(f)["features"]
    with open(model_filepath,"rb") as f:
        model=pickle.load(f)
    ID_COLS = ["horse_id", "jockey_id", "trainer_id"]
    with open("../data/03_train/category_map.pkl", "rb") as f:
        category_map = pickle.load(f)
    for c in ID_COLS:
        if c in feature_cols and c in features.columns:
            features[c] = features[c].astype("string").astype("category").cat.set_categories(category_map[c])
    prediction_df=features[["race_id","umaban","tansyo_odds","popularity","agari"]].copy()
    prediction_df["pred"]=model.predict(features[feature_cols])
    return prediction_df.sort_values("pred",ascending=False)