import pickle
from pathlib import Path
import pandas as pd
import yaml

DATA_DIR=Path("..","data")
MODEL_DIR=DATA_DIR/"03_train"

def predict(
        features: pd.DataFrame,
        model_filepath: Path=MODEL_DIR/"model.pickle",
        config_filepath: Path="config.yaml"
):
    with open(config_filepath,"r") as f:
        feature_cols=yaml.safe_load(f)["features"]
    with open(model_filepath,"rb") as f:
        model=pickle.load(f)
    prediction_df=features[["race_id","umaban","tansyo_odds","popularity","agari"]].copy()
    prediction_df["pred"]=model.predict(features[feature_cols])
    return prediction_df.sort_values("pred",ascending=False)