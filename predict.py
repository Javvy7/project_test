# predict.py
import joblib
import pandas as pd

from javidan_nuriyev_eda import load_data
from javidan_nuriyev_fe1 import feature_engineering1
from javidan_nuriyev_fe2 import feature_engineering2


def main():
    # Yeni data yüklə
    df = load_data("new_data.csv")

    # Feature Engineering
    df = feature_engineering1(df)
    df = feature_engineering2(df)

    # Modeli yüklə
    model = joblib.load("model.joblib")

    # Prediction
    predictions = model.predict(df)
    df["prediction"] = predictions
    df.to_csv("predictions.csv", index=False)
    print("Predictions saved to predictions.csv")


if __name__ == "__main__":
    main()
