# train.py
import joblib

from javidan_nuriyev_eda import load_data
from javidan_nuriyev_fe1 import feature_engineering1
from javidan_nuriyev_fe2 import feature_engineering2
from javidan_nuriyev_trainmodel import train_model


def main():
    # 1. Data yüklə
    df = load_data("data.csv")

    # 2. Feature Engineering
    df = feature_engineering1(df)
    df = feature_engineering2(df)

    # 3. Model Training
    model = train_model(df)

    # 4. Modeli saxla
    joblib.dump(model, "model.joblib")
    print("Model saved as model.joblib")


if __name__ == "__main__":
    main()
