import pandas as pd
import os

from src.preprocessing import preprocess_data
from src.feature_engineering import feature_engineering, save_featured_data


def run_pipeline():
    print("=== 데이터 파이프라인 시작 ===")

    # 1. 데이터 로드
    df = pd.read_csv("data/raw/smoking_health_data.csv")
    print("데이터 로드 완료")

    # 2. 전처리
    df = preprocess_data(df)
    print("전처리 완료")

    # 3. feature engineering
    df = feature_engineering(df)
    print("feature engineering 완료")

    # 4. 저장
    os.makedirs("data/processed", exist_ok=True)
    save_featured_data(df, "data/processed/smoking_processed.csv")
    print("데이터 저장 완료")

    print("=== 파이프라인 종료 ===")


if __name__ == "__main__":
    run_pipeline()