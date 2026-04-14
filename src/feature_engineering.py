import pandas as pd
import numpy as np


# =========================
# 1. 파생변수 생성
# =========================
def create_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # BMI 기반 비만 여부
    if "BMI" in df.columns:
        df["비만 여부"] = (df["BMI"] >= 25).astype(int)

    # 중성지방 고위험군 (150 이상)
    if "중성 지방" in df.columns:
        df["중성지방 위험"] = (df["중성 지방"] >= 150).astype(int)

    # 공복혈당 고위험군 (100 이상)
    if "공복 혈당" in df.columns:
        df["혈당 이상"] = (df["공복 혈당"] >= 100).astype(int)

    # 헤모글로빈 낮음 여부 (성별 없으므로 단순 기준)
    if "헤모글로빈" in df.columns:
        df["빈혈 위험"] = (df["헤모글로빈"] < 13).astype(int)

    return df


# =========================
# 2. 불필요 컬럼 제거
# =========================
def drop_unused_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    drop_cols = ["ID"]

    df = df.drop(columns=[col for col in drop_cols if col in df.columns])

    return df


# =========================
# 3. 전체 feature engineering
# =========================
def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df = create_features(df)
    df = drop_unused_columns(df)

    return df


# =========================
# 4. 저장
# =========================
def save_featured_data(df: pd.DataFrame, path: str):
    df.to_csv(path, index=False)