import pandas as pd
import numpy as np

# =========================
# 1. 결측치 처리
# =========================
def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # 시력 → 최빈값
    if "시력" in df.columns:
        df["시력"] = df["시력"].fillna(df["시력"].mode()[0])

    # 나머지 주요 수치형 → 중앙값
    for col in ["공복 혈당", "혈압", "중성 지방"]:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    return df


# =========================
# 2. 도메인 기반 이상치 처리
# =========================
def handle_domain_outliers(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # 시력
    if "시력" in df.columns:
        df["시력"] = df["시력"].clip(0, 2.5)

    # 혈압
    if "혈압" in df.columns:
        df.loc[df["혈압"] < 30, "혈압"] = np.nan
        df["혈압"] = df["혈압"].fillna(df["혈압"].median())

    # BMI
    if "BMI" in df.columns:
        df.loc[df["BMI"] < 15, "BMI"] = np.nan
        df.loc[df["BMI"] > 45, "BMI"] = np.nan
        df["BMI"] = df["BMI"].fillna(df["BMI"].median())

    # 몸무게
    if "몸무게(kg)" in df.columns:
        df.loc[df["몸무게(kg)"] < 40, "몸무게(kg)"] = np.nan
        df["몸무게(kg)"] = df["몸무게(kg)"].fillna(df["몸무게(kg)"].median())

    return df


# =========================
# 3. IQR 기반 이상치 처리
# =========================
def clip_outliers(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    df = df.copy()

    for col in cols:
        if col not in df.columns:
            continue

        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1

        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR

        df[col] = df[col].clip(lower, upper)

    return df


# 4. 로그 변환 (먼저 실행)
# =========================
def log_transform(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "중성 지방" in df.columns:
        df["중성 지방_log"] = np.log1p(df["중성 지방"].clip(lower=0) + 1e-9)

    return df


# =========================
# 5. 수치 정리
# =========================
def clean_numeric_format(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # 시력
    if "시력" in df.columns:
        df["시력"] = df["시력"].round(2)
    
    # 혈청 크레아티닌
    if "혈청 크레아티닌" in df.columns:
        df["혈청 크레아티닌"] = df["혈청 크레아티닌"].round(2)
    
    # log 변환 값 (분석용 → 소수 3자리로 설정)
    if "중성 지방_log" in df.columns:
        df["중성 지방_log"] = df["중성 지방_log"].round(3)

    return df


# =========================
# 6. 전체 파이프라인
# =========================
def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df = handle_missing_values(df)
    df = handle_domain_outliers(df)

    # 통계 기반 이상치 처리
    outlier_cols = ["혈청 크레아티닌", "저밀도지단백", "헤모글로빈"]
    df = clip_outliers(df, outlier_cols)

    df = log_transform(df)
    df = clean_numeric_format(df)

    return df


# =========================
# 7. 저장
# =========================
def save_processed_data(df: pd.DataFrame, path: str):
    df.to_csv(path, index=False)