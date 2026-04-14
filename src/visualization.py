import matplotlib.pyplot as plt
import seaborn as sns
import platform

system = platform.system()

if system == "Darwin":  # macOS
    plt.rc('font', family='AppleGothic')
elif system == "Windows":  # Windows
    plt.rc('font', family='Malgun Gothic')
elif system == "Linux":  # Linux
    plt.rc('font', family='DejaVu Sans')

plt.rcParams['axes.unicode_minus'] = False

# =========================
# 1. 타겟 분포
# =========================
def plot_target_distribution(df, target="label"):
    plt.figure(figsize=(6, 4))
    sns.countplot(data=df, x=target)
    plt.title("Target Distribution")
    plt.xlabel(target)
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()


# =========================
# 2. 수치형 변수 vs 타겟 (boxplot + histplot)
# =========================
def plot_numeric_vs_target(df, features, target="label"):
    n = len(features)
    rows = (n + 1) // 2

    plt.figure(figsize=(12, 5 * rows))

    for i, col in enumerate(features, 1):
        plt.subplot(rows, 2, i)

        # boxplot
        sns.boxplot(data=df, x=target, y=col, showfliers=False)

        # histplot (같이 겹쳐서 표시)
        sns.histplot(data=df, x=col, hue=target, kde=True, alpha=0.3)

        plt.title(f"{col} vs {target}")

    plt.tight_layout()
    plt.show()


# =========================
# 3. 범주형 변수 vs 타겟
# =========================
def plot_categorical_vs_target(df, col, target="label"):
    plt.figure(figsize=(6, 4))
    sns.countplot(data=df, x=col, hue=target)
    plt.title(f"{col} vs {target}")
    plt.tight_layout()
    plt.show()


# =========================
# 4. 상관관계 히트맵
# =========================
def plot_correlation_heatmap(df):
    corr = df.corr(numeric_only=True)

    plt.figure(figsize=(12, 8))
    sns.heatmap(corr, annot=True, cmap="coolwarm")
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.show()


# =========================
# 5. 결측치 시각화
# =========================
def plot_missing_values(df):
    plt.figure(figsize=(10, 6))
    sns.heatmap(df.isnull(), cbar=False)
    plt.title("Missing Value Heatmap")
    plt.tight_layout()
    plt.show()