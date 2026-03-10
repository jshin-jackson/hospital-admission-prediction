"""
모델 성능 평가 및 시각화.

사용법:
    python -m src.models.evaluate \
        [--model models/best_model.pkl] \
        [--data data/synthetic/hospital_data.csv] \
        [--output-dir reports/figures]
"""

from __future__ import annotations

import argparse
import os
import platform

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split

from src.data.schema import TARGET
from src.features.pipeline import load_pipeline, prepare_dataframe
from src.models.train import compute_metrics

DEFAULT_MODEL = "models/best_model.pkl"
DEFAULT_DATA = "data/synthetic/hospital_data.csv"
DEFAULT_OUTPUT_DIR = "reports/figures"


def _configure_font() -> None:
    if platform.system() == "Darwin":
        matplotlib.rc("font", family="AppleGothic")
    elif platform.system() == "Windows":
        matplotlib.rc("font", family="Malgun Gothic")
    else:
        matplotlib.rc("font", family="DejaVu Sans")
    matplotlib.rcParams["axes.unicode_minus"] = False


def plot_pred_vs_actual(
    y_test: np.ndarray,
    y_pred: np.ndarray,
    metrics: dict[str, float],
    output_path: str | None = None,
) -> None:
    """예측값 vs 실제값 산점도."""
    _configure_font()
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(x=y_test, y=y_pred, alpha=0.3, ax=ax)
    lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
    ax.plot(lims, lims, "r--", linewidth=1.5, label="완벽 예측")
    ax.set_xlabel("실제 입원일수")
    ax.set_ylabel("예측 입원일수")
    ax.set_title(
        f"입원일수 예측 결과\n"
        f"RMSE={metrics['rmse']:.2f}  MAE={metrics['mae']:.2f}  R²={metrics['r2']:.3f}"
    )
    ax.legend()
    plt.tight_layout()
    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        plt.savefig(output_path, dpi=150)
        print(f"저장: {output_path}")
    else:
        plt.show()
    plt.close(fig)


def plot_residuals(
    y_test: np.ndarray,
    y_pred: np.ndarray,
    output_path: str | None = None,
) -> None:
    """잔차 분포 히스토그램."""
    _configure_font()
    residuals = y_pred - y_test
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    sns.histplot(residuals, kde=True, ax=axes[0])
    axes[0].axvline(0, color="r", linestyle="--")
    axes[0].set_xlabel("잔차 (예측 - 실제)")
    axes[0].set_title("잔차 분포")

    sns.scatterplot(x=y_pred, y=residuals, alpha=0.3, ax=axes[1])
    axes[1].axhline(0, color="r", linestyle="--")
    axes[1].set_xlabel("예측 입원일수")
    axes[1].set_ylabel("잔차")
    axes[1].set_title("잔차 vs 예측값")

    plt.tight_layout()
    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        plt.savefig(output_path, dpi=150)
        print(f"저장: {output_path}")
    else:
        plt.show()
    plt.close(fig)


def plot_feature_importance(
    pipeline,
    output_path: str | None = None,
) -> None:
    """피처 중요도 막대 그래프 (tree 기반 모델에만 해당)."""
    _configure_font()
    model = pipeline.named_steps["model"]
    if not hasattr(model, "feature_importances_"):
        print("이 모델은 feature_importances_를 지원하지 않습니다.")
        return

    preprocessor = pipeline.named_steps["preprocessor"]
    num_names = preprocessor.transformers_[0][2]
    cat_encoder = preprocessor.transformers_[1][1].named_steps["onehot"]
    cat_names = list(cat_encoder.get_feature_names_out(preprocessor.transformers_[1][2]))
    feature_names = list(num_names) + cat_names

    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:20]
    top_indices = indices[::-1]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(
        [feature_names[i] for i in top_indices],
        importances[top_indices],
    )
    ax.set_xlabel("중요도")
    ax.set_title("피처 중요도 (상위 20개)")
    plt.tight_layout()
    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        plt.savefig(output_path, dpi=150)
        print(f"저장: {output_path}")
    else:
        plt.show()
    plt.close(fig)


def run_evaluation(
    model_path: str = DEFAULT_MODEL,
    data_path: str = DEFAULT_DATA,
    output_dir: str = DEFAULT_OUTPUT_DIR,
) -> dict[str, float]:
    pipeline = load_pipeline(model_path)
    df = pd.read_csv(data_path)
    X = prepare_dataframe(df)
    y = df[TARGET]

    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    y_pred = pipeline.predict(X_test)
    metrics = compute_metrics(y_test.to_numpy(), y_pred)

    print(f"\n평가 결과 ({model_path})")
    print(f"  RMSE : {metrics['rmse']:.3f}")
    print(f"  MAE  : {metrics['mae']:.3f}")
    print(f"  R²   : {metrics['r2']:.3f}")

    plot_pred_vs_actual(
        y_test.to_numpy(), y_pred, metrics,
        output_path=os.path.join(output_dir, "pred_vs_actual.png"),
    )
    plot_residuals(
        y_test.to_numpy(), y_pred,
        output_path=os.path.join(output_dir, "residuals.png"),
    )
    plot_feature_importance(
        pipeline,
        output_path=os.path.join(output_dir, "feature_importance.png"),
    )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="모델 성능 평가 및 시각화")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="모델 pkl 경로")
    parser.add_argument("--data", default=DEFAULT_DATA, help="데이터 CSV 경로")
    parser.add_argument(
        "--output-dir", default=DEFAULT_OUTPUT_DIR, help="그래프 저장 디렉토리"
    )
    args = parser.parse_args()
    run_evaluation(
        model_path=args.model,
        data_path=args.data,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
