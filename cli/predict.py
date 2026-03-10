"""
통합 CLI 예측 스크립트.

사용법:
    python -m cli.predict \\
        --age 55 \\
        --gender M \\
        --diagnosis pneumonia \\
        --comorbidity hypertension \\
        --num-medications 4 \\
        --prior-admissions 2 \\
        --lab-result-abnormal yes \\
        [--model models/best_model.pkl]
"""

from __future__ import annotations

import argparse
import sys

from src.data.schema import (
    AGE_MAX,
    AGE_MIN,
    COMORBIDITIES,
    DIAGNOSES,
    GENDERS,
    NUM_MEDICATIONS_MAX,
    NUM_MEDICATIONS_MIN,
    PRIOR_ADMISSIONS_MAX,
    PRIOR_ADMISSIONS_MIN,
)
from src.features.pipeline import load_pipeline, prepare_dataframe

DEFAULT_MODEL = "models/best_model.pkl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="병원 입원일수 예측 CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--age",
        type=int,
        required=True,
        help=f"환자 나이 ({AGE_MIN}~{AGE_MAX})",
    )
    parser.add_argument(
        "--gender",
        type=str,
        required=True,
        choices=GENDERS,
        help="성별 (M: 남성, F: 여성)",
    )
    parser.add_argument(
        "--diagnosis",
        type=str,
        required=True,
        choices=DIAGNOSES,
        help=f"주요 병명: {', '.join(DIAGNOSES)}",
    )
    parser.add_argument(
        "--comorbidity",
        type=str,
        default="none",
        choices=COMORBIDITIES,
        help=f"기저질환: {', '.join(COMORBIDITIES)}",
    )
    parser.add_argument(
        "--num-medications",
        type=int,
        default=0,
        help=f"복용 약물 수 ({NUM_MEDICATIONS_MIN}~{NUM_MEDICATIONS_MAX})",
    )
    parser.add_argument(
        "--prior-admissions",
        type=int,
        default=0,
        help=f"과거 입원 횟수 ({PRIOR_ADMISSIONS_MIN}~{PRIOR_ADMISSIONS_MAX})",
    )
    parser.add_argument(
        "--lab-result-abnormal",
        type=str,
        default="no",
        choices=["yes", "no"],
        help="검사 이상 여부",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help="모델 pkl 파일 경로",
    )
    return parser.parse_args()


def validate_ranges(args: argparse.Namespace) -> None:
    errors = []
    if not (AGE_MIN <= args.age <= AGE_MAX):
        errors.append(f"나이는 {AGE_MIN}~{AGE_MAX} 범위여야 합니다. (입력: {args.age})")
    if not (NUM_MEDICATIONS_MIN <= args.num_medications <= NUM_MEDICATIONS_MAX):
        errors.append(
            f"복용 약물 수는 {NUM_MEDICATIONS_MIN}~{NUM_MEDICATIONS_MAX} 범위여야 합니다. "
            f"(입력: {args.num_medications})"
        )
    if not (PRIOR_ADMISSIONS_MIN <= args.prior_admissions <= PRIOR_ADMISSIONS_MAX):
        errors.append(
            f"과거 입원 횟수는 {PRIOR_ADMISSIONS_MIN}~{PRIOR_ADMISSIONS_MAX} 범위여야 합니다. "
            f"(입력: {args.prior_admissions})"
        )
    if errors:
        for e in errors:
            print(f"입력 오류: {e}", file=sys.stderr)
        sys.exit(1)


def main() -> None:
    args = parse_args()
    validate_ranges(args)

    try:
        pipeline = load_pipeline(args.model)
    except FileNotFoundError:
        print(
            f"모델 파일을 찾을 수 없습니다: {args.model}\n"
            "먼저 학습을 실행하세요:\n"
            "  python -m src.models.train",
            file=sys.stderr,
        )
        sys.exit(1)

    input_data = {
        "age": args.age,
        "gender": args.gender,
        "diagnosis": args.diagnosis,
        "comorbidity": args.comorbidity,
        "num_medications": args.num_medications,
        "prior_admissions": args.prior_admissions,
        "lab_result_abnormal": args.lab_result_abnormal,
    }

    df = prepare_dataframe(input_data)
    predicted = float(pipeline.predict(df)[0])
    rounded = max(1, round(predicted))

    print("\n예측 결과")
    print("-" * 30)
    print(f"  나이              : {args.age}세")
    print(f"  성별              : {'남성' if args.gender == 'M' else '여성'}")
    print(f"  병명              : {args.diagnosis}")
    print(f"  기저질환          : {args.comorbidity}")
    print(f"  복용 약물 수      : {args.num_medications}개")
    print(f"  과거 입원 횟수    : {args.prior_admissions}회")
    print(f"  검사 이상 여부    : {args.lab_result_abnormal}")
    print("-" * 30)
    print(f"  예상 입원일수     : {predicted:.1f}일 (반올림: {rounded}일)")
    print(f"  사용 모델         : {args.model}")


if __name__ == "__main__":
    main()
