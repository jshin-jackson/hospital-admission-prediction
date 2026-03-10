"""
합성 병원 데이터 생성 스크립트.

실제 소규모 시드 데이터를 기반으로 CTGAN을 사용해 대규모 합성 데이터를 생성.
생성된 데이터는 data/synthetic/ 에 저장.

사용법:
    python -m src.data.generate [--n-samples 100000] [--output data/synthetic/hospital_data.csv]
"""

import argparse
import os
import random

import numpy as np
import pandas as pd
from sdv.metadata import Metadata
from sdv.single_table import CTGANSynthesizer

from src.data.schema import (
    COMORBIDITIES,
    DIAGNOSES,
    DIAGNOSIS_BASE_DAYS,
    GENDERS,
    TARGET,
)

SEED_DATA_PATH = "data/raw/seed_data.csv"
DEFAULT_OUTPUT = "data/synthetic/hospital_data.csv"


def build_seed_data() -> pd.DataFrame:
    """시드 데이터 생성: 각 병명 * 성별 * 기저질환 조합으로 다양한 케이스 구성."""
    random.seed(42)
    np.random.seed(42)

    rows = []
    for diagnosis in DIAGNOSES:
        base_days = DIAGNOSIS_BASE_DAYS[diagnosis]
        for gender in GENDERS:
            for comorbidity in COMORBIDITIES:
                age = random.randint(18, 85)
                num_medications = random.randint(0, 10)
                prior_admissions = random.randint(0, 5)
                lab_abnormal = random.choice(["yes", "no"])

                day_noise = np.random.normal(0, 1.0)
                comorbidity_bonus = 0 if comorbidity == "none" else 1.5
                lab_bonus = 1.0 if lab_abnormal == "yes" else 0.0
                age_factor = (age - 18) / (85 - 18) * 2.0

                days = max(
                    1,
                    round(
                        base_days
                        + comorbidity_bonus
                        + lab_bonus
                        + age_factor
                        + day_noise
                    ),
                )

                rows.append(
                    {
                        "age": age,
                        "gender": gender,
                        "diagnosis": diagnosis,
                        "comorbidity": comorbidity,
                        "num_medications": num_medications,
                        "prior_admissions": prior_admissions,
                        "lab_result_abnormal": lab_abnormal,
                        TARGET: days,
                    }
                )

    return pd.DataFrame(rows)


def generate_synthetic_data(n_samples: int, output_path: str) -> pd.DataFrame:
    """CTGAN으로 합성 데이터를 생성하고 CSV로 저장."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    os.makedirs(os.path.dirname(SEED_DATA_PATH) or ".", exist_ok=True)

    seed_df = build_seed_data()
    seed_df.to_csv(SEED_DATA_PATH, index=False)
    print(f"시드 데이터 저장: {SEED_DATA_PATH} ({len(seed_df)}건)")

    metadata = Metadata()
    metadata.detect_table_from_dataframe(table_name="hospital", data=seed_df)

    synthesizer = CTGANSynthesizer(metadata, epochs=300, verbose=True)
    synthesizer.fit(seed_df)

    synthetic_df = synthesizer.sample(n_samples)

    # 값 범위 후처리
    synthetic_df["age"] = synthetic_df["age"].clip(1, 100).round().astype(int)
    synthetic_df["num_medications"] = (
        synthetic_df["num_medications"].clip(0, 15).round().astype(int)
    )
    synthetic_df["prior_admissions"] = (
        synthetic_df["prior_admissions"].clip(0, 10).round().astype(int)
    )
    synthetic_df[TARGET] = synthetic_df[TARGET].clip(1, 60).round().astype(int)

    # 범주형 값 정제 (CTGAN이 새 값을 생성할 수 있음)
    synthetic_df["gender"] = synthetic_df["gender"].where(
        synthetic_df["gender"].isin(GENDERS), other="M"
    )
    synthetic_df["diagnosis"] = synthetic_df["diagnosis"].where(
        synthetic_df["diagnosis"].isin(DIAGNOSES), other="flu"
    )
    synthetic_df["comorbidity"] = synthetic_df["comorbidity"].where(
        synthetic_df["comorbidity"].isin(COMORBIDITIES), other="none"
    )
    synthetic_df["lab_result_abnormal"] = synthetic_df["lab_result_abnormal"].where(
        synthetic_df["lab_result_abnormal"].isin(["yes", "no"]), other="no"
    )

    synthetic_df.to_csv(output_path, index=False)
    print(f"합성 데이터 저장 완료: {output_path} ({len(synthetic_df):,}건)")
    return synthetic_df


def main() -> None:
    parser = argparse.ArgumentParser(description="합성 병원 데이터 생성")
    parser.add_argument(
        "--n-samples",
        type=int,
        default=100_000,
        help="생성할 합성 데이터 수 (기본: 100,000)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=DEFAULT_OUTPUT,
        help=f"출력 CSV 경로 (기본: {DEFAULT_OUTPUT})",
    )
    args = parser.parse_args()

    generate_synthetic_data(n_samples=args.n_samples, output_path=args.output)


if __name__ == "__main__":
    main()
