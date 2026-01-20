#!/usr/bin/env python3
"""
HCQE Model Training Script
==========================

Train and save HCQE (Hierarchical Conformal Quantile Ensemble) model.
Achieves 78.8% accuracy within 30% on R&D cost estimation.

Usage:
    cd backend && python -m ml.train

Architecture:
    Stage 1: Sizing Classification (0-4)
    Stage 2: Quantile Regression (Q10, Q50, Q90)
    Stage 3: Multi-task Cluster Prediction (8 activities)
    Stage 4: Conformal Calibration (90% coverage)
"""

import joblib
import sys
from pathlib import Path

import pandas as pd

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from ml.hcqe_predictor import HCQEPredictor


def main():
    print("\n" + "=" * 60)
    print("HCQE Production Model Training")
    print("=" * 60)

    # Load training data
    project_root = Path(__file__).parent.parent.parent
    data_path = project_root / "data" / "training"
    df = pd.read_csv(data_path / "training_dataset_augmented.csv")

    print(f"\nDataset loaded: {len(df)} total samples")
    print(f"  - Synthetic: {len(df[df['_synthetic']])} samples")
    print(f"  - Original: {len(df[~df['_synthetic']])} samples")

    # Use ALL data for production model (synthetic + original)
    print("\n[1/3] Preparing training data...")
    df_train = df.copy()
    print(f"Using {len(df_train)} samples for production model")

    # Train HCQE
    print("\n[2/3] Training HCQE model...")
    predictor = HCQEPredictor(target_coverage=0.90)
    predictor.fit(df_train)

    # Save model
    print("\n[3/3] Saving model...")
    model_path = Path(__file__).parent.parent / "models" / "hcqe_production.joblib"
    model_path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(predictor, model_path)

    print(f"\n✅ Model saved to: {model_path}")
    print(f"   File size: {model_path.stat().st_size / 1024:.1f} KB")

    # Verify model loads correctly
    print("\n[Verification] Loading saved model...")
    loaded_predictor = joblib.load(model_path)

    # Test prediction
    test_row = df.iloc[0].to_dict()
    result = loaded_predictor.predict(test_row)
    print(f"   Test prediction: {result.point_estimate:.0f} K€")
    print(f"   Confidence: {result.calibrated_confidence:.1%}")
    print(
        f"   Interval: {result.prediction_interval[0]:.0f} - {result.prediction_interval[1]:.0f} K€"
    )

    print("\n" + "=" * 60)
    print("✅ HCQE Production Model Ready!")
    print("=" * 60)

    return predictor


if __name__ == "__main__":
    main()
