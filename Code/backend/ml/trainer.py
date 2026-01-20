"""
FPT Cost Brain 2.0 - Model Trainer
Batch retraining and shadow evaluation
"""

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from ml.features import FeatureExtractor, FeatureSet
from ml.model import CostPredictionModel, ModelRegistry, ModelVersion


@dataclass
class TrainingConfig:
    """Configuration for model training."""

    # Data settings
    min_samples: int = 50
    validation_split: float = 0.2
    test_split: float = 0.1

    # Model settings
    use_gradient_boosting: bool = True
    use_neural_network: bool = True
    use_random_forest: bool = True

    # Training parameters
    gb_n_estimators: int = 100
    gb_max_depth: int = 6
    gb_learning_rate: float = 0.1

    rf_n_estimators: int = 100
    rf_max_depth: int = 10

    nn_hidden_sizes: list[int] = field(default_factory=lambda: [128, 64, 32])
    nn_epochs: int = 100
    nn_learning_rate: float = 0.001

    # Evaluation thresholds
    min_r2_improvement: float = 0.05
    max_mape: float = 35.0  # Maximum acceptable MAPE


@dataclass
class TrainingResult:
    """Result from model training."""

    success: bool
    version_id: str
    metrics: dict[str, float]
    training_samples: int
    validation_samples: int
    training_time_seconds: float
    improvement_vs_baseline: float | None = None
    auto_promoted: bool = False
    error_message: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "success": self.success,
            "version_id": self.version_id,
            "metrics": self.metrics,
            "training_samples": self.training_samples,
            "validation_samples": self.validation_samples,
            "training_time_seconds": self.training_time_seconds,
            "improvement_vs_baseline": self.improvement_vs_baseline,
            "auto_promoted": self.auto_promoted,
            "error_message": self.error_message,
        }


class ModelTrainer:
    """
    Train and evaluate cost prediction models.

    Handles:
    - Feature preparation
    - Model training (ensemble)
    - Cross-validation
    - Shadow evaluation
    - Auto-promotion
    """

    def __init__(
        self,
        models_dir: Path | str,
        config: TrainingConfig | None = None,
    ):
        self.models_dir = Path(models_dir)
        self.config = config or TrainingConfig()
        self.feature_extractor = FeatureExtractor()
        self.registry = ModelRegistry(models_dir)

    async def train(
        self,
        training_data: list[dict[str, Any]],
        feedback_data: list[dict[str, Any]] | None = None,
    ) -> TrainingResult:
        """
        Train a new model version.

        Args:
            training_data: List of historical PR data with hours
            feedback_data: Optional feedback corrections to incorporate

        Returns:
            TrainingResult with metrics and status
        """
        import time

        start_time = time.time()
        version_id = (
            f"v_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        )

        try:
            # Step 1: Prepare features
            feature_sets, targets = self._prepare_data(training_data, feedback_data)

            if len(feature_sets) < self.config.min_samples:
                return TrainingResult(
                    success=False,
                    version_id=version_id,
                    metrics={},
                    training_samples=len(feature_sets),
                    validation_samples=0,
                    training_time_seconds=time.time() - start_time,
                    error_message=f"Insufficient samples: {len(feature_sets)} < {self.config.min_samples}",
                )

            # Step 2: Split data
            X, y = self.feature_extractor.prepare_training_data(feature_sets, targets)
            X_train, X_val, X_test, y_train, y_val, y_test = self._split_data(X, y)

            # Step 3: Normalize features
            X_train = self.feature_extractor.normalize_features(X_train, fit=True)
            X_val = self.feature_extractor.normalize_features(X_val)
            X_test = self.feature_extractor.normalize_features(X_test)

            # Step 4: Train models
            model = CostPredictionModel()
            model.version = ModelVersion(
                version_id=version_id,
                model_type="ensemble",
                feature_names=self.feature_extractor.get_feature_names(),
                training_samples=len(X_train),
            )

            if self.config.use_gradient_boosting:
                gb_model = self._train_gradient_boosting(X_train, y_train)
                model.set_model("gradient_boosting", gb_model)

            if self.config.use_random_forest:
                rf_model = self._train_random_forest(X_train, y_train)
                model.set_model("random_forest", rf_model)

            # Note: Neural network requires PyTorch, optional
            if self.config.use_neural_network:
                try:
                    nn_model = self._train_neural_network(
                        X_train, y_train, X_val, y_val
                    )
                    model.set_model("neural_network", nn_model)
                except ImportError:
                    pass  # PyTorch not available

            # Step 5: Evaluate
            metrics = self._evaluate(model, X_val, y_val)
            test_metrics = self._evaluate(model, X_test, y_test)
            metrics["test_mae"] = test_metrics["mae"]
            metrics["test_mape"] = test_metrics["mape"]

            # Step 6: Register model
            self.registry.register(
                model=model,
                version_id=version_id,
                metrics=metrics,
                description=f"Trained on {len(X_train)} samples",
            )

            # Step 7: Check for auto-promotion
            improvement = None
            auto_promoted = False

            current = self.registry.get_active()
            if current and current.version:
                baseline_metrics = current.version.metrics
                if baseline_metrics.get("mape"):
                    improvement = baseline_metrics["mape"] - metrics["mape"]

                    # Auto-promote if significant improvement
                    if improvement >= self.config.min_r2_improvement * 100:
                        self.registry.activate(version_id)
                        auto_promoted = True
            else:
                # No current model, activate this one
                self.registry.activate(version_id)
                auto_promoted = True

            return TrainingResult(
                success=True,
                version_id=version_id,
                metrics=metrics,
                training_samples=len(X_train),
                validation_samples=len(X_val),
                training_time_seconds=time.time() - start_time,
                improvement_vs_baseline=improvement,
                auto_promoted=auto_promoted,
            )

        except Exception as e:
            return TrainingResult(
                success=False,
                version_id=version_id,
                metrics={},
                training_samples=0,
                validation_samples=0,
                training_time_seconds=time.time() - start_time,
                error_message=str(e),
            )

    def _prepare_data(
        self,
        training_data: list[dict[str, Any]],
        feedback_data: list[dict[str, Any]] | None = None,
    ) -> tuple[list[FeatureSet], list[float]]:
        """Prepare feature sets and targets from training data."""
        feature_sets = []
        targets = []

        for item in training_data:
            try:
                parsed_pr = item.get("parsed_pr", item)
                answers = item.get("answers", {})
                similar_prs = item.get("similar_prs", [])
                embedding = item.get("embedding")

                feature_set = self.feature_extractor.extract(
                    parsed_pr=parsed_pr,
                    answers=answers,
                    similar_prs=similar_prs,
                    embedding=embedding,
                )

                target = item.get("total_hours", item.get("actual_hours"))
                if target is not None and target > 0:
                    feature_sets.append(feature_set)
                    targets.append(float(target))

            except Exception:
                continue  # Skip invalid items

        # Incorporate feedback corrections
        if feedback_data:
            for feedback in feedback_data:
                try:
                    # Adjust training data based on corrections
                    corrected_hours = feedback.get("corrected_value")
                    original_item = feedback.get("original_item", {})

                    if corrected_hours and original_item:
                        feature_set = self.feature_extractor.extract(
                            parsed_pr=original_item,
                        )
                        feature_sets.append(feature_set)
                        targets.append(float(corrected_hours))

                except Exception:
                    continue

        return feature_sets, targets

    def _split_data(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Split data into train/validation/test sets."""
        n = len(X)
        indices = np.random.permutation(n)

        test_size = int(n * self.config.test_split)
        val_size = int(n * self.config.validation_split)

        test_idx = indices[:test_size]
        val_idx = indices[test_size : test_size + val_size]
        train_idx = indices[test_size + val_size :]

        return (
            X[train_idx],
            X[val_idx],
            X[test_idx],
            y[train_idx],
            y[val_idx],
            y[test_idx],
        )

    def _train_gradient_boosting(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> Any:
        """Train Gradient Boosting model."""
        from sklearn.ensemble import GradientBoostingRegressor

        model = GradientBoostingRegressor(
            n_estimators=self.config.gb_n_estimators,
            max_depth=self.config.gb_max_depth,
            learning_rate=self.config.gb_learning_rate,
            random_state=42,
        )
        model.fit(X, y)
        return model

    def _train_random_forest(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> Any:
        """Train Random Forest model."""
        from sklearn.ensemble import RandomForestRegressor

        model = RandomForestRegressor(
            n_estimators=self.config.rf_n_estimators,
            max_depth=self.config.rf_max_depth,
            random_state=42,
            n_jobs=-1,
        )
        model.fit(X, y)
        return model

    def _train_neural_network(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> Any:
        """Train Neural Network model using sklearn MLPRegressor."""
        from sklearn.neural_network import MLPRegressor

        model = MLPRegressor(
            hidden_layer_sizes=tuple(self.config.nn_hidden_sizes),
            max_iter=self.config.nn_epochs,
            learning_rate_init=self.config.nn_learning_rate,
            early_stopping=True,
            validation_fraction=0.1,
            random_state=42,
        )
        model.fit(X_train, y_train)
        return model

    def _evaluate(
        self,
        model: CostPredictionModel,
        X: np.ndarray,
        y: np.ndarray,
    ) -> dict[str, float]:
        """Evaluate model on data."""
        predictions = []

        for i in range(len(X)):
            pred = model.predict(X[i])
            predictions.append(pred.predicted_hours)

        predictions = np.array(predictions)
        errors = np.abs(predictions - y)

        # Calculate metrics
        mae = float(np.mean(errors))
        median_error = float(np.median(errors))

        # Guard against division by zero in MAPE calculation
        # Filter out zero actuals to avoid inf/nan
        nonzero_mask = y != 0
        if np.any(nonzero_mask):
            mape = float(np.mean(errors[nonzero_mask] / y[nonzero_mask]) * 100)
            within_30pct = float(
                np.mean(errors[nonzero_mask] / y[nonzero_mask] < 0.3) * 100
            )
        else:
            mape = float("inf")
            within_30pct = 0.0

        # R² score
        ss_res = np.sum((y - predictions) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

        return {
            "mae": mae,
            "median_error": median_error,
            "mape": mape,
            "r2": float(r2),
            "within_30pct": within_30pct,
        }

    async def shadow_evaluate(
        self,
        candidate_version: str,
        test_prs: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """
        Shadow evaluate a candidate model against production.

        Args:
            candidate_version: Version ID of candidate model
            test_prs: List of test PRs with actual hours

        Returns:
            Comparison results
        """
        # Prepare test data
        feature_sets, targets = self._prepare_data(test_prs, None)
        X, y = self.feature_extractor.prepare_training_data(feature_sets, targets)
        X = self.feature_extractor.normalize_features(X)

        # Get current production model
        production = self.registry.get_active()

        # Load candidate
        candidate_path = self.models_dir / f"model_{candidate_version}.pkl"
        candidate = CostPredictionModel(model_path=candidate_path)
        candidate.load()

        results = {
            "test_samples": len(X),
            "production": {},
            "candidate": {},
        }

        if production:
            results["production"] = self._evaluate(production, X, y)

        results["candidate"] = self._evaluate(candidate, X, y)

        # Calculate improvement
        if results["production"].get("mape") and results["candidate"].get("mape"):
            improvement = results["production"]["mape"] - results["candidate"]["mape"]
            results["improvement_mape"] = improvement
            results["should_promote"] = (
                improvement >= self.config.min_r2_improvement * 100
            )

        return results

    async def promote_if_better(
        self,
        candidate_version: str,
        test_prs: list[dict[str, Any]],
    ) -> bool:
        """
        Promote candidate if it performs better on test data.

        Args:
            candidate_version: Version to evaluate
            test_prs: Test data

        Returns:
            True if promoted
        """
        results = await self.shadow_evaluate(candidate_version, test_prs)

        if results.get("should_promote", False):
            self.registry.activate(candidate_version)
            return True

        return False
