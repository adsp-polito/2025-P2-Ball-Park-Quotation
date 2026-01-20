"""
FPT Cost Brain 2.0 - Cost Prediction Model
Ensemble model wrapper with uncertainty estimation
"""

import json
import pickle
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class ModelVersion:
    """Model version metadata."""

    version_id: str
    model_type: str
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    metrics: dict[str, float] = field(default_factory=dict)
    feature_names: list[str] = field(default_factory=list)
    is_active: bool = False
    training_samples: int = 0
    description: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "version_id": self.version_id,
            "model_type": self.model_type,
            "created_at": self.created_at,
            "metrics": self.metrics,
            "feature_names": self.feature_names,
            "is_active": self.is_active,
            "training_samples": self.training_samples,
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ModelVersion":
        """Create from dictionary."""
        return cls(**data)


@dataclass
class PredictionResult:
    """Result from model prediction."""

    predicted_hours: float
    confidence: float
    lower_bound: float
    upper_bound: float
    method: str
    breakdown: dict[str, float] = field(default_factory=dict)
    reasoning: str = ""


class CostPredictionModel:
    """
    Ensemble cost prediction model.

    Combines multiple prediction methods:
    1. Gradient Boosting (primary)
    2. Neural Network (secondary)
    3. Similar PRs average (fallback)

    Provides uncertainty estimation via:
    - Prediction intervals from ensemble disagreement
    - Confidence scoring based on feature coverage
    """

    def __init__(
        self,
        model_path: Path | str | None = None,
        version: ModelVersion | None = None,
    ):
        self.model_path = Path(model_path) if model_path else None
        self.version = version
        self._models: dict[str, Any] = {}
        self._is_loaded = False

    def load(self, path: Path | str | None = None) -> bool:
        """
        Load model from disk.

        Args:
            path: Optional path override

        Returns:
            True if loaded successfully
        """
        load_path = Path(path) if path else self.model_path

        if not load_path or not load_path.exists():
            return False

        try:
            with open(load_path, "rb") as f:
                data = pickle.load(f)

            self._models = data.get("models", {})
            self.version = ModelVersion.from_dict(data.get("version", {}))
            self._is_loaded = True
            return True

        except Exception:
            return False

    def save(self, path: Path | str | None = None) -> bool:
        """
        Save model to disk.

        Args:
            path: Optional path override

        Returns:
            True if saved successfully
        """
        save_path = Path(path) if path else self.model_path

        if not save_path:
            return False

        try:
            save_path.parent.mkdir(parents=True, exist_ok=True)

            data = {
                "models": self._models,
                "version": self.version.to_dict() if self.version else {},
            }

            with open(save_path, "wb") as f:
                pickle.dump(data, f)

            return True

        except Exception:
            return False

    def predict(
        self,
        features: np.ndarray,
        similar_prs: list[dict[str, Any]] | None = None,
    ) -> PredictionResult:
        """
        Make prediction with uncertainty estimation.

        Args:
            features: Feature vector or matrix
            similar_prs: Optional similar PRs for fallback

        Returns:
            PredictionResult with prediction and confidence
        """
        similar_prs = similar_prs or []

        # Ensure 2D input
        if features.ndim == 1:
            features = features.reshape(1, -1)

        predictions = []
        methods_used = []

        # Method 1: Gradient Boosting
        if "gradient_boosting" in self._models:
            try:
                gb_pred = self._models["gradient_boosting"].predict(features)[0]
                predictions.append(gb_pred)
                methods_used.append("gradient_boosting")
            except Exception:
                pass

        # Method 2: Neural Network
        if "neural_network" in self._models:
            try:
                nn_pred = self._models["neural_network"].predict(features)[0]
                predictions.append(nn_pred)
                methods_used.append("neural_network")
            except Exception:
                pass

        # Method 3: Random Forest
        if "random_forest" in self._models:
            try:
                rf_pred = self._models["random_forest"].predict(features)[0]
                predictions.append(rf_pred)
                methods_used.append("random_forest")
            except Exception:
                pass

        # Fallback: Similar PRs average
        if not predictions and similar_prs:
            avg_hours = sum(sp.get("total_hours", 0) for sp in similar_prs) / len(
                similar_prs
            )
            predictions.append(avg_hours)
            methods_used.append("similar_prs_average")

        # Fallback: Default estimate
        if not predictions:
            predictions.append(500.0)  # Default 500 hours
            methods_used.append("default")

        # Calculate ensemble prediction
        predicted_hours = float(np.mean(predictions))

        # Calculate uncertainty from ensemble disagreement
        if len(predictions) > 1:
            std = float(np.std(predictions))
            confidence = max(0.3, 1.0 - (std / predicted_hours))
        else:
            confidence = 0.5  # Lower confidence for single model

        # Calculate prediction interval (95%)
        margin = predicted_hours * (1 - confidence) * 2
        lower_bound = max(0, predicted_hours - margin)
        upper_bound = predicted_hours + margin

        return PredictionResult(
            predicted_hours=predicted_hours,
            confidence=confidence,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            method="+".join(methods_used),
            breakdown={method: pred for method, pred in zip(methods_used, predictions)},
            reasoning=f"Ensemble of {len(methods_used)} methods",
        )

    def predict_with_breakdown(
        self,
        features: np.ndarray,
        activity_codes: list[str],
        similar_prs: list[dict[str, Any]] | None = None,
    ) -> tuple[PredictionResult, list[dict[str, Any]]]:
        """
        Predict total and distribute to activities.

        Args:
            features: Feature vector
            activity_codes: List of activity codes to distribute to
            similar_prs: Optional similar PRs

        Returns:
            (total_prediction, activity_breakdown)
        """
        total = self.predict(features, similar_prs)

        # Default distribution weights
        default_weights = {
            "REQ": 0.10,
            "DES": 0.20,
            "DEV": 0.35,
            "TEST": 0.20,
            "DOC": 0.08,
            "PM": 0.07,
        }

        # Calculate distribution
        breakdown = []
        remaining = total.predicted_hours
        used_weight = 0.0

        for code in activity_codes[:-1]:  # All but last
            weight = default_weights.get(code, 0.1)
            hours = total.predicted_hours * weight
            breakdown.append(
                {
                    "activity_code": code,
                    "hours": hours,
                    "weight": weight,
                }
            )
            remaining -= hours
            used_weight += weight

        # Last activity gets remainder
        if activity_codes:
            breakdown.append(
                {
                    "activity_code": activity_codes[-1],
                    "hours": max(0, remaining),
                    "weight": 1.0 - used_weight,
                }
            )

        return total, breakdown

    def set_model(self, name: str, model: Any) -> None:
        """Set a model in the ensemble."""
        self._models[name] = model
        self._is_loaded = True

    def get_model(self, name: str) -> Any | None:
        """Get a model from the ensemble."""
        return self._models.get(name)

    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._is_loaded and len(self._models) > 0


class ModelRegistry:
    """
    Registry for managing multiple model versions.

    Supports:
    - Version tracking
    - A/B testing
    - Shadow evaluation
    - Rollback
    """

    def __init__(self, models_dir: Path | str):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self._active_model: CostPredictionModel | None = None
        self._versions: dict[str, ModelVersion] = {}

    def register(
        self,
        model: CostPredictionModel,
        version_id: str,
        metrics: dict[str, float],
        description: str = "",
    ) -> ModelVersion:
        """
        Register a new model version.

        Args:
            model: The model to register
            version_id: Unique version identifier
            metrics: Training/validation metrics
            description: Version description

        Returns:
            The created ModelVersion
        """
        version = ModelVersion(
            version_id=version_id,
            model_type="ensemble",
            metrics=metrics,
            feature_names=model.version.feature_names if model.version else [],
            training_samples=model.version.training_samples if model.version else 0,
            description=description,
        )

        model.version = version
        model_path = self.models_dir / f"model_{version_id}.pkl"
        model.save(model_path)

        self._versions[version_id] = version
        self._save_registry()

        return version

    def activate(self, version_id: str) -> bool:
        """
        Activate a model version.

        Args:
            version_id: Version to activate

        Returns:
            True if activated successfully
        """
        if version_id not in self._versions:
            return False

        # Deactivate current
        for v in self._versions.values():
            v.is_active = False

        # Activate new
        self._versions[version_id].is_active = True

        # Load the model
        model_path = self.models_dir / f"model_{version_id}.pkl"
        self._active_model = CostPredictionModel(model_path=model_path)
        self._active_model.load()

        self._save_registry()
        return True

    def get_active(self) -> CostPredictionModel | None:
        """Get the active model."""
        if self._active_model and self._active_model.is_loaded:
            return self._active_model

        # Find active version
        for version_id, version in self._versions.items():
            if version.is_active:
                model_path = self.models_dir / f"model_{version_id}.pkl"
                self._active_model = CostPredictionModel(model_path=model_path)
                self._active_model.load()
                return self._active_model

        return None

    def list_versions(self) -> list[ModelVersion]:
        """List all versions."""
        return list(self._versions.values())

    def compare(
        self,
        version_a: str,
        version_b: str,
        test_features: np.ndarray,
        test_targets: np.ndarray,
    ) -> dict[str, Any]:
        """
        Compare two model versions.

        Args:
            version_a: First version ID
            version_b: Second version ID
            test_features: Test feature matrix
            test_targets: Test target values

        Returns:
            Comparison results
        """
        results = {}

        for version_id in [version_a, version_b]:
            model_path = self.models_dir / f"model_{version_id}.pkl"
            model = CostPredictionModel(model_path=model_path)

            if not model.load():
                results[version_id] = {"error": "Failed to load"}
                continue

            predictions = []
            for i in range(len(test_features)):
                pred = model.predict(test_features[i])
                predictions.append(pred.predicted_hours)

            predictions = np.array(predictions)
            errors = np.abs(predictions - test_targets)

            results[version_id] = {
                "mae": float(np.mean(errors)),
                "median_error": float(np.median(errors)),
                "mape": float(np.mean(errors / test_targets) * 100),
                "within_30pct": float(np.mean(errors / test_targets < 0.3) * 100),
            }

        return results

    def _save_registry(self) -> None:
        """Save registry metadata."""
        registry_path = self.models_dir / "registry.json"
        data = {vid: v.to_dict() for vid, v in self._versions.items()}

        with open(registry_path, "w") as f:
            json.dump(data, f, indent=2)

    def _load_registry(self) -> None:
        """Load registry metadata."""
        registry_path = self.models_dir / "registry.json"

        if registry_path.exists():
            with open(registry_path) as f:
                data = json.load(f)

            self._versions = {vid: ModelVersion.from_dict(v) for vid, v in data.items()}
