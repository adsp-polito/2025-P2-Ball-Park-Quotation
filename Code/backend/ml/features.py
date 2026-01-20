"""
FPT Cost Brain 2.0 - Feature Engineering
Extract and transform features for ML prediction
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np


class FeatureType(str, Enum):
    """Types of features for the model."""

    NUMERIC = "numeric"
    CATEGORICAL = "categorical"
    TEXT = "text"
    EMBEDDING = "embedding"
    BINARY = "binary"


@dataclass
class FeatureDefinition:
    """Definition of a single feature."""

    name: str
    feature_type: FeatureType
    required: bool = False
    default_value: Any = None
    categories: list[str] | None = None  # For categorical features
    embedding_dim: int | None = None  # For embedding features


@dataclass
class FeatureSet:
    """Container for extracted features."""

    numeric_features: dict[str, float] = field(default_factory=dict)
    categorical_features: dict[str, str] = field(default_factory=dict)
    text_features: dict[str, str] = field(default_factory=dict)
    embedding_features: dict[str, list[float]] = field(default_factory=dict)
    binary_features: dict[str, bool] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_vector(self, feature_order: list[str]) -> np.ndarray:
        """Convert features to numpy vector for model input."""
        values = []

        for feature_name in feature_order:
            if feature_name in self.numeric_features:
                values.append(self.numeric_features[feature_name])
            elif feature_name in self.binary_features:
                values.append(1.0 if self.binary_features[feature_name] else 0.0)
            elif feature_name in self.embedding_features:
                values.extend(self.embedding_features[feature_name])
            else:
                values.append(0.0)  # Default for missing features

        return np.array(values, dtype=np.float32)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "numeric": self.numeric_features,
            "categorical": self.categorical_features,
            "text": self.text_features,
            "binary": self.binary_features,
            "metadata": self.metadata,
        }


# Standard feature definitions for cost estimation
STANDARD_FEATURES: list[FeatureDefinition] = [
    # Numeric features
    FeatureDefinition(
        name="activity_count",
        feature_type=FeatureType.NUMERIC,
        required=True,
        default_value=0,
    ),
    FeatureDefinition(
        name="mentioned_hours",
        feature_type=FeatureType.NUMERIC,
        default_value=0,
    ),
    FeatureDefinition(
        name="similar_prs_count",
        feature_type=FeatureType.NUMERIC,
        default_value=0,
    ),
    FeatureDefinition(
        name="similar_avg_hours",
        feature_type=FeatureType.NUMERIC,
        default_value=0,
    ),
    FeatureDefinition(
        name="complexity_score",
        feature_type=FeatureType.NUMERIC,
        default_value=0.5,
    ),
    # Categorical features
    FeatureDefinition(
        name="program_family",
        feature_type=FeatureType.CATEGORICAL,
        categories=["A320", "A330", "A350", "A380", "A220", "Unknown"],
    ),
    FeatureDefinition(
        name="program_size",
        feature_type=FeatureType.CATEGORICAL,
        categories=["small", "medium", "large", "xl"],
    ),
    FeatureDefinition(
        name="hw_sw_split",
        feature_type=FeatureType.CATEGORICAL,
        categories=["HW", "SW", "Mixed"],
    ),
    FeatureDefinition(
        name="risk_level",
        feature_type=FeatureType.CATEGORICAL,
        categories=["low", "medium", "high"],
    ),
    # Binary features
    FeatureDefinition(
        name="has_program_family",
        feature_type=FeatureType.BINARY,
        default_value=False,
    ),
    FeatureDefinition(
        name="has_customer",
        feature_type=FeatureType.BINARY,
        default_value=False,
    ),
    FeatureDefinition(
        name="has_timeline_pressure",
        feature_type=FeatureType.BINARY,
        default_value=False,
    ),
    # Embedding features
    FeatureDefinition(
        name="pr_embedding",
        feature_type=FeatureType.EMBEDDING,
        embedding_dim=3072,
    ),
]


class FeatureExtractor:
    """
    Extract features from PR data for ML prediction.

    Handles:
    - Numeric feature extraction and normalization
    - Categorical encoding (one-hot)
    - Text feature extraction
    - Embedding generation
    """

    def __init__(
        self,
        feature_definitions: list[FeatureDefinition] | None = None,
    ):
        self.features = feature_definitions or STANDARD_FEATURES
        self._feature_map = {f.name: f for f in self.features}
        self._category_encoders: dict[str, dict[str, int]] = {}

        # Build category encoders
        for feat in self.features:
            if feat.feature_type == FeatureType.CATEGORICAL and feat.categories:
                self._category_encoders[feat.name] = {
                    cat: i for i, cat in enumerate(feat.categories)
                }

    def extract(
        self,
        parsed_pr: dict[str, Any],
        answers: dict[str, str] | None = None,
        similar_prs: list[dict[str, Any]] | None = None,
        embedding: list[float] | None = None,
    ) -> FeatureSet:
        """
        Extract all features from PR data.

        Args:
            parsed_pr: Parsed PR document data
            answers: Q&A answers if available
            similar_prs: List of similar historical PRs
            embedding: Pre-computed PR embedding

        Returns:
            FeatureSet with all extracted features
        """
        answers = answers or {}
        similar_prs = similar_prs or []

        feature_set = FeatureSet()

        # Extract activity-based features
        activities = parsed_pr.get("raw_activities", [])
        feature_set.numeric_features["activity_count"] = len(activities)

        total_mentioned = sum(
            float(a.get("hours", 0)) for a in activities if a.get("hours")
        )
        feature_set.numeric_features["mentioned_hours"] = total_mentioned

        # Similar PRs features
        feature_set.numeric_features["similar_prs_count"] = len(similar_prs)

        if similar_prs:
            avg_hours = sum(sp.get("total_hours", 0) for sp in similar_prs) / len(
                similar_prs
            )
            feature_set.numeric_features["similar_avg_hours"] = avg_hours
        else:
            feature_set.numeric_features["similar_avg_hours"] = 0

        # Categorical features
        program_family = parsed_pr.get("program_family", "Unknown")
        feature_set.categorical_features["program_family"] = program_family

        # Binary features
        feature_set.binary_features["has_program_family"] = bool(
            parsed_pr.get("program_family")
        )
        feature_set.binary_features["has_customer"] = bool(parsed_pr.get("customer"))

        # Check timeline pressure from answers
        has_timeline = any(
            "urgent" in str(a).lower() or "deadline" in str(a).lower()
            for a in answers.values()
        )
        feature_set.binary_features["has_timeline_pressure"] = has_timeline

        # Embedding features
        if embedding:
            feature_set.embedding_features["pr_embedding"] = embedding

        # Metadata
        feature_set.metadata = {
            "pr_code": parsed_pr.get("pr_code"),
            "title": parsed_pr.get("title"),
            "extraction_method": "standard",
        }

        return feature_set

    def extract_from_state(self, state: dict[str, Any]) -> FeatureSet:
        """Extract features from EstimationState."""
        parsed_pr = state.get("parsed_pr", {})
        answers = state.get("answers", {})
        similar_prs = state.get("similar_prs", [])
        embedding = state.get("embedding")

        return self.extract(
            parsed_pr=parsed_pr,
            answers=answers,
            similar_prs=similar_prs,
            embedding=embedding,
        )

    def encode_categorical(
        self,
        feature_name: str,
        value: str,
    ) -> list[float]:
        """One-hot encode a categorical feature."""
        encoder = self._category_encoders.get(feature_name)
        if not encoder:
            return [0.0]

        encoded = [0.0] * len(encoder)
        idx = encoder.get(value, encoder.get("Unknown", 0))
        encoded[idx] = 1.0
        return encoded

    def get_feature_names(self) -> list[str]:
        """Get ordered list of feature names."""
        names = []

        for feat in self.features:
            if feat.feature_type == FeatureType.CATEGORICAL and feat.categories:
                for cat in feat.categories:
                    names.append(f"{feat.name}_{cat}")
            elif feat.feature_type == FeatureType.EMBEDDING and feat.embedding_dim:
                for i in range(feat.embedding_dim):
                    names.append(f"{feat.name}_{i}")
            else:
                names.append(feat.name)

        return names

    def prepare_training_data(
        self,
        feature_sets: list[FeatureSet],
        targets: list[float],
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Prepare feature matrix and target vector for training.

        Args:
            feature_sets: List of extracted feature sets
            targets: List of target values (total hours)

        Returns:
            (X, y) tuple of numpy arrays
        """
        feature_order = self.get_feature_names()

        X = np.array([fs.to_vector(feature_order) for fs in feature_sets])
        y = np.array(targets, dtype=np.float32)

        return X, y

    def normalize_features(
        self,
        X: np.ndarray,
        fit: bool = False,
    ) -> np.ndarray:
        """
        Normalize numeric features.

        Args:
            X: Feature matrix
            fit: Whether to fit the normalizer (for training)

        Returns:
            Normalized feature matrix
        """
        if fit:
            self._feature_means = np.mean(X, axis=0)
            self._feature_stds = np.std(X, axis=0)
            self._feature_stds[self._feature_stds == 0] = 1  # Avoid division by zero

        if hasattr(self, "_feature_means") and hasattr(self, "_feature_stds"):
            return (X - self._feature_means) / self._feature_stds

        return X
