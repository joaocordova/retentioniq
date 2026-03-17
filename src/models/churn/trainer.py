"""XGBoost churn model training for RetentionIQ.

Provides end-to-end functionality: training with early stopping,
threshold optimization, evaluation with a comprehensive metrics suite,
and batch scoring.  Every experiment is fully tracked in MLflow.
"""

from __future__ import annotations

import os
from typing import Any

import mlflow
import mlflow.xgboost
import numpy as np
import pandas as pd
import structlog
import xgboost as xgb
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    auc,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.exceptions import DataError, ModelError

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train_churn_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    config: dict[str, Any],
) -> xgb.XGBClassifier:
    """Train an XGBoost churn classifier with early stopping.

    Hyperparameters are read entirely from *config* (the ``churn`` section
    of ``configs/models.yaml``).  Class imbalance is handled via
    ``scale_pos_weight`` which is computed automatically when set to
    ``"auto"``.

    Args:
        X_train: Training feature matrix.
        y_train: Binary training labels (1 = churned).
        X_val: Validation feature matrix.
        y_val: Binary validation labels.
        config: Churn model configuration dict.

    Returns:
        A fitted ``xgb.XGBClassifier``.

    Raises:
        DataError: If inputs are empty or misaligned.
        ModelError: If training fails.
    """
    _validate_train_inputs(X_train, y_train, X_val, y_val)

    hp: dict[str, Any] = config.get("hyperparameters", {})

    # Auto-compute class weight if requested
    scale_pos_weight = hp.get("scale_pos_weight", 1)
    if scale_pos_weight == "auto":
        n_neg = int((y_train == 0).sum())
        n_pos = int((y_train == 1).sum())
        scale_pos_weight = n_neg / max(n_pos, 1)
        logger.info(
            "auto_scale_pos_weight",
            n_pos=n_pos,
            n_neg=n_neg,
            scale_pos_weight=round(scale_pos_weight, 4),
        )

    early_stopping_rounds = hp.get("early_stopping_rounds", 50)
    eval_metric = hp.get("eval_metric", "aucpr")

    model = xgb.XGBClassifier(
        max_depth=hp.get("max_depth", 6),
        learning_rate=hp.get("learning_rate", 0.05),
        n_estimators=hp.get("n_estimators", 500),
        subsample=hp.get("subsample", 0.8),
        colsample_bytree=hp.get("colsample_bytree", 0.8),
        scale_pos_weight=scale_pos_weight,
        eval_metric=eval_metric,
        early_stopping_rounds=early_stopping_rounds,
        use_label_encoder=False,
        random_state=42,
        n_jobs=-1,
    )

    try:
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )
    except Exception as exc:
        raise ModelError(
            f"XGBoost training failed: {exc}"
        ) from exc

    best_iteration = model.best_iteration
    logger.info(
        "churn_model_trained",
        best_iteration=best_iteration,
        n_features=X_train.shape[1],
        n_train=len(X_train),
    )

    _log_training_to_mlflow(model, config, X_train, y_train, X_val, y_val)
    return model


# ---------------------------------------------------------------------------
# Threshold optimisation
# ---------------------------------------------------------------------------


def find_optimal_threshold(
    model: xgb.XGBClassifier,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    method: str = "f1_optimal",
) -> float:
    """Find the classification threshold that optimises a given metric.

    Supported methods:
    - ``f1_optimal``: threshold maximising F1-score.
    - ``precision_at_recall``: highest precision at recall >= 0.5.

    Args:
        model: A fitted XGBoost classifier.
        X_val: Validation features.
        y_val: Validation labels.
        method: Optimisation method name.

    Returns:
        The optimal probability threshold as a float.
    """
    probas = model.predict_proba(X_val)[:, 1]

    if method == "f1_optimal":
        precision, recall, thresholds = precision_recall_curve(
            y_val, probas
        )
        # F1 = 2 * P * R / (P + R)
        with np.errstate(divide="ignore", invalid="ignore"):
            f1_scores = np.where(
                (precision + recall) > 0,
                2 * precision * recall / (precision + recall),
                0,
            )
        # thresholds has len = len(precision) - 1
        best_idx = np.argmax(f1_scores[:-1])
        threshold = float(thresholds[best_idx])
    elif method == "precision_at_recall":
        precision, recall, thresholds = precision_recall_curve(
            y_val, probas
        )
        # Find highest precision where recall >= 0.5
        mask = recall[:-1] >= 0.5
        if mask.any():
            best_idx = np.argmax(precision[:-1] * mask)
            threshold = float(thresholds[best_idx])
        else:
            threshold = 0.5
    else:
        logger.warning(
            "unknown_threshold_method",
            method=method,
            fallback=0.5,
        )
        threshold = 0.5

    logger.info(
        "optimal_threshold_found",
        method=method,
        threshold=round(threshold, 4),
    )
    return threshold


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def evaluate_model(
    model: xgb.XGBClassifier,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Evaluate the churn model on a held-out test set.

    Computes AUC-ROC, AUC-PR, precision, recall, F1, and Brier score.
    Results are also logged to MLflow.

    Args:
        model: A fitted XGBoost classifier.
        X_test: Test feature matrix.
        y_test: Binary test labels.
        threshold: Classification threshold for binary predictions.

    Returns:
        Dict of metric names to values.
    """
    probas = model.predict_proba(X_test)[:, 1]
    preds = (probas >= threshold).astype(int)

    prec_curve, rec_curve, _ = precision_recall_curve(y_test, probas)
    auc_pr = auc(rec_curve, prec_curve)

    metrics: dict[str, float] = {
        "auc_roc": float(roc_auc_score(y_test, probas)),
        "auc_pr": float(auc_pr),
        "precision": float(precision_score(y_test, preds, zero_division=0)),
        "recall": float(recall_score(y_test, preds, zero_division=0)),
        "f1": float(f1_score(y_test, preds, zero_division=0)),
        "brier_score": float(brier_score_loss(y_test, probas)),
        "threshold": threshold,
    }

    # Log to MLflow
    _log_evaluation_to_mlflow(metrics, y_test, probas, preds)

    logger.info("churn_model_evaluated", **metrics)
    return metrics


# ---------------------------------------------------------------------------
# Batch scoring
# ---------------------------------------------------------------------------


def score_members(
    model: xgb.XGBClassifier,
    features: pd.DataFrame,
    threshold: float = 0.5,
) -> pd.DataFrame:
    """Score a batch of members and assign risk tiers.

    Risk tiers:
    - ``high``:   probability >= threshold
    - ``medium``: probability >= threshold * 0.6
    - ``low``:    probability < threshold * 0.6

    Args:
        model: A fitted XGBoost classifier.
        features: Feature matrix. If ``member_id`` is present it is
            preserved in the output; otherwise the DataFrame index is used.
        threshold: Decision threshold for the ``high`` tier.

    Returns:
        DataFrame with columns ``member_id``, ``churn_probability``,
        and ``risk_tier``.
    """
    has_member_id = "member_id" in features.columns
    if has_member_id:
        member_ids = features["member_id"].values
        scoring_features = features.drop(columns=["member_id"])
    else:
        member_ids = features.index.values
        scoring_features = features

    probas = model.predict_proba(scoring_features)[:, 1]

    medium_threshold = threshold * 0.6
    tiers = np.where(
        probas >= threshold,
        "high",
        np.where(probas >= medium_threshold, "medium", "low"),
    )

    result = pd.DataFrame(
        {
            "member_id": member_ids,
            "churn_probability": np.round(probas, 6),
            "risk_tier": tiers,
        }
    )

    tier_counts = result["risk_tier"].value_counts().to_dict()
    logger.info(
        "members_scored",
        n_members=len(result),
        tier_distribution=tier_counts,
    )
    return result


# ---------------------------------------------------------------------------
# MLflow helpers
# ---------------------------------------------------------------------------


def _log_training_to_mlflow(
    model: xgb.XGBClassifier,
    config: dict[str, Any],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
) -> None:
    """Log training run to MLflow."""
    tracking_uri = os.getenv(
        "MLFLOW_TRACKING_URI", "http://localhost:5000"
    )
    experiment_name = config.get(
        "experiment_name", "retentioniq-churn"
    )

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    hp = config.get("hyperparameters", {})

    with mlflow.start_run(run_name="xgboost_churn"):
        # Params
        for key, value in hp.items():
            mlflow.log_param(key, value)
        mlflow.log_param("n_train", len(X_train))
        mlflow.log_param("n_val", len(X_val))
        mlflow.log_param(
            "class_balance",
            round(float(y_train.mean()), 4),
        )

        # Validation metrics
        val_probas = model.predict_proba(X_val)[:, 1]
        val_auc_roc = roc_auc_score(y_val, val_probas)
        prec_c, rec_c, _ = precision_recall_curve(y_val, val_probas)
        val_auc_pr = auc(rec_c, prec_c)

        mlflow.log_metric("val_auc_roc", val_auc_roc)
        mlflow.log_metric("val_aucpr", val_auc_pr)
        mlflow.log_metric("best_iteration", model.best_iteration)

        # Log model
        mlflow.xgboost.log_model(
            model,
            artifact_path="model",
            registered_model_name="retentioniq-churn",
        )

        logger.info(
            "training_logged_to_mlflow",
            val_auc_roc=round(val_auc_roc, 4),
            val_aucpr=round(val_auc_pr, 4),
        )


def _log_evaluation_to_mlflow(
    metrics: dict[str, float],
    y_test: pd.Series,
    probas: np.ndarray,
    preds: np.ndarray,
) -> None:
    """Log evaluation metrics and artifacts to MLflow."""
    try:
        # Log metrics to the active run, or start a new one
        active_run = mlflow.active_run()
        if active_run:
            for name, value in metrics.items():
                mlflow.log_metric(f"test_{name}", value)
        else:
            tracking_uri = os.getenv(
                "MLFLOW_TRACKING_URI", "http://localhost:5000"
            )
            mlflow.set_tracking_uri(tracking_uri)
            mlflow.set_experiment("retentioniq-churn")
            with mlflow.start_run(run_name="churn_evaluation"):
                for name, value in metrics.items():
                    mlflow.log_metric(f"test_{name}", value)

                # Confusion matrix
                cm = confusion_matrix(y_test, preds)
                cm_path = "/tmp/confusion_matrix.csv"
                pd.DataFrame(
                    cm,
                    index=["actual_0", "actual_1"],
                    columns=["pred_0", "pred_1"],
                ).to_csv(cm_path)
                mlflow.log_artifact(cm_path, artifact_path="eval")

                # Calibration curve
                try:
                    frac_pos, mean_pred = calibration_curve(
                        y_test, probas, n_bins=10
                    )
                    cal_path = "/tmp/calibration.csv"
                    pd.DataFrame(
                        {
                            "mean_predicted": mean_pred,
                            "fraction_positive": frac_pos,
                        }
                    ).to_csv(cal_path, index=False)
                    mlflow.log_artifact(
                        cal_path, artifact_path="eval"
                    )
                except ValueError:
                    logger.warning(
                        "calibration_curve_skipped",
                        reason="not enough data for 10 bins",
                    )

    except Exception:
        logger.warning(
            "mlflow_eval_logging_failed",
            exc_info=True,
        )


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def _validate_train_inputs(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
) -> None:
    """Validate training inputs before fitting."""
    if X_train.empty or X_val.empty:
        raise DataError(
            "Training or validation feature matrix is empty."
        )
    if len(X_train) != len(y_train):
        raise DataError(
            f"X_train ({len(X_train)}) and y_train ({len(y_train)}) "
            f"have different lengths."
        )
    if len(X_val) != len(y_val):
        raise DataError(
            f"X_val ({len(X_val)}) and y_val ({len(y_val)}) "
            f"have different lengths."
        )
    if y_train.nunique() < 2:
        raise DataError(
            "y_train must contain both positive and negative examples. "
            f"Found only class(es): {y_train.unique().tolist()}"
        )
