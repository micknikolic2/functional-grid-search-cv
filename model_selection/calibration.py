"""
Probability calibration utilities for classification models.

This module provides:
    - compute_ece: Expected Calibration Error (ECE) for probabilistic classifiers
    - calibrate_classifier: comparison of isotonic vs Platt (sigmoid) calibration

The implementation assumes binary classification, i.e. `predict_proba`
returns an array of shape (n_samples, 2), and we use the probability for the
positive class (column 1).
"""

from typing import Dict, Any
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.base import ClassifierMixin
from sklearn.utils.validation import check_is_fitted


def compute_ece(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 15
) -> float:
    """
    Compute Expected Calibration Error (ECE) for binary classification.

    ECE measures the discrepancy between predicted probabilities and
    empirical accuracies across probability bins.

    Parameters:
        y_true : array-like of shape (n_samples,)
            True binary labels (0/1).

        y_prob : array-like of shape (n_samples,)
            Predicted probabilities for the positive class.

        n_bins : int, default=15
            Number of equal-width bins in [0, 1] used to group predictions.

    Returns:
        float
            The scalar ECE value. Lower is better.
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    y_prob = np.clip(y_prob, 1e-12, 1 - 1e-12)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)

    for i in range(n_bins):
        left, right = bin_edges[i], bin_edges[i + 1]
        in_bin = (y_prob >= left) & (y_prob < right) if i < n_bins - 1 else (
            (y_prob >= left) & (y_prob <= right)
        )

        if not np.any(in_bin):
            continue

        prob_bin = y_prob[in_bin]
        true_bin = y_true[in_bin]

        avg_confidence = prob_bin.mean()
        avg_accuracy = true_bin.mean()
        bin_weight = len(true_bin) / n

        ece += bin_weight * abs(avg_accuracy - avg_confidence)

    return float(ece)


def calibrate_classifier(
    base_estimator: ClassifierMixin,
    X,
    y,
    cv: int = 5,
    n_bins: int = 15
) -> Dict[str, Any]:
    """
    Calibrate a fitted classifier using isotonic regression and Platt scaling,
    and select the method with the lower ECE.

    This function:
        1. Wraps the base estimator in `CalibratedClassifierCV` with
           method="isotonic" and method="sigmoid" (Platt scaling).
        2. Fits both calibrators on the provided data (X, y).
        3. Computes ECE on the same data for both calibrated models.
        4. Returns the calibrated model with the lower ECE and diagnostics.

    Parameters:
        base_estimator : ClassifierMixin
            A classifier that is already fitted and supports `predict_proba`
            or `decision_function`.

        X : array-like of shape (n_samples, n_features)
            Feature matrix.

        y : array-like of shape (n_samples,)
            True binary labels (0/1).

        cv : int, default=5
            Number of cross-validation folds for `CalibratedClassifierCV`.

        n_bins : int, default=15
            Number of bins used in ECE computation.

    Returns:
        dict
            Dictionary with keys:
                - "best_model": the selected calibrated classifier
                - "best_method": "isotonic" or "sigmoid"
                - "best_ece": ECE of the selected model
                - "isotonic_ece": ECE of the isotonic model
                - "sigmoid_ece": ECE of the sigmoid model
    """
    check_is_fitted(base_estimator)

    iso_clf = CalibratedClassifierCV(
        base_estimator, method="isotonic", cv=cv
    )
    iso_clf.fit(X, y)
    iso_prob = iso_clf.predict_proba(X)[:, 1]
    iso_ece = compute_ece(y, iso_prob, n_bins=n_bins)

    sig_clf = CalibratedClassifierCV(
        base_estimator, method="sigmoid", cv=cv
    )
    sig_clf.fit(X, y)
    sig_prob = sig_clf.predict_proba(X)[:, 1]
    sig_ece = compute_ece(y, sig_prob, n_bins=n_bins)

    if iso_ece <= sig_ece:
        best_model = iso_clf
        best_method = "isotonic"
        best_ece = iso_ece
    else:
        best_model = sig_clf
        best_method = "sigmoid"
        best_ece = sig_ece

    return {
        "best_model": best_model,
        "best_method": best_method,
        "best_ece": best_ece,
        "isotonic_ece": iso_ece,
        "sigmoid_ece": sig_ece,
    }