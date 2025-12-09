"""
Probability calibration utilities for classification models.

This module provides:
    - compute_ece: Expected Calibration Error (ECE) for binary and multiclass
    - calibrate_classifier: comparison of isotonic vs Platt (sigmoid) calibration

The implementation supports both binary and multiclass classification.
For ECE, it uses the standard definition based on the confidence of the
predicted class and its empirical accuracy in probability bins.
"""

from typing import Dict, Any, Optional
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.base import ClassifierMixin


def compute_ece(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 15
) -> float:
    """
    Compute Expected Calibration Error (ECE) for binary or multiclass problems.

    ECE is defined over bins of predicted confidence. For each sample, the
    predicted class is the argmax over class probabilities, and the confidence
    is the corresponding maximum probability. Bins are formed over confidence
    values in [0, 1]. In each bin, the absolute difference between empirical
    accuracy and mean confidence is weighted by the fraction of samples in
    that bin and summed across all bins.

    Parameters:
        y_true : array-like of shape (n_samples,)
            True class labels.

        y_prob : array-like of shape (n_samples,) or (n_samples, n_classes)
            Predicted probabilities. For binary classification, this can be a
            one-dimensional array of positive-class probabilities; internally
            it is converted to a two-column representation.

        n_bins : int, default=15
            Number of equal-width bins in [0, 1] used to group predictions.

    Returns:
        float
            Scalar ECE value. Lower values indicate better calibration.
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    eps = 1e-12

    if y_prob.ndim == 1:
        p_pos = np.clip(y_prob, eps, 1.0 - eps)
        y_prob = np.vstack([1.0 - p_pos, p_pos]).T
    else:
        y_prob = np.clip(y_prob, eps, 1.0 - eps)
        row_sums = y_prob.sum(axis=1, keepdims=True)
        y_prob = y_prob / np.where(row_sums == 0.0, 1.0, row_sums)

    confidences = y_prob.max(axis=1)
    predictions = y_prob.argmax(axis=1)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)

    for i in range(n_bins):
        left, right = bin_edges[i], bin_edges[i + 1]
        if i < n_bins - 1:
            in_bin = (confidences >= left) & (confidences < right)
        else:
            in_bin = (confidences >= left) & (confidences <= right)

        if not np.any(in_bin):
            continue

        conf_bin = confidences[in_bin]
        true_bin = y_true[in_bin]
        pred_bin = predictions[in_bin]

        avg_confidence = float(conf_bin.mean())
        avg_accuracy = float((true_bin == pred_bin).mean())
        bin_weight = len(true_bin) / n

        ece += bin_weight * abs(avg_accuracy - avg_confidence)

    return float(ece)


def calibrate_classifier(
    base_estimator: ClassifierMixin,
    X,
    y,
    cv: int = 5,
    n_bins: int = 15,
    eval_X: Optional[Any] = None,
    eval_y: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Calibrate a classifier using isotonic regression and Platt scaling,
    and select the method with the lower ECE.

    The calibration is performed via CalibratedClassifierCV, which applies
    cross-validated calibration on (X, y). ECE is then computed on a separate
    evaluation set (eval_X, eval_y) if provided, or on the same data otherwise.

    This function supports both binary and multiclass classification through
    `compute_ece`, which operates on predicted class confidences.

    Parameters:
        base_estimator : ClassifierMixin
            A scikit-learn-compatible classifier. It is treated as an unfitted
            prototype; CalibratedClassifierCV will clone and fit it internally.

        X : array-like of shape (n_samples, n_features)
            Feature matrix used for calibration fitting.

        y : array-like of shape (n_samples,)
            True class labels corresponding to X.

        cv : int, default=5
            Number of cross-validation folds for CalibratedClassifierCV.

        n_bins : int, default=15
            Number of bins used in ECE computation.

        eval_X : array-like of shape (n_eval_samples, n_features), optional
            Feature matrix used for evaluating calibration quality. If None,
            X is used.

        eval_y : array-like of shape (n_eval_samples,), optional
            True labels for eval_X. If None, y is used.

    Returns:
        dict
            Dictionary with keys:
                - "best_model": calibrated classifier with lowest ECE
                - "best_method": "isotonic" or "sigmoid"
                - "best_ece": ECE of the selected model
                - "isotonic_ece": ECE of the isotonic model
                - "sigmoid_ece": ECE of the sigmoid model
    """
    if eval_X is None or eval_y is None:
        eval_X, eval_y = X, y

    iso_clf = CalibratedClassifierCV(
        base_estimator, method="isotonic", cv=cv
    )
    iso_clf.fit(X, y)
    iso_prob = iso_clf.predict_proba(eval_X)
    iso_ece = compute_ece(eval_y, iso_prob, n_bins=n_bins)

    sig_clf = CalibratedClassifierCV(
        base_estimator, method="sigmoid", cv=cv
    )
    sig_clf.fit(X, y)
    sig_prob = sig_clf.predict_proba(eval_X)
    sig_ece = compute_ece(eval_y, sig_prob, n_bins=n_bins)

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