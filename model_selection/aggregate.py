"""
Cross-validation result aggregation utilities.

This module provides:
    - aggregate_cv_results: construction of a sklearn-style ``cv_results_`` dictionary

The implementation transforms raw fold-level outputs produced by
``fit_and_score_fn`` into a structured result dictionary consistent
with scikit-learn’s ``GridSearchCV``. It supports both single-metric
and multi-metric scoring, includes per-split metrics, computes means
and standard deviations across folds, and assigns rankings for each
metric. The resulting structure is fully compatible with downstream
analysis and model selection procedures.
"""


# Import libraries

import numpy as np
from collections import defaultdict

def _apply(fn):
    return lambda x: fn(x)

def aggregate_cv_results(scored_results, scoring, return_train_score):
    """
    Aggregate fold-level results into a sklearn-style ``cv_results_`` structure.

    This function transforms the list of dictionaries returned by
    ``fit_and_score_fn`` into a result dictionary consistent with the
    conventions of scikit-learn’s ``GridSearchCV``. It supports both
    single-metric and multi-metric scoring, includes per-split results,
    computes means and standard deviations across folds, and assigns
    rankings for each metric.

    Parameters:
        scored_results : list of dict
            Flat list of dictionaries, each corresponding to one evaluation
            of (parameter_set × cross-validation fold).

        scoring : callable, dict, or None
            Scoring specification used during evaluation.

        return_train_score : bool
            Whether training metrics were computed and should be included.

    Returns:
        dict
            Sklearn-compatible ``cv_results_`` dictionary containing:
                - ``params`` : list of parameter dicts
                - ``param_<name>`` : lists of parameter values
                - ``splitX_test_<metric>`` entries
                - ``splitX_train_<metric>`` entries (optional)
                - ``mean_test_<metric>`` / ``std_test_<metric>``
                - ``mean_train_<metric>`` / ``std_train_<metric>``
                - ``rank_test_<metric>`` for each metric
    """
    if isinstance(scoring, dict):
        metrics = list(scoring.keys())
    else:
        metrics = ["score"]

    grouped = defaultdict(list)
    for r in scored_results:
        grouped[r["candidate_index"]].append(r)

    candidate_ids = sorted(grouped.keys())
    params_list = [grouped[cid][0]["params"] for cid in candidate_ids]

    out = {"params": params_list}

    param_names = params_list[0].keys()
    for name in param_names:
        out[f"param_{name}"] = [p[name] for p in params_list]

    n_splits = len(grouped[candidate_ids[0]])

    fit_time = {f"split{i}_fit_time": [] for i in range(n_splits)}
    predict_time = {f"split{i}_predict_time": [] for i in range(n_splits)}
    score_time = {f"split{i}_score_time": [] for i in range(n_splits)}

    mean_fit = []
    std_fit = []
    mean_pred = []
    std_pred = []
    mean_score_t = []
    std_score_t = []

    for cid in candidate_ids:
        splits = grouped[cid]

        ft = [d.get("fit_time", np.nan) for d in splits]
        pt = [d.get("predict_time", np.nan) for d in splits]
        st = [d.get("score_time", np.nan) for d in splits]

        mean_fit.append(np.nanmean(ft))
        std_fit.append(np.nanstd(ft))
        mean_pred.append(np.nanmean(pt))
        std_pred.append(np.nanstd(pt))
        mean_score_t.append(np.nanmean(st))
        std_score_t.append(np.nanstd(st))

        for i, d in enumerate(splits):
            fit_time[f"split{i}_fit_time"].append(d.get("fit_time", np.nan))
            predict_time[f"split{i}_predict_time"].append(d.get("predict_time", np.nan))
            score_time[f"split{i}_score_time"].append(d.get("score_time", np.nan))

    out.update(fit_time)
    out.update(predict_time)
    out.update(score_time)

    out["mean_fit_time"] = mean_fit
    out["std_fit_time"] = std_fit
    out["mean_predict_time"] = mean_pred
    out["std_predict_time"] = std_pred
    out["mean_score_time"] = mean_score_t
    out["std_score_time"] = std_score_t

    for metric in metrics:
        test_key = f"test_{metric}" if metric != "score" else "test_score"
        train_key = f"train_{metric}" if metric != "score" else "train_score"

        split_test = {f"split{i}_{test_key}": [] for i in range(n_splits)}
        split_train = {f"split{i}_{train_key}": [] for i in range(n_splits)} if return_train_score else None

        mean_test = []
        std_test = []
        mean_train = []
        std_train = []

        for cid in candidate_ids:
            splits = grouped[cid]

            tv = [d.get(test_key, np.nan) for d in splits]
            mean_test.append(_apply(np.nanmean)(tv))
            std_test.append(_apply(np.nanstd)(tv))

            if return_train_score:
                trv = [d.get(train_key, np.nan) for d in splits]
                mean_train.append(np.nanmean(trv))
                std_train.append(np.nanstd(trv))

            for i, d in enumerate(splits):
                split_test[f"split{i}_{test_key}"].append(d.get(test_key, np.nan))
                if return_train_score:
                    split_train[f"split{i}_{train_key}"].append(d.get(train_key, np.nan))

        out.update(split_test)
        if return_train_score:
            out.update(split_train)

        out[f"mean_test_{metric}"] = mean_test
        out[f"std_test_{metric}"] = std_test

        if return_train_score:
            out[f"mean_train_{metric}"] = mean_train
            out[f"std_train_{metric}"] = std_train

        safe_means = [(-np.inf if (m is None or (isinstance(m, float) and np.isnan(m))) else m) for m in mean_test]
        ranks = np.argsort(np.argsort([-m for m in safe_means])) + 1
        out[f"rank_test_{metric}"] = list(ranks)

    return out