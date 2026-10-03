"""Cross-validated base models and a stacked ensemble.

Design:
* Each feature block (handcrafted, text embedding, audio embedding) gets its own
  base regressor, so we can see how much every modality contributes.
* Base models are evaluated with repeated stratified K-fold. Out-of-fold (OOF)
  predictions give an honest estimate of generalisation; test predictions are
  the average of all fold models.
* A non-negative linear stacker combines the OOF predictions. Its weights are
  directly interpretable as "how much each modality is trusted".
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import lightgbm as lgb
import numpy as np
from scipy.stats import pearsonr
from sklearn.base import RegressorMixin, clone
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from .config import SEED

LABEL_MIN, LABEL_MAX = 0.0, 5.0

# Recordings drowned in broadband noise were rated 0 ("unscorable"). In training they all have median spectral
# flatness 0.51-0.56, while every genuine recording is <= 0.17, so a fixed threshold in the gap separates them
# perfectly. They are excluded from regression training and predicted as 0 at inference.
UNSCORABLE_FLATNESS = 0.35


def is_unscorable(spectral_flatness) -> np.ndarray:
    return np.asarray(spectral_flatness) > UNSCORABLE_FLATNESS


def rmse(y, p) -> float:
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(p)) ** 2)))


def pearson(y, p) -> float:
    return float(pearsonr(y, p)[0])


def strat_bins(y: np.ndarray, min_count: int = 10) -> np.ndarray:
    """Discretise labels for stratification, merging rare (mostly low) scores into their neighbour."""
    bins = np.round(np.asarray(y) * 2) / 2
    values, counts = np.unique(bins, return_counts=True)
    min_count = min(min_count, counts.max())  # tiny datasets: always keep at least the largest class
    for v, c in zip(values, counts):
        if c < min_count:
            others = values[(values != v) & (counts >= min_count)]
            bins[bins == v] = others[np.argmin(np.abs(others - v))]
    return (bins * 2).astype(int)  # integer class codes, as sklearn's stratified splitters require


def ridge() -> RegressorMixin:
    return make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-1, 5, 25)))


def svr() -> RegressorMixin:
    """RBF-kernel SVR: a non-linear counterpart to Ridge for the dense embeddings."""
    return make_pipeline(StandardScaler(), SVR(C=3.0, epsilon=0.1, gamma="scale"))


def lgbm() -> RegressorMixin:
    return lgb.LGBMRegressor(
        n_estimators=400, learning_rate=0.02, num_leaves=15, min_child_samples=15,
        subsample=0.8, subsample_freq=1, colsample_bytree=0.7, reg_lambda=1.0,
        random_state=SEED, verbose=-1,
    )


@dataclass
class CVResult:
    name: str
    oof: np.ndarray
    test: np.ndarray
    rmse: float
    pearson: float
    models: list = field(default_factory=list, repr=False)


def cross_validate(
    name: str,
    make_model: Callable[[], RegressorMixin],
    X: np.ndarray,
    y: np.ndarray,
    X_test: np.ndarray | None = None,
    n_splits: int = 5,
    n_repeats: int = 3,
    keep_models: bool = False,
) -> CVResult:
    cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=SEED)
    oof_sum, oof_cnt = np.zeros(len(y)), np.zeros(len(y))
    test_preds, models = [], []
    for tr, va in cv.split(X, strat_bins(y)):
        model = clone(make_model()).fit(X[tr], y[tr])
        oof_sum[va] += model.predict(X[va])
        oof_cnt[va] += 1
        if X_test is not None:
            test_preds.append(model.predict(X_test))
        if keep_models:
            models.append(model)
    oof = np.clip(oof_sum / oof_cnt, LABEL_MIN, LABEL_MAX)
    test = np.clip(np.mean(test_preds, axis=0), LABEL_MIN, LABEL_MAX) if test_preds else np.array([])
    return CVResult(name, oof, test, rmse(y, oof), pearson(y, oof), models)


@dataclass
class StackResult:
    weights: dict[str, float]
    intercept: float
    oof: np.ndarray  # nested-CV predictions of the stacker (honest estimate)
    test: np.ndarray
    rmse: float
    pearson: float
    stacker: LinearRegression


def stack(results: list[CVResult], y: np.ndarray) -> StackResult:
    """Fit a non-negative linear blend of base-model OOF predictions.

    The stacker itself is re-evaluated with an outer K-fold over the OOF matrix,
    so the reported stack score is not optimistically biased by fitting weights
    on the same predictions it is scored on.
    """
    P = np.column_stack([r.oof for r in results])
    T = np.column_stack([r.test for r in results]) if results[0].test.size else None
    nested = np.zeros(len(y))
    for tr, va in StratifiedKFold(5, shuffle=True, random_state=SEED + 1).split(P, strat_bins(y)):
        nested[va] = LinearRegression(positive=True).fit(P[tr], y[tr]).predict(P[va])
    nested = np.clip(nested, LABEL_MIN, LABEL_MAX)

    stacker = LinearRegression(positive=True).fit(P, y)
    test = np.clip(stacker.predict(T), LABEL_MIN, LABEL_MAX) if T is not None else np.array([])
    return StackResult(
        weights={r.name: float(w) for r, w in zip(results, stacker.coef_)},
        intercept=float(stacker.intercept_),
        oof=nested, test=test, rmse=rmse(y, nested), pearson=pearson(y, nested), stacker=stacker,
    )
