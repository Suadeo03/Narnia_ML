# features/pipeline.py
# Shared model pipeline construction — single source of truth for
# team_code.py (the shipped submission) AND loso_cv.py (the validation
# harness).
#
# Root cause this file fixes (2026-07-03): loso_cv.py had its own
# hand-copied Pipeline() construction, predating AgeResidualizer (Entry 4).
# When AgeResidualizer was added to team_code.py, loso_cv.py's copy was
# never updated — LOSO silently kept validating the OLD 48-feature
# pipeline while team_code.py had already moved to 50. Caught because a
# post-"Entry 4" LOSO run matched the logged Entry 3 numbers to 4 decimal
# places across every metric, including per-fold top-feature importance —
# if AgeResidualizer had actually been active, at least some numerical
# drift would be expected (different feature count into the imputer,
# different fitted trees).
#
# Fix: both team_code.py and loso_cv.py now import build_pipeline() from
# here instead of each constructing their own Pipeline inline. There is
# now exactly ONE place that defines "what the model is" — changing it
# here changes it everywhere that matters, and it is structurally
# impossible for the validation harness and the submission to silently
# diverge again the way they just did.
#
# Lives under features/ (not repo root) deliberately: check_submission_files.py
# only resolves `from features.X import ...` / `from features import ...`
# into required-file paths. A root-level shared module would instead be
# misclassified as a third-party pip package by that script's import parser
# and flagged as "missing from requirements.txt" — a confusing false
# warning for something that isn't a package at all.

# features/pipeline.py
# Shared model pipeline construction — single source of truth for
# team_code.py AND loso_cv.py / reg_sweep.py.
#
# 2026-07-08 addition: build_logreg_pipeline(), for the Ridge/Lasso/
# ElasticNet regularization sweep. Reuses the same AgeResidualizer ->
# SimpleImputer -> [CalibratedClassifierCV] skeleton as build_pipeline(),
# swapping XGBClassifier for LogisticRegression, with two changes that
# matter specifically for a regularization sweep:
#
#   1. Added StandardScaler between the imputer and the classifier. This
#      was NOT verified to already exist in whatever build_logreg_pipeline()
#      produced the 0.6002 large-set LOSO result (learning_log.md,
#      2026-07-07) — that code wasn't available when this was written.
#      A C sweep is not meaningfully comparable across features without
#      scaling first: Age (~0-100), BMI (~15-50), event-rate features
#      (~0-30/hr), and ratio features (~0-3) sit on wildly different
#      scales, so an unscaled C controls regularization strength
#      inconsistently across coefficients. IF a scaler already existed in
#      the original build_logreg_pipeline(), the 0.6002 baseline was
#      already fit this way and nothing changes. If it did NOT exist,
#      this sweep's results are not directly comparable to that 0.6002
#      number at face value — confirm which case you're in before treating
#      a sweep win as a clean improvement over the existing baseline.
#   2. class_weight='balanced' instead of an XGBoost-style scale_pos_weight
#      — sklearn's LogisticRegression equivalent, same "derive from actual
#      data, don't hardcode" principle already established for XGBoost
#      (learning_log.md, 2026-06-30).
#
# solver='saga' is required for L1 and ElasticNet penalties; used
# uniformly (including for L2) so penalty type alone varies across the
# sweep, not solver + penalty together.

import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from features.age_residuals import AgeResidualizer
from features import IDX_AGE, IDX_CA_RATE, IDX_EEG_VAR_REM_WAKE

XGB_PARAMS = dict(
    n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.7,
    colsample_bytree=0.6, reg_alpha=0.1, reg_lambda=2.0, min_child_weight=5,
    random_state=42, eval_metric='auc', verbosity=0,
)


def _scale_pos_weight(y_train):
    y_train = np.asarray(y_train)
    n_pos = int((y_train == 1).sum())
    n_neg = int((y_train == 0).sum())
    return n_neg / n_pos if n_pos > 0 else 1.0


def _age_residualizer():
    return AgeResidualizer(age_idx=IDX_AGE, ca_rate_idx=IDX_CA_RATE,
                            eeg_var_rem_wake_idx=IDX_EEG_VAR_REM_WAKE)


def build_pipeline(y_train, calibrated=True):
    """XGBoost pipeline — unchanged from the existing shipped version."""
    spw = _scale_pos_weight(y_train)
    xgb = XGBClassifier(scale_pos_weight=spw, **XGB_PARAMS)
    classifier = CalibratedClassifierCV(xgb, method='sigmoid', cv=5) if calibrated else xgb
    return Pipeline([
        ('age_residual', _age_residualizer()),
        ('imputer', SimpleImputer(strategy='median')),
        ('classifier', classifier),
    ])


def build_logreg_pipeline(y_train, penalty='l2', C=1.0, l1_ratio=None,
                           calibrated=True, max_iter=5000):
    """
    Builds the logreg pipeline for the regularization sweep:

        AgeResidualizer -> SimpleImputer(median) -> StandardScaler
            -> LogisticRegression(penalty, C[, l1_ratio]) [+ CalibratedClassifierCV]

    penalty: 'l1', 'l2', or 'elasticnet'. 'elasticnet' requires l1_ratio
        in [0, 1] (0 = pure L2, 1 = pure L1).
    C: inverse regularization strength (smaller C = stronger regularization),
        sklearn's convention — same direction as everywhere else in sklearn.
    y_train: only used for class_weight sanity/logging parity with
        build_pipeline's scale_pos_weight derivation; class_weight='balanced'
        computes its own weights internally regardless.

    Returns an UNFITTED Pipeline, same contract as build_pipeline().
    """
    if penalty == 'elasticnet' and l1_ratio is None:
        raise ValueError("penalty='elasticnet' requires l1_ratio in [0, 1].")

    lr_kwargs = dict(
        penalty=penalty, C=C, solver='saga', max_iter=max_iter,
        class_weight='balanced', random_state=42,
    )
    if penalty == 'elasticnet':
        lr_kwargs['l1_ratio'] = l1_ratio

    lr = LogisticRegression(**lr_kwargs)
    classifier = CalibratedClassifierCV(lr, method='sigmoid', cv=5) if calibrated else lr

    return Pipeline([
        ('age_residual', _age_residualizer()),
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('classifier', classifier),
    ])


def extract_fitted_coefficients(fitted_pipeline):
    """
    Pulls the linear coefficients back out of a fitted build_logreg_pipeline()
    Pipeline, for the coefficient-stability-across-folds check. Handles both
    calibrated=True (CalibratedClassifierCV wraps N cloned+refit estimators —
    averages their coefficients) and calibrated=False (direct LogisticRegression).

    Returns a 1D np.array of length 50 (post-AgeResidualizer feature count),
    in STANDARDIZED-feature space (i.e. these are coefficients on scaled
    features, not raw units) — comparable across folds/features for a
    stability check, but do not reinterpret them as raw-unit effect sizes
    without un-scaling first.
    """
    clf = fitted_pipeline.named_steps['classifier']
    if isinstance(clf, CalibratedClassifierCV):
        coefs = np.stack([cc.estimator.coef_[0] for cc in clf.calibrated_classifiers_])
        return coefs.mean(axis=0)
    return clf.coef_[0]