# features/sample_weighting.py
# Team Narnia — PhysioNet Challenge 2026
#
# Entry 6 (2026-07-16/19): compute_value_weighted_sample_weights(), promoted
# from tools/reg_sweep.py (where it was developed and validated on Kaggle —
# see learning_log.md / learning_log_3, 2026-07-16 entries) into a shared
# features/ module, same precedent as AgeResidualizer (features/age_residuals.py):
# one definition, imported by both team_code.py (the real submission path)
# and tools/reg_sweep.py (the validation harness it was proven in), so the
# two can never silently diverge the way loso_cv.py's hand-copied Pipeline
# once did (see features/pipeline.py header for that incident).
#
# Validated result this function is responsible for (alpha=1.0, on top of
# C=0.001): +12.9% relative reward vs. C=0.001 alone, AUROC flat (max 0.21σ
# across all 3 folds). Reproduced bit-for-bit on an independent Kaggle run
# (learning_log_3, 2026-07-16 final entry). Combined with the C=0.001 change:
# +52% relative reward vs. the original shipped Entry 5 baseline, AUROC flat
# throughout the entire chain.

import numpy as np

from evaluate_model import compute_prevalence


def _lookup_prevalence_train(age_to_prevalence, age):
    """Same nearest-key fallback logic as tools/test_age_banded_threshold.py's
    _lookup_prevalence — kept as a local duplicate rather than a cross-file
    import, matching reg_sweep.py's original standalone-on-Kaggle discipline."""
    key = round(age)
    if key in age_to_prevalence:
        return age_to_prevalence[key]
    nearest = min(age_to_prevalence.keys(), key=lambda k: abs(k - age))
    return age_to_prevalence[nearest]


def compute_value_weighted_sample_weights(y_train, age_train, alpha):
    """
    Added 2026-07-16 (tools/reg_sweep.py), promoted here 2026-07-19 for
    Entry 6. Builds a per-training-sample weight array that layers a
    reward-VALUE-aware boost on top of whatever class_weight='balanced'
    already does (LOGREG_PARAMS — confirmed already active in every logreg
    config tested this project; this is an ADDITION, not a replacement).
    sklearn composes class_weight and an explicit sample_weight array
    multiplicatively, so both apply together.

    ONLY positive-class samples get boosted — negatives keep weight 1.0
    regardless of age, matching the actual intent (prioritize getting
    high-value positives right, not reweighting the whole population by
    age indiscriminately).

    CRITICAL LEAKAGE DISCIPLINE: age_to_prevalence here is fit using ONLY
    y_train/age_train. At LOSO-validation time (reg_sweep.py) that means
    only the patients in THIS fold's training set, never the held-out test
    fold. At real submission time (team_code.py) there is no fold rotation
    at all — train_model() trains once on whatever training set the
    organizers provide, so passing that full training set here is itself
    the correct, leakage-safe usage; there is no held-out slice to leak
    from in production. This is deliberately different from how
    compute_prevalence is used elsewhere in this project: reward SCORING
    correctly uses the FULL population as its reference (confirmed
    2026-07-09 finding), because that's evaluating an already-fixed
    decision against reality. This is different — it directly shapes what
    the MODEL LEARNS from these exact training patients, so using anything
    beyond the training set's own data here would leak test-correlated
    information into training, the same class of mistake AgeResidualizer's
    train-fold-only fitting discipline exists to prevent.

    alpha=0.0 reproduces IDENTICAL behavior to no weighting at all (returns
    all-ones) — backward compatible, matches every previously logged result
    exactly when alpha isn't explicitly set.
    """
    if alpha == 0.0:
        return np.ones(len(y_train))

    age_to_prevalence_train = compute_prevalence(age_train, y_train, age_train, gap=2)
    prevalences = np.array([_lookup_prevalence_train(age_to_prevalence_train, a)
                             for a in age_train])
    value = (1.0 / prevalences) - 1.0
    # Normalize using this training set's own min/max — not any global
    # reference — same train-only discipline as the prevalence fit above.
    value_norm = (value - value.min()) / (value.max() - value.min() + 1e-12)

    weights = np.ones(len(y_train))
    pos_mask = (y_train == 1)
    weights[pos_mask] = 1.0 + alpha * value_norm[pos_mask]
    return weights


# Candidate (2026-07-29): proximity-weighted positive boost. TESTED AND
# CLOSED 2026-07-29 (thirteenth ceiling door) — no promote at beta 0.5/1.0/
# 2.0 on the large set: I0002 negative and worsening with beta, effect
# sub-noise (mean delta +0.0009/+0.0011/-0.0007 vs a ~0.005-0.007 floor).
# See learning_log_4.md, 2026-07-29 (later) for the full result and the
# explicit "benign mixed result, NOT the Entry 6/9 trap" reading (spread
# SHRANK, S0001 improved every beta). Kept in-repo but DORMANT: imported
# only by tools/ablation_proximity_weight.py, never by team_code.py, same
# status HRV_ACTIVE_NAMES and SPECTRAL_ACTIVE_NAMES hold. Nothing to revert.
# Do NOT wire into team_code.py — Entry 8 stands.
#
# Distinct MECHANISM from compute_value_weighted_sample_weights above:
# that boosts positives by age-PREVALENCE (a reward-metric construct, and
# empirically AUROC-FLAT — it shifts calibrated magnitude near the
# boundary, not patient ranking). This boosts by DIAGNOSIS PROXIMITY (how
# soon after the PSG the first qualifying ICD code landed), a label-timing
# axis uncorrelated with the reward construct. Hypothesis: near-term
# positives carry cleaner prodromal signal at PSG time, so concentrating
# the fit on them changes the LEARNED DIRECTION -> changes ranking -> can
# move age-conditioned AUROC (the metric that decides the Challenge),
# unlike value-weighting or any threshold change. In-project prior: the
# 2026-07-19 site-ranking-flip diagnostic (I0002's positives captured
# ~345 days closer to diagnosis and it ranks best) is direct evidence the
# axis is real. It is only a hypothesis about DIRECTION, not magnitude —
# it may come back flat (joining value-weighting) or negative (if far-term
# positives carry real signal that down-weighting discards, a genuine risk
# at only ~500 positives). That is what the ablation is for.


def compute_proximity_weighted_sample_weights(y_train, time_to_event_train, beta):
    """
    Per-training-sample weight array that boosts POSITIVE-class samples by
    how soon their cognitive-impairment diagnosis followed the PSG. Layers
    on top of class_weight='balanced' (LOGREG_PARAMS) exactly like
    compute_value_weighted_sample_weights does, and is intended to COMPOSE
    multiplicatively with it (sample_weight = value_weight * proximity_weight),
    the same way sklearn already composes class_weight and sample_weight.

    beta scales the boost. beta=0.0 returns all-ones — byte-identical to no
    proximity weighting, so an un-set beta reproduces every prior result
    exactly (same backward-compatible contract as alpha=0.0 above).

    Weight assigned to positive i:  1.0 + beta * proximity_i,  where
    proximity_i = (tte_max - tte_i) / (tte_max - tte_min)  in [0, 1]
    (1.0 = soonest-diagnosed positive in this training set, 0.0 = latest).
    Negatives — and any positive with a missing/NaN Time_to_Event, which
    should not occur but is handled defensively — keep weight 1.0.

    NOT LEAKAGE. Time_to_Event is derived from the future diagnosis date
    and is correctly flagged pure-leakage AS A FEATURE (FEATURES.md — it
    must never enter the extraction vector; the model never sees it at
    inference). Used HERE it is a training-time SAMPLE WEIGHT only: it
    governs how much each training row contributes to the fit, it is not
    an input to prediction. This is the identical category to
    compute_value_weighted_sample_weights reading y_train itself to build
    its weights — allowed for the same reason.

    LEAKAGE DISCIPLINE (mirrors value-weighting). The [tte_min, tte_max]
    normalization reference is taken from the POSITIVES IN whatever
    y_train/time_to_event_train is passed in — nothing else. At LOSO-
    validation time that is this fold's TRAINING positives only, never the
    held-out fold (the ablation tool passes only the training split's
    Time_to_Event, exactly as reg_sweep.py passes only the training split
    to value-weighting). At real submission time there is no fold rotation,
    so the full training set is itself the correct, leakage-safe reference.
    Never pass a held-out fold's Time_to_Event here.
    """
    y_train = np.asarray(y_train)
    tte = np.asarray(time_to_event_train, dtype=float)

    if beta == 0.0:
        return np.ones(len(y_train))

    weights = np.ones(len(y_train), dtype=float)
    pos_mask = (y_train == 1) & np.isfinite(tte)
    if pos_mask.sum() == 0:
        return weights

    pos_tte = tte[pos_mask]
    lo, hi = pos_tte.min(), pos_tte.max()
    # If every positive shares one Time_to_Event, there is nothing to
    # differentiate — proximity collapses to 0 and all positives keep 1.0.
    proximity = (hi - pos_tte) / (hi - lo + 1e-12)   # 1=soonest, 0=latest
    weights[pos_mask] = 1.0 + beta * proximity
    return weights