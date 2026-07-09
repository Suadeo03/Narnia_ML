#!/usr/bin/env python

# Team Narnia — PhysioNet Challenge 2026
# Entry 3: Added Platt-scaled calibration (CalibratedClassifierCV) and
#          explicit threshold (0.12, tuned via loso_cv.py pooled sweep)
#          replacing model.predict()'s default 0.5 cutoff.
# Entry 4: Added AgeResidualizer pipeline step (2 new features:
#          CA_rate_age_residual, EEG_var_REM_Wake_age_residual). Both were
#          previously written off as uninformative but age-residualized EDA
#          (2026-07-02) showed real signal masked by an age confound. Fit at
#          train time only, applied identically at inference — see
#          features/age_residuals.py. Feature count: 48 -> 50.
# Entry 4 (fix, 2026-07-03): moved model-pipeline construction (XGBoost
#          params, calibration, AgeResidualizer wiring) out of this file and
#          into features/pipeline.py's build_pipeline(), shared with
#          loso_cv.py. This closes a real bug where loso_cv.py had its own
#          hand-copied Pipeline() that silently never got the AgeResidualizer
#          step added here — LOSO was validating a stale 48-feature model
#          while this file had already moved to 50. See features/pipeline.py
#          for the full incident writeup.
#
# See features/FEATURES.md and LEARNING_LOG.md for full rationale.

################################################################################
# Libraries
################################################################################

import joblib
import numpy as np
import os
from tqdm import tqdm

from helper_code import *

# Feature modules
from features.demographic        import extract_demographic_features
from features.caisr_base         import extract_caisr_base_features
from features.caisr_enriched     import extract_caisr_enriched_features
from features.physiological_ratios import (extract_physiological_ratio_features,
                                            N_RATIO_FEATURES)
from features.human              import extract_human_annotations_features
from features.pipeline           import build_pipeline
from features import (N_CAISR_BASE_FEATURES, N_CAISR_ENRICHED_FEATURES,
                      N_DEMOGRAPHIC_FEATURES)

################################################################################
# Configuration
################################################################################

SCRIPT_DIR       = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CSV_PATH = os.path.join(SCRIPT_DIR, 'channel_table.csv')

# Fallback sizes for missing files
_N_BASE     = N_CAISR_BASE_FEATURES      # 12
_N_ENRICHED = N_CAISR_ENRICHED_FEATURES  # 11
_N_RATIO    = N_RATIO_FEATURES           # 15

# Entry 3 — calibration + threshold.
# PLACEHOLDER — replace with the value loso_cv.py's loso_threshold_sweep.csv
# recommends (maximizes reward without costing >~0.02 AUROC vs the AUROC-
# optimal threshold). Do NOT submit with 0.10 unverified — run the sweep
# first. Local prevalence is ~7.6% in the small training set; 0.5 (the old
# default) is far too conservative for the reward metric.
#verified THRESHOLD
THRESHOLD = 0.12

################################################################################
# Required functions
################################################################################

def train_model(data_folder, model_folder, verbose, csv_path=DEFAULT_CSV_PATH):
    if verbose:
        print('Finding the Challenge data...')

    patient_data_file     = os.path.join(data_folder, DEMOGRAPHICS_FILE)
    patient_metadata_list = find_patients(patient_data_file)
    num_records           = len(patient_metadata_list)

    if num_records == 0:
        raise FileNotFoundError('No data were provided.')

    if verbose:
        print('Extracting features and labels from the data...')

    features = []
    labels   = []

    pbar = tqdm(range(num_records), desc='Extracting Features',
                unit='record', disable=not verbose)
    for i in pbar:
        try:
            record     = patient_metadata_list[i]
            patient_id = record[HEADERS['bids_folder']]
            site_id    = record[HEADERS['site_id']]
            session_id = record[HEADERS['session_id']]

            if verbose:
                pbar.set_postfix({'patient': patient_id})

            # ── Demographics ─────────────────────────────────────────────────
            demo_file    = os.path.join(data_folder, DEMOGRAPHICS_FILE)
            patient_data = load_demographics(demo_file, patient_id, session_id)
            demo_f = extract_demographic_features(patient_data)

            # ── Physiological EDF ─────────────────────────────────────────────
            phys_file = os.path.join(
                data_folder, PHYSIOLOGICAL_DATA_SUBFOLDER,
                site_id, f'{patient_id}_ses-{session_id}.edf')
            if not os.path.exists(phys_file):
                if verbose:
                    tqdm.write(
                        f'  ! Missing physiological EDF for {patient_id} — skipping.')
                continue
            phys_data, phys_fs = load_signal_data(phys_file)

            # ── CAISR Annotations ─────────────────────────────────────────────
            algo_file = os.path.join(
                data_folder, ALGORITHMIC_ANNOTATIONS_SUBFOLDER,
                site_id, f'{patient_id}_ses-{session_id}_caisr_annotations.edf')
            if os.path.exists(algo_file):
                algo_data, _ = load_signal_data(algo_file)
                caisr_base     = extract_caisr_base_features(algo_data)
                caisr_enriched = extract_caisr_enriched_features(algo_data)
                # Ratio features require BOTH phys + CAISR
                ratio_f = extract_physiological_ratio_features(
                    phys_data, phys_fs, algo_data, csv_path=csv_path)
            else:
                tqdm.write(
                    f'Error loading EDF file: [Errno 2] No such file or directory: '
                    f"'{algo_file}'")
                caisr_base     = np.full(_N_BASE,     float('nan'))
                caisr_enriched = np.full(_N_ENRICHED, float('nan'))
                ratio_f        = np.full(_N_RATIO,    float('nan'))

            # ── Human Annotations (training only — not used in model) ─────────
            human_file = os.path.join(
                data_folder, HUMAN_ANNOTATIONS_SUBFOLDER,
                site_id, f'{patient_id}_ses-{session_id}_expert_annotations.edf')
            if os.path.exists(human_file):
                human_data, _ = load_signal_data(human_file)
                _ = extract_human_annotations_features(human_data)

            # ── Label ─────────────────────────────────────────────────────────
            label = load_diagnoses(
                os.path.join(data_folder, DEMOGRAPHICS_FILE), patient_id)

            if label == 0 or label == 1:
                # 48-feature extraction vector. AgeResidualizer (pipeline
                # step, Entry 4) appends the 2 age-residualized features
                # at model.fit() time — do not hstack them here.
                features.append(np.hstack([
                    demo_f,
                    caisr_base,
                    caisr_enriched,
                    ratio_f,
                ]))
                labels.append(label)

            if 'phys_data'  in locals(): del phys_data
            if 'algo_data'  in locals(): del algo_data
            if 'human_data' in locals(): del human_data

        except Exception as e:
            tqdm.write(f'  !!! Error on record {i+1} ({patient_id}): {e}')
            continue

    pbar.close()

    features = np.asarray(features, dtype=np.float32)
    labels   = np.asarray(labels,   dtype=bool)

    if verbose:
        n_pos = int(labels.sum())
        n_neg = int((~labels).sum())
        print(f'Training on {len(labels)} patients '
              f'({n_pos} positive, {n_neg} negative)...')
        print(f'Feature vector shape: {features.shape}')

    # ── Model pipeline (Entry 2 XGBoost config + Entry 3 calibration +
    #    Entry 4 AgeResidualizer) ───────────────────────────────────────────────
    # Built via features/pipeline.py's build_pipeline() — the single shared
    # definition also used by loso_cv.py, so the validation harness and this
    # submission can never silently diverge on what "the model" is again
    # (see features/pipeline.py header for the incident that motivated this).
    model = build_pipeline(labels, calibrated=True)
    model.fit(features, labels)

    os.makedirs(model_folder, exist_ok=True)
    save_model(model_folder, model)

    if verbose:
        print('Done.')
        print()


def load_model(model_folder, verbose):
    return joblib.load(os.path.join(model_folder, 'model.sav'))


def run_model(model, record, data_folder, verbose):
    model = model['model']

    patient_id = record[HEADERS['bids_folder']]
    site_id    = record[HEADERS['site_id']]
    session_id = record[HEADERS['session_id']]

    # ── Demographics ──────────────────────────────────────────────────────────
    demo_file    = os.path.join(data_folder, DEMOGRAPHICS_FILE)
    patient_data = load_demographics(demo_file, patient_id, session_id)
    demo_f = extract_demographic_features(patient_data)

    # ── Physiological EDF ─────────────────────────────────────────────────────
    phys_file = os.path.join(
        data_folder, PHYSIOLOGICAL_DATA_SUBFOLDER,
        site_id, f'{patient_id}_ses-{session_id}.edf')
    phys_data = phys_fs = None
    if os.path.exists(phys_file):
        phys_data, phys_fs = load_signal_data(phys_file)

    # ── CAISR Annotations ─────────────────────────────────────────────────────
    algo_file = os.path.join(
        data_folder, ALGORITHMIC_ANNOTATIONS_SUBFOLDER,
        site_id, f'{patient_id}_ses-{session_id}_caisr_annotations.edf')
    if os.path.exists(algo_file):
        algo_data, _ = load_signal_data(algo_file)
        caisr_base     = extract_caisr_base_features(algo_data)
        caisr_enriched = extract_caisr_enriched_features(algo_data)
        if phys_data is not None:
            ratio_f = extract_physiological_ratio_features(
                phys_data, phys_fs, algo_data)
        else:
            ratio_f = np.full(_N_RATIO, float('nan'))
    else:
        caisr_base     = np.full(_N_BASE,     float('nan'))
        caisr_enriched = np.full(_N_ENRICHED, float('nan'))
        ratio_f        = np.full(_N_RATIO,    float('nan'))

    # NOTE: this is the 48-feature extraction vector (demo/base/enriched/
    # ratios). The loaded `model` pipeline's age_residual step appends the
    # 2 Entry 4 age-residualized features automatically — do not hstack
    # them here, and do not hardcode 50 anywhere in this function.
    features = np.hstack([
        demo_f,
        caisr_base,
        caisr_enriched,
        ratio_f,
    ]).reshape(1, -1)

    # ── Entry 3: calibrated probability + explicit threshold ──────────────────
    # model.predict_proba() is already calibrated — calibration is baked into
    # the pipeline itself (see train_model()). model.predict() is NOT used
    # here: its default 0.5 cutoff is far too conservative at ~7.6% local
    # prevalence, which is why entry 1/2 reward was 0.011 despite reasonable
    # AUROC. THRESHOLD is set from loso_cv.py's pooled threshold sweep.
    probability_output = model.predict_proba(features)[0][1]
    binary_output       = int(probability_output > THRESHOLD)

    return binary_output, probability_output


################################################################################
# Utilities
################################################################################

def save_model(model_folder, model):
    joblib.dump({'model': model},
                os.path.join(model_folder, 'model.sav'),
                protocol=0)