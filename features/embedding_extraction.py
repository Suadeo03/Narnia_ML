# features/embedding_extraction.py
# Option B (borrowed representation) — channel prep for a frozen pretrained
# sleep-EEG foundation model. See tools/TESTING_PLAN_endgame.md ADDENDUM
# 2026-07-26 for the rationale (representation ceiling, 0.77 field frontier).
#
# This module does ONLY the channel-mapping / resampling / epoching step —
# pure numpy, no torch dependency here on purpose, so it can be unit-tested
# and reused regardless of which foundation model (OSF-Base, SleepFM, ...)
# consumes its output. The actual model forward pass lives in
# tools/extract_foundation_embeddings.py.
#
# Target spec: OSF-Base (yang-ai-lab/OSF-Base, MIT license, confirmed
# 2026-07-27). 12 channels, 64 Hz, 30s epochs, shape [12, 1920]. Channel
# order per the model card is FIXED — do not reorder:
#   ECG, EMG_Chin, EMG_LLeg, EMG_RLeg, ABD, THX, NP, SN,
#   EOG_E1_A2, EOG_E2_A1, EEG_C3_A2, EEG_C4_A1
#
# Montage check against channel_table.csv (done 2026-07-27, refined against
# a real I0002 EDF the same day): 10 of 12 channels are exact matches. Two
# are known approximations, not exact-spec matches:
#   - SN (snore): genuinely absent from this dataset entirely. Zero-filled
#     — see MISSING_CHANNEL_FILL and the returned `missing_channels` flag.
#   - EOG_E1_A2 / EOG_E2_A1: this dataset ships EEG already bipolar-
#     referenced (c3-m2, c4-m1, etc.) but EOG as raw monopolar (e1, e2)
#     with NO standalone m1/m2 channel recorded anywhere — confirmed
#     directly against a real EDF, not inferred from channel_table.csv
#     alone. True mastoid-referenced EOG is not recoverable from this
#     data. Falls back to raw e1/e2 as the closest available proxy —
#     flagged in the returned `approximated_channels` list, not silently
#     treated as spec-exact.
# Both gaps are real, undiagnosed risk (OSF's own paper motivation is that
# "existing FMs fail to generalize to missing channels" — unclear whether
# OSF's own channel-invariant pretraining actually solves this for a
# channel that's missing/approximated for EVERY patient, vs. missing for
# a minority). Flagging per project discipline rather than assuming it's
# fine.
#
# Equipment-scale confound: OSF's demo does not document any per-channel
# normalization step. Given this project's own confirmed finding (2026-07-01
# EDA) that absolute signal amplitude is equipment/site-dependent, this
# module z-scores each channel independently per-patient before resampling
# — cheap insurance against the same confound that killed the original
# absolute-Hjorth features, at the cost of discarding true absolute-amplitude
# information the foundation model might otherwise use. Documented as a
# deliberate choice, not a default worth assuming away.

import numpy as np
import os
import sys
from fractions import Fraction
from scipy.signal import resample_poly

_FEATURES_DIR    = os.path.dirname(os.path.abspath(__file__))
_REPO_DIR        = os.path.dirname(_FEATURES_DIR)
DEFAULT_CSV_PATH = os.path.join(_REPO_DIR, 'channel_table.csv')

# Repo root must be on sys.path BEFORE importing helper_code -- running this
# file directly (`python features/embedding_extraction.py`) otherwise only
# puts features/ itself on sys.path, not the repo root one level up.
if _REPO_DIR not in sys.path:
    sys.path.insert(0, _REPO_DIR)

from helper_code import (
    load_rename_rules, standardize_channel_names_rename_only,
    derive_bipolar_signal
)

TARGET_FS          = 64.0
EPOCH_SECONDS      = 30.0
SAMPLES_PER_EPOCH  = int(round(TARGET_FS * EPOCH_SECONDS))   # 1920
N_CHANNELS         = 12

# Fixed order — matches yang-ai-lab/OSF-Base's model card exactly.
OSF_CHANNEL_ORDER = [
    'ECG', 'EMG_Chin', 'EMG_LLeg', 'EMG_RLeg', 'ABD', 'THX', 'NP', 'SN',
    'EOG_E1_A2', 'EOG_E2_A1', 'EEG_C3_A2', 'EEG_C4_A1',
]

# Value to fill the missing SN (snore) channel with, AFTER z-scoring the
# other channels (so 0.0 reads as "flat/silent", not an arbitrary offset).
MISSING_CHANNEL_FILL = 0.0

# Bipolar derivations needed beyond what's already a direct channel.
# (target_standard_name, positive_lead, [reference_lead(s)])
_BIPOLAR_DERIVATIONS = [
    ('c3-m2', 'c3', ['m2']),
    ('c4-m1', 'c4', ['m1']),
    ('e1-m2', 'e1', ['m2']),
    ('e2-m1', 'e2', ['m1']),
    ('chin1-chin2', 'chin 1', ['chin 2']),
]

# standard OSF channel name -> list of candidate standardized names in
# this project's channel_table.csv, in preference order.
_CANDIDATES = {
    'ECG':        ['ecg'],
    'EMG_Chin':   ['chin1-chin2', 'chin'],
    'EMG_LLeg':   ['lat'],
    'EMG_RLeg':   ['rat'],
    'ABD':        ['abd'],
    'THX':        ['chest'],
    'NP':         ['ptaf', 'airflow'],   # ptaf preferred; airflow (thermal) as fallback
    'SN':         [],                    # confirmed absent from this dataset — see header
    # No standalone m1/m2 channel exists anywhere in this dataset's raw
    # EDFs (confirmed 2026-07-27: EEG arrives already bipolar-referenced
    # -- c3-m2, c4-m1, etc. -- with no separate mastoid channel recorded).
    # True EOG_E1_A2/E2_A1 is therefore NOT RECOVERABLE from this data.
    # Falling back to raw monopolar e1/e2 as the closest available proxy
    # -- flagged in `approximated_channels` below, not silently accepted
    # as the real spec.
    'EOG_E1_A2':  ['e1-m2', 'e1'],
    'EOG_E2_A1':  ['e2-m1', 'e2'],
    'EEG_C3_A2':  ['c3-m2'],
    'EEG_C4_A1':  ['c4-m1'],
}


def _standardize_channels(phys_data, phys_fs, csv_path):
    original_labels = list(phys_data.keys())
    rename_rules = load_rename_rules(os.path.abspath(csv_path))
    rename_map, cols_to_drop = standardize_channel_names_rename_only(
        original_labels, rename_rules)

    channels = {}
    fs_map = {}
    for old_label, data in phys_data.items():
        if old_label in cols_to_drop:
            continue
        new_label = rename_map.get(old_label, old_label.lower())
        channels[new_label] = data
        if old_label in phys_fs:
            fs_map[new_label] = phys_fs[old_label]

    for target, pos, neg_list in _BIPOLAR_DERIVATIONS:
        if target in channels or pos not in channels:
            continue
        if not all(n in channels for n in neg_list):
            continue
        ref = (channels[neg_list[0]] if len(neg_list) == 1
               else tuple(channels[n] for n in neg_list))
        derived = derive_bipolar_signal(channels[pos], ref)
        if derived is not None:
            channels[target] = derived
            fs_map[target] = fs_map.get(pos, 200.0)

    return channels, fs_map


def _resample_to_target(sig, native_fs, target_fs=TARGET_FS):
    """Rational-factor polyphase resample. native_fs approximated to the
    nearest /1000 fraction — fine for the sampling rates seen in PSG EDFs
    (typically integer or .5 Hz)."""
    if sig is None or len(sig) < 2:
        return None
    frac = Fraction(target_fs / float(native_fs)).limit_denominator(1000)
    up, down = frac.numerator, frac.denominator
    if up == down:
        return np.asarray(sig, dtype=np.float64)
    return resample_poly(np.asarray(sig, dtype=np.float64), up, down)


def _zscore(sig):
    if sig is None or len(sig) == 0:
        return sig
    mu = float(np.mean(sig))
    sd = float(np.std(sig))
    if sd < 1e-12:
        return np.zeros_like(sig)
    return (sig - mu) / sd


def build_osf_epoch_tensor(phys_data, phys_fs, algo_data,
                            csv_path=DEFAULT_CSV_PATH,
                            max_epochs=None):
    """
    Build the [n_epochs, 12, 1920] tensor OSF-Base expects, plus a quality
    report dict. Pure numpy — caller is responsible for handing this to
    a torch model.

    Returns (epoch_tensor, report) where:
        epoch_tensor: np.ndarray[float32] of shape (n_epochs, 12, 1920),
            or None if extraction failed (missing physio/CAISR, or every
            candidate channel absent).
        report: {
            'missing_channels': [OSF channel names filled with
                                  MISSING_CHANNEL_FILL, e.g. always
                                  includes 'SN' for this dataset],
            'n_epochs': int,
            'n_valid_stage_epochs': int,   # epochs with a real stage code
        }

    Epoch count is driven by CAISR's stage_caisr array (30s resolution,
    same granularity OSF expects) — one epoch per stage code, valid or
    not. Epochs beyond the shortest resampled channel's length are
    dropped rather than zero-padded, to avoid feeding the model a
    partially-fabricated final epoch.

    NaN fallback: returns (None, report) if physio/CAISR missing, or if
    NEITHER EEG channel (C3-A2, C4-A1) nor ECG is available — those are
    the two modalities every OSF downstream task in the paper actually
    depends on; without at least one of them this isn't a meaningful
    embedding request, not just an incomplete one.
    """
    report = {'missing_channels': [], 'approximated_channels': [],
               'n_epochs': 0, 'n_valid_stage_epochs': 0, 'stage_codes': None}

    if not phys_data or not algo_data:
        return None, report

    stages_raw = algo_data.get('stage_caisr', np.array([]))
    if len(stages_raw) < 1:
        return None, report
    report['n_valid_stage_epochs'] = int(np.sum(stages_raw < 9.0))

    channels, fs_map = _standardize_channels(phys_data, phys_fs, csv_path)

    # Resolve + resample + z-score each of the 12 target channels.
    resampled = {}
    for osf_name in OSF_CHANNEL_ORDER:
        candidates = _CANDIDATES[osf_name]
        sig, fs = None, None
        for idx, c in enumerate(candidates):
            if c in channels and channels[c] is not None:
                sig = channels[c]
                fs = fs_map.get(c, 200.0)
                if idx > 0:
                    report['approximated_channels'].append(f'{osf_name}<-{c}')
                break
        if sig is None:
            report['missing_channels'].append(osf_name)
            resampled[osf_name] = None
            continue
        rs = _resample_to_target(sig, fs)
        resampled[osf_name] = _zscore(rs) if rs is not None else None
        if rs is None:
            report['missing_channels'].append(osf_name)

    have_eeg = (resampled['EEG_C3_A2'] is not None or
                resampled['EEG_C4_A1'] is not None)
    have_ecg = resampled['ECG'] is not None
    if not have_eeg and not have_ecg:
        return None, report

    # Epoch length = shortest available real (non-missing) channel,
    # capped by the CAISR stage count so we never epoch past what's
    # actually staged.
    real_lengths = [len(v) for v in resampled.values() if v is not None]
    if not real_lengths:
        return None, report
    n_epochs_by_signal = min(real_lengths) // SAMPLES_PER_EPOCH
    n_epochs = min(n_epochs_by_signal, len(stages_raw))
    if max_epochs is not None:
        n_epochs = min(n_epochs, max_epochs)
    if n_epochs < 1:
        return None, report

    total_samples = n_epochs * SAMPLES_PER_EPOCH
    tensor = np.full((n_epochs, N_CHANNELS, SAMPLES_PER_EPOCH),
                      MISSING_CHANNEL_FILL, dtype=np.float32)

    for ch_idx, osf_name in enumerate(OSF_CHANNEL_ORDER):
        sig = resampled[osf_name]
        if sig is None:
            continue   # stays at MISSING_CHANNEL_FILL for every epoch
        sig = sig[:total_samples]
        tensor[:, ch_idx, :] = sig.reshape(n_epochs, SAMPLES_PER_EPOCH)

    report['n_epochs'] = n_epochs
    # Per-epoch CAISR stage code aligned to exactly the epochs in `tensor`
    # (2026-07-30, Option B arm 1). Additive — every prior caller ignores
    # it and the tensor is unchanged. Enables stage-conditional pooling
    # downstream (stage_pool_cls) without re-touching the EDFs.
    report['stage_codes'] = np.asarray(stages_raw[:n_epochs])
    return tensor, report


# --- Stage-conditional pooling (Option B, arm 1 — 2026-07-30) ---------------
# CAISR stage_caisr encoding, per the Challenge data spec:
#   1 = N3, 2 = N2, 3 = N1, 4 = REM, 5 = Wake, 9 = Unavailable
# Row order of the pooled output is FIXED and must match the eval
# (tools/evaluate_osf_stage_pooled.py). Do not reorder.
STAGE_ORDER = [('N3', 1), ('N2', 2), ('N1', 3), ('REM', 4), ('Wake', 5)]
N_STAGES = len(STAGE_ORDER)


def stage_pool_cls(cls_seq, stage_codes):
    """
    Stage-conditional mean-pool of a per-epoch CLS embedding sequence.

    cls_seq:     [n_epochs, D] per-epoch CLS embeddings (D=768 for OSF-Base).
    stage_codes: [n_epochs] CAISR stage code per epoch (1/2/3/4/5; 9=unavail).

    Returns (pooled, counts):
      pooled: [N_STAGES, D] float32 — mean CLS within each stage, row order
              STAGE_ORDER (N3, N2, N1, REM, Wake). A stage with zero epochs
              for this patient yields an all-NaN row; the downstream eval
              imputes it on the TRAINING fold only, same discipline as any
              other missing feature.
      counts: [N_STAGES] int — epochs contributing to each stage row.

    Pure numpy, torch-free — unit-testable without a model or a GPU. This is
    the ONLY new signal-processing step for Option B arm 1: the whole-night
    mean (the CLOSED 2026-07-29 result) collapses every stage together; this
    keeps them separate, on the hypothesis that neurodegeneration-relevant
    EEG structure lives in specific stages (N3 slow-wave, REM) and is washed
    out by a whole-night average. Epochs with stage 9 (unavailable) are
    excluded from every bucket.
    """
    cls_seq = np.asarray(cls_seq, dtype=np.float64)
    stage_codes = np.asarray(stage_codes).astype(float)
    D = cls_seq.shape[1]
    pooled = np.full((N_STAGES, D), np.nan, dtype=np.float32)
    counts = np.zeros(N_STAGES, dtype=int)
    n = min(len(cls_seq), len(stage_codes))
    cls_seq, stage_codes = cls_seq[:n], stage_codes[:n]
    for i, (_, code) in enumerate(STAGE_ORDER):
        mask = stage_codes == code
        c = int(mask.sum())
        counts[i] = c
        if c > 0:
            pooled[i] = cls_seq[mask].mean(axis=0).astype(np.float32)
    return pooled, counts


if __name__ == '__main__':
    # T0-style synthetic self-test — shapes and NaN-fallback only, no
    # real data. Run directly: python features/embedding_extraction.py
    rng = np.random.default_rng(0)
    n_samples_200hz = 200 * 60 * 60 * 4   # 4 hours @ 200Hz

    phys_data = {
        'c3': rng.normal(size=n_samples_200hz),
        'm2': rng.normal(size=n_samples_200hz),
        'c4': rng.normal(size=n_samples_200hz),
        'm1': rng.normal(size=n_samples_200hz),
        'e1': rng.normal(size=n_samples_200hz),
        'e2': rng.normal(size=n_samples_200hz),
        'chin 1': rng.normal(size=n_samples_200hz),
        'chin 2': rng.normal(size=n_samples_200hz),
        'lat': rng.normal(size=n_samples_200hz),
        'rat': rng.normal(size=n_samples_200hz),
        'ecg': rng.normal(size=n_samples_200hz),
        'abd': rng.normal(size=n_samples_200hz),
        'chest': rng.normal(size=n_samples_200hz),
        'ptaf': rng.normal(size=n_samples_200hz),
    }
    phys_fs = {k: 200.0 for k in phys_data}
    n_epochs_stage = int(n_samples_200hz / 200.0 / 30.0)
    algo_data = {'stage_caisr': rng.integers(1, 6, size=n_epochs_stage).astype(float)}

    tensor, report = build_osf_epoch_tensor(phys_data, phys_fs, algo_data)
    assert tensor is not None, "synthetic full-channel case should not be None"
    assert tensor.shape[1:] == (N_CHANNELS, SAMPLES_PER_EPOCH), tensor.shape
    assert report['missing_channels'] == ['SN'], report['missing_channels']
    print(f"OK — synthetic tensor shape {tensor.shape}, "
          f"missing_channels={report['missing_channels']}")

    # Missing-everything case
    tensor_none, report_none = build_osf_epoch_tensor({}, {}, {})
    assert tensor_none is None
    print("OK — empty input correctly returns None")

    # stage_codes now aligned to the tensor's epoch count
    assert report['stage_codes'] is not None
    assert len(report['stage_codes']) == tensor.shape[0], \
        (len(report['stage_codes']), tensor.shape[0])
    print(f"OK — report['stage_codes'] aligned, len={len(report['stage_codes'])}")

    # stage_pool_cls: shapes, row order, NaN for absent stage
    n_ep = 100
    cls = rng.normal(size=(n_ep, 768))
    codes = np.array(([1] * 40) + ([2] * 30) + ([4] * 30))  # N3, N2, REM only
    pooled, counts = stage_pool_cls(cls, codes)
    assert pooled.shape == (N_STAGES, 768), pooled.shape
    assert list(counts) == [40, 30, 0, 30, 0], list(counts)          # N3,N2,N1,REM,Wake
    assert np.isnan(pooled[2]).all() and np.isnan(pooled[4]).all()   # N1, Wake absent
    assert np.allclose(pooled[0], cls[:40].mean(0), atol=1e-4)       # N3 row correct
    print(f"OK — stage_pool_cls shapes/counts/NaN correct, counts={list(counts)}")
