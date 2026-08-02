# features/spo2_desaturation.py
# SpO2 desaturation depth — cross-signal feature, planned since v1
# (FEATURES.md "Week 2-3: Cross-signal features", never implemented).
# Reviewed 2026-07-29 against every closed test in this project's logs
# (spectral, HRV, NF1/NF2, wake fragmentation, REM latency, no-arousal-
# entropy, age-residual rescan, domain alignment, OSF-Base embeddings) —
# none touch event-conditioned oxygen desaturation. Genuinely untested.
#
# Requires BOTH physiological EDF (spo2 channel) and CAISR annotations
# (resp_caisr, for real event timing) — a true cross-signal feature, not
# derivable from either file alone. Standardized channel name confirmed
# directly against channel_table.csv: 'spo2' (first alias in the
# spo2;sao2;osat;... row).
#
# Mechanism: for each apnea/hypopnea event, the oxygen desaturation nadir
# typically lags event termination by seconds to ~1-2 minutes (respiratory
# and circulatory transit delay) — this is more clinically specific than
# the whole-night SpO2 summary statistics already in physiological.py's
# (unused, dropped-at-Entry-2) SpO2 Hjorth block, which can't distinguish
# "many small dips" from "few deep ones" the way an event-anchored nadir
# search can.
#
# NOT wired into team_code.py's extraction — standalone, ablation-testable
# first, same discipline as every other candidate function in this
# project (spectral.py, caisr_enriched.py's unwired candidates).

import numpy as np
import os
import sys

_FEATURES_DIR    = os.path.dirname(os.path.abspath(__file__))
_REPO_DIR        = os.path.dirname(_FEATURES_DIR)
DEFAULT_CSV_PATH = os.path.join(_REPO_DIR, 'channel_table.csv')

if _REPO_DIR not in sys.path:
    sys.path.insert(0, _REPO_DIR)

from helper_code import (
    load_rename_rules, standardize_channel_names_rename_only
)

N_SPO2_DESAT_FEATURES = 1

# resp_caisr event codes (confirmed against features/caisr_enriched.py):
# 0=none 1=OA 2=CA 3=MA 4=HY 5=RERA. Desaturation-anchored search uses
# the four event types with a real airflow/effort disruption (OA/CA/MA/HY)
# — RERA (5) is arousal-only, not expected to reliably desaturate, and is
# deliberately excluded rather than included "just in case."
_DESAT_EVENT_CODES = (1, 2, 3, 4)

# Nadir search window after event END, seconds. 120s is a conservative
# clinical choice covering typical circulatory delay for the desaturation
# nadir to appear after event termination -- not independently validated
# against this dataset's own event/SpO2 timing relationship. Worth a
# sensitivity check (60s vs 120s vs 180s) if this candidate survives its
# first LOSO screen, not before.
_NADIR_WINDOW_SECONDS = 120.0

# Physiologically implausible SpO2 readings (sensor dropout / artifact) —
# excluded from the nadir search rather than silently biasing it toward
# a spurious near-zero value.
_SPO2_MIN_PLAUSIBLE = 50.0
_SPO2_MAX_PLAUSIBLE = 100.0

# Minimum number of valid (non-artifact-corrupted) events required to
# trust the per-patient average nadir -- same "gate on estimability, not
# duration" discipline as caisr_enriched.py's NF2 (_MIN_AROUSAL_EVENTS).
_MIN_EVENTS = 3


def _get_channel(channels, candidates):
    for c in candidates:
        if c in channels and channels[c] is not None:
            return channels[c]
    return None


def _get_fs(fs_map, candidates, default=1.0):
    for c in candidates:
        if c in fs_map:
            return float(fs_map[c])
    return default


def extract_spo2_desaturation_depth(phys_data, phys_fs, algo_data,
                                     csv_path=DEFAULT_CSV_PATH):
    """
    Mean SpO2 desaturation nadir across all apnea/hypopnea events in the
    recording. For each OA/CA/MA/HY event (from resp_caisr, 1Hz), finds
    the minimum SpO2 reading in the _NADIR_WINDOW_SECONDS after the
    event's END, then averages across all valid events for one
    per-patient value.

    Returns np.ndarray of length N_SPO2_DESAT_FEATURES (1):
        [0] mean SpO2 desaturation nadir (percent, e.g. 91.3) — NOT a
            drop-from-baseline delta, the raw nadir level itself. Lower
            = deeper desaturation = expected higher clinical severity.

    NaN fallback:
        - Missing physiological EDF or CAISR annotations
        - No spo2 channel present in this patient's file
        - No resp_caisr events of the four desaturation-relevant types
        - Fewer than _MIN_EVENTS (3) events survive the artifact filter
          (a patient could have many events but all with corrupted SpO2
          in the search window — gate on what's actually estimable, not
          on the raw event count alone)
    """
    if not phys_data or not algo_data:
        return np.full(N_SPO2_DESAT_FEATURES, float('nan'))

    resp = algo_data.get('resp_caisr', np.array([]))
    if len(resp) == 0:
        return np.full(N_SPO2_DESAT_FEATURES, float('nan'))

    original_labels = list(phys_data.keys())
    rename_rules = load_rename_rules(os.path.abspath(csv_path))
    rename_map, cols_to_drop = standardize_channel_names_rename_only(
        original_labels, rename_rules)

    channels, fs_map = {}, {}
    for old_label, data in phys_data.items():
        if old_label in cols_to_drop:
            continue
        new_label = rename_map.get(old_label, old_label.lower())
        channels[new_label] = data
        if old_label in phys_fs:
            fs_map[new_label] = phys_fs[old_label]

    spo2 = _get_channel(channels, ['spo2'])
    if spo2 is None or len(spo2) == 0:
        return np.full(N_SPO2_DESAT_FEATURES, float('nan'))
    spo2_fs = _get_fs(fs_map, ['spo2'], default=1.0)
    spo2 = np.asarray(spo2, dtype=float)

    # Unit auto-detection: confirmed 2026-07-29 against a real dev_subset
    # patient that this dataset stores spo2 as a FRACTION (0-1, mean 0.924)
    # rather than a percentage (0-100) -- caused every patient to return
    # NaN in the first real test, since the artifact filter (50-100) never
    # matched any value. Detect and rescale rather than hardcoding one
    # convention, in case other sites/source systems in this dataset use
    # the opposite convention (channel_table.csv's spo2 row aliases many
    # different recording-system channel names, which plausibly differ).
    finite = spo2[np.isfinite(spo2)]
    if len(finite) > 0 and np.nanmax(finite) <= 1.5:
        spo2 = spo2 * 100.0

    # Detect event END indices: any of the 4 desat-relevant codes,
    # falling edge (event -> none). resp_caisr is 1Hz, so index == seconds.
    is_event = np.isin(resp, _DESAT_EVENT_CODES).astype(int)
    edges = np.diff(is_event, prepend=0)
    event_end_secs = np.where(edges == -1)[0]  # falling edges = event ends

    if len(event_end_secs) == 0:
        return np.full(N_SPO2_DESAT_FEATURES, float('nan'))

    window_samples = int(round(_NADIR_WINDOW_SECONDS * spo2_fs))
    nadirs = []
    for end_sec in event_end_secs:
        start_idx = int(round(end_sec * spo2_fs))
        end_idx = start_idx + window_samples
        if start_idx >= len(spo2):
            continue
        window = spo2[start_idx:min(end_idx, len(spo2))]
        if len(window) == 0:
            continue
        valid = window[(window >= _SPO2_MIN_PLAUSIBLE) & (window <= _SPO2_MAX_PLAUSIBLE)]
        if len(valid) == 0:
            continue
        nadirs.append(float(np.min(valid)))

    if len(nadirs) < _MIN_EVENTS:
        return np.full(N_SPO2_DESAT_FEATURES, float('nan'))

    return np.array([float(np.mean(nadirs))], dtype=float)


if __name__ == '__main__':
    # T0 synthetic self-test — run directly: python features/spo2_desaturation.py
    rng = np.random.default_rng(0)
    n_seconds = 3600 * 6  # 6-hour recording

    # Synthetic resp_caisr: 20 OA events (code 1), 30s each, spaced out
    resp = np.zeros(n_seconds, dtype=int)
    event_starts = rng.choice(np.arange(0, n_seconds - 200, 200), size=20, replace=False)
    for s in event_starts:
        resp[s:s + 30] = 1

    spo2_fs = 1.0
    spo2 = np.full(n_seconds, 96.0)
    # Inject a real desaturation dip ~20-40s after each event end
    for s in event_starts:
        dip_start = s + 30 + rng.integers(10, 40)
        spo2[dip_start:dip_start + 15] = 88.0

    phys_data = {'spo2': spo2}
    phys_fs = {'spo2': spo2_fs}
    algo_data = {'resp_caisr': resp}

    result = extract_spo2_desaturation_depth(phys_data, phys_fs, algo_data)
    assert not np.isnan(result[0]), "synthetic case with real events should not be NaN"
    assert 85 < result[0] < 92, f"expected nadir near 88, got {result[0]}"
    print(f"OK — synthetic nadir estimate: {result[0]:.2f} (expected ~88.0)")

    # No-events case
    result_none = extract_spo2_desaturation_depth(
        {'spo2': spo2}, phys_fs, {'resp_caisr': np.zeros(n_seconds, dtype=int)})
    assert np.isnan(result_none[0])
    print("OK — no-events case correctly returns NaN")

    # Missing spo2 channel case
    result_missing = extract_spo2_desaturation_depth(
        {'ecg': spo2}, phys_fs, algo_data)
    assert np.isnan(result_missing[0])
    print("OK — missing spo2 channel correctly returns NaN")
