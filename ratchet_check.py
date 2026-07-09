#!/usr/bin/env python

"""
ratchet_check.py
Team Narnia — PhysioNet Challenge 2026

Answers one question before you submit: "is this LOSO result actually worse
than what we've already confirmed, or is it inside the noise band we already
characterized?" Motivated directly by the Entry 1-3 leaderboard sequence
(0.624 -> 0.616 -> 0.606), which LOOKED like a monotonic decline but was
confirmed via Hanley-McNeil SE to be ~0.12-0.40 sigma per step — i.e. noise,
not regression. This script automates that same check instead of re-deriving
it by hand every time.

One-directional ratchet: only ever fails on a REGRESSION beyond the sigma
threshold. An improvement of any size always passes — this is a gate against
backsliding, not a two-sided significance test.

Baselines live in ratchet_baselines.json, hand-curated (see that file's
_readme). This script never writes to it — promoting a new result to a
baseline is a deliberate decision, not something a script should do for you.

Usage (standalone):
    python ratchet_check.py --loso-results loso_results.csv --baseline small_entry3

    # If your loso_results.csv uses different column names than the
    # defaults below, point at them explicitly:
    python ratchet_check.py --loso-results loso_results.csv --baseline small_entry3 \\
        --auroc-col age_cond_auroc --n-pos-col n_pos_test --n-neg-col n_neg_test

Also importable — see check_ratchet() for use from check_submission_files.py.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Candidate column names this script will try, in order, for each required
# quantity — loso_cv.py's actual schema hasn't been pinned down here, so
# guess defensively and FAIL LOUDLY (listing real columns found) rather than
# silently mis-reading the wrong column.
_AUROC_COL_CANDIDATES = ['age_cond_auroc', 'age_conditioned_auroc', 'auroc_age', 'auroc']
_N_POS_COL_CANDIDATES = ['n_pos', 'n_pos_test', 'num_pos', 'n_positive']
_N_NEG_COL_CANDIDATES = ['n_neg', 'n_neg_test', 'num_neg', 'n_negative']


def _resolve_column(df, candidates, explicit, quantity_name):
    if explicit is not None:
        if explicit not in df.columns:
            raise ValueError(
                f"--{quantity_name}-col '{explicit}' not found in loso_results.csv. "
                f"Actual columns: {list(df.columns)}")
        return explicit
    for cand in candidates:
        if cand in df.columns:
            return cand
    raise ValueError(
        f"Could not find a column for '{quantity_name}' in loso_results.csv. "
        f"Tried: {candidates}. Actual columns: {list(df.columns)}. "
        f"Pass --{quantity_name}-col explicitly to point at the right one.")


def hanley_mcneil_se(auroc, n_pos, n_neg):
    """
    Standard error of an AUROC estimate (Hanley & McNeil, 1982).
    Same formula already used by hand in learning_log.md's 2026-07-01
    entry to confirm the Entry 1-3 leaderboard decline was noise.
    """
    if n_pos <= 1 or n_neg <= 1:
        raise ValueError(
            f"Need n_pos > 1 and n_neg > 1 for a Hanley-McNeil SE "
            f"(got n_pos={n_pos}, n_neg={n_neg}).")

    q1 = auroc / (2 - auroc)
    q2 = (2 * auroc ** 2) / (1 + auroc)

    variance = (
        auroc * (1 - auroc)
        + (n_pos - 1) * (q1 - auroc ** 2)
        + (n_neg - 1) * (q2 - auroc ** 2)
    ) / (n_pos * n_neg)

    return float(np.sqrt(max(variance, 0.0)))


def compare_auroc(candidate_auroc, candidate_n_pos, candidate_n_neg,
                   baseline, sigma_threshold=1.0):
    """
    Compares a candidate LOSO AUROC against a baseline using pooled
    Hanley-McNeil SEs (treated as independent — an approximation, same one
    already accepted in the 2026-07-01 by-hand analysis).

    Returns a dict with the verdict. Only ever fails on regression:
    candidate < baseline AND the gap exceeds sigma_threshold * SE_diff.
    An improving candidate always passes regardless of magnitude.
    """
    baseline_auroc = baseline['age_cond_auroc']
    baseline_n_pos = baseline['n_pos']
    baseline_n_neg = baseline['n_neg']

    se_candidate = hanley_mcneil_se(candidate_auroc, candidate_n_pos, candidate_n_neg)
    se_baseline = hanley_mcneil_se(baseline_auroc, baseline_n_pos, baseline_n_neg)
    se_diff = float(np.sqrt(se_candidate ** 2 + se_baseline ** 2))

    delta = candidate_auroc - baseline_auroc
    sigmas = delta / se_diff if se_diff > 0 else float('inf')

    if delta >= 0:
        verdict = 'PASS (improvement)'
        is_regression = False
    elif sigmas >= -sigma_threshold:
        verdict = 'PASS (within noise band)'
        is_regression = False
    else:
        verdict = 'FAIL (regression exceeds noise band)'
        is_regression = True

    return {
        'metric': 'age_cond_auroc',
        'candidate': candidate_auroc,
        'baseline': baseline_auroc,
        'delta': delta,
        'se_diff': se_diff,
        'sigmas': sigmas,
        'sigma_threshold': sigma_threshold,
        'verdict': verdict,
        'is_regression': is_regression,
    }


def compare_reward(candidate_reward, baseline, max_pct_drop=0.15):
    """
    Reward has no clean analytic SE — it's a prevalence-weighted score per
    patient (compute_reward in evaluate_model.py), not a simple binomial
    rate, so Hanley-McNeil doesn't apply. This is a blunt percentage-drop
    heuristic, explicitly informational rather than a rigorous statistical
    gate. A proper version would bootstrap over loso_probabilities.csv
    (resample patients with replacement, recompute reward each time) — not
    implemented here; flag if you want that added.
    """
    baseline_reward = baseline['reward']
    if baseline_reward == 0:
        pct_drop = float('inf') if candidate_reward < 0 else 0.0
    else:
        pct_drop = (baseline_reward - candidate_reward) / abs(baseline_reward)

    if candidate_reward >= baseline_reward:
        verdict = 'PASS (improvement)'
        is_regression = False
    elif pct_drop <= max_pct_drop:
        verdict = 'PASS (within tolerance, NOT statistically rigorous)'
        is_regression = False
    else:
        verdict = 'WARN (drop exceeds tolerance, NOT statistically rigorous)'
        is_regression = True

    return {
        'metric': 'reward',
        'candidate': candidate_reward,
        'baseline': baseline_reward,
        'pct_drop': pct_drop,
        'max_pct_drop': max_pct_drop,
        'verdict': verdict,
        'is_regression': is_regression,
        'note': 'Percentage-drop heuristic only — reward has no analytic SE. '
                'Not as rigorous as the AUROC check.',
    }


def load_baselines(path='ratchet_baselines.json'):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Baseline file not found: {path}. Run from the repo root, or "
            f"pass --baselines-file to point at it explicitly.")
    with open(path) as f:
        data = json.load(f)
    data.pop('_readme', None)
    return data


def summarize_loso_results(csv_path, auroc_col=None, n_pos_col=None, n_neg_col=None):
    """
    Pools per-fold LOSO results into a single (auroc, n_pos, n_neg) triple.
    auroc is the pooled-weighted mean across folds (weighted by each fold's
    n_pos + n_neg, i.e. test-set size) — NOT a simple average of per-fold
    AUROCs, since folds have very different sizes (e.g. I0006 n=20 pos vs
    S0001 n=56 pos at small scale). n_pos / n_neg are summed across folds.
    """
    df = pd.read_csv(csv_path)

    auroc_col = _resolve_column(df, _AUROC_COL_CANDIDATES, auroc_col, 'auroc')
    n_pos_col = _resolve_column(df, _N_POS_COL_CANDIDATES, n_pos_col, 'n-pos')
    n_neg_col = _resolve_column(df, _N_NEG_COL_CANDIDATES, n_neg_col, 'n-neg')

    n_pos_total = int(df[n_pos_col].sum())
    n_neg_total = int(df[n_neg_col].sum())

    fold_sizes = df[n_pos_col] + df[n_neg_col]
    pooled_auroc = float(np.average(df[auroc_col], weights=fold_sizes))

    return {
        'age_cond_auroc': pooled_auroc,
        'n_pos': n_pos_total,
        'n_neg': n_neg_total,
        'per_fold_auroc': dict(zip(df.index.astype(str), df[auroc_col])),
    }


def check_ratchet(loso_results_csv, baseline_key, baselines_file='ratchet_baselines.json',
                   candidate_reward=None, sigma_threshold=1.0, max_reward_pct_drop=0.15,
                   auroc_col=None, n_pos_col=None, n_neg_col=None):
    """
    Main entry point — also called from check_submission_files.py.
    Returns (passed: bool, results: list[dict], messages: list[str]).
    """
    messages = []
    baselines = load_baselines(baselines_file)

    if baseline_key not in baselines:
        raise KeyError(
            f"Baseline key '{baseline_key}' not found in {baselines_file}. "
            f"Available: {list(baselines.keys())}")

    baseline = baselines[baseline_key]
    if baseline.get('n_pos') is None or baseline.get('n_neg') is None:
        raise ValueError(
            f"Baseline '{baseline_key}' has no n_pos/n_neg filled in "
            f"(status: {baseline.get('status', 'unknown')}) — cannot compute "
            f"a Hanley-McNeil comparison SE. Fill these in ratchet_baselines.json "
            f"before using this baseline as a gate.")

    candidate = summarize_loso_results(loso_results_csv, auroc_col, n_pos_col, n_neg_col)

    results = [compare_auroc(
        candidate['age_cond_auroc'], candidate['n_pos'], candidate['n_neg'],
        baseline, sigma_threshold=sigma_threshold)]

    if candidate_reward is not None:
        results.append(compare_reward(
            candidate_reward, baseline, max_pct_drop=max_reward_pct_drop))
    else:
        messages.append(
            "No --candidate-reward passed — reward ratchet check skipped. "
            "AUROC-only gate is NOT a full picture; pass reward explicitly "
            "once you have a pooled threshold sweep result.")

    passed = not any(r['is_regression'] for r in results)
    return passed, results, messages


def _print_report(baseline_key, results, messages):
    print(f"Ratchet check against baseline: {baseline_key}\n")
    for r in results:
        print(f"  [{r['metric']}]")
        print(f"    candidate: {r['candidate']:.4f}   baseline: {r['baseline']:.4f}")
        if r['metric'] == 'age_cond_auroc':
            print(f"    delta: {r['delta']:+.4f}   SE_diff: {r['se_diff']:.4f}   "
                  f"sigmas: {r['sigmas']:+.2f} (threshold: {r['sigma_threshold']})")
        else:
            print(f"    pct_drop: {r['pct_drop']:.1%}   tolerance: {r['max_pct_drop']:.1%}")
            print(f"    note: {r['note']}")
        print(f"    verdict: {r['verdict']}\n")

    for m in messages:
        print(f"NOTE: {m}\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--loso-results', required=True,
                         help='Path to loso_cv.py\'s per-fold results CSV.')
    parser.add_argument('--baseline', required=True,
                         help='Key into ratchet_baselines.json (e.g. small_entry3).')
    parser.add_argument('--baselines-file', default='ratchet_baselines.json')
    parser.add_argument('--candidate-reward', type=float, default=None,
                         help='Pooled reward at your chosen threshold, if you have one '
                              '(from a separate threshold-sweep step). Optional.')
    parser.add_argument('--sigma-threshold', type=float, default=1.0,
                         help='AUROC regression must exceed this many pooled SEs to fail. '
                              'Default 1.0 (matches the "noise floor" framing already used '
                              'in learning_log.md).')
    parser.add_argument('--max-reward-pct-drop', type=float, default=0.15,
                         help='Reward regression tolerance as a fraction of baseline. '
                              'Default 0.15 (15%%). NOT statistically derived.')
    parser.add_argument('--auroc-col', default=None)
    parser.add_argument('--n-pos-col', default=None)
    parser.add_argument('--n-neg-col', default=None)
    args = parser.parse_args()

    try:
        passed, results, messages = check_ratchet(
            args.loso_results, args.baseline, args.baselines_file,
            candidate_reward=args.candidate_reward,
            sigma_threshold=args.sigma_threshold,
            max_reward_pct_drop=args.max_reward_pct_drop,
            auroc_col=args.auroc_col, n_pos_col=args.n_pos_col, n_neg_col=args.n_neg_col,
        )
    except (FileNotFoundError, KeyError, ValueError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(2)

    _print_report(args.baseline, results, messages)
    print('PASS' if passed else 'FAIL')
    sys.exit(0 if passed else 1)