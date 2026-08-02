#!/usr/bin/env python
# Team Narnia — PhysioNet Challenge 2026
# OSF-MIL fine-tune SUBMISSION team_code (arm 3, 2026-08-01).
#
# ⚠ This file was the Entry-8 logreg submission; it is now the OSF-MIL
#   fine-tune submission. Entry 8 is preserved in git history (revert with
#   `git checkout <entry8-commit-or-branch> -- team_code.py requirements.txt`).
#   Entry 8 remains the banked standing submission and the fallback the
#   test-set entry can always be chosen as — this OSF entry is a validation
#   PROBE (best-of scoring: it cannot dislodge Entry 8).
#
# THIN entry point: all model logic lives in features/osf_mil.py (single source
# of truth shared with tools/finetune_osf_mil.py and tools/evaluate_osf_ft.py),
# the same thin-team_code / referenced-feature-module structure Entry 8 used
# with features/pipeline.py.
#
# requirements.txt: UNCOMMENT the torch/einops lines for this submission.
# Bundled artifacts expected in the Docker build context:
#   OSF-Base/                    OSF code package (osf.backbone.vit1d_cls)
#   OSF-Base/osf_backbone.pth    backbone init + pretrained weights (341 MB)
#   models/osf_ft_best.pt        the fine-tuned checkpoint (341 MB)
# Request GPU. Full assembly checklist: SUBMISSION_OSF_FT.md.
#
# Model shipped: conservative MIL fine-tune (last block + top-level norm +
# gated-attention head). Clean full-negative held-out I0006 = 0.6910 (+0.0159
# vs frozen Entry-8's 0.6751). The bundled checkpoint was trained on S0001+
# I0002 with I0006 HELD OUT — a valid FIRST validation-probe lower bound;
# re-fine-tune on all three sites for the entry you'd lock in for the test set.
#
# DECISIONS baked as defaults (see SUBMISSION_OSF_FT.md): ship-weights (no
# in-harness fine-tune — verify the rules permit shipping trained weights);
# THRESHOLD nominal (AUROC, the ranking metric that decides the winner, is
# threshold-independent).

import os
from features.osf_mil import (
    package_checkpoint, load_osf_mil_model, run_osf_mil_record,
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_OSF_REPO   = os.environ.get('NARNIA_OSF_REPO', os.path.join(SCRIPT_DIR, 'OSF-Base'))
_OSF_BB     = os.environ.get('NARNIA_OSF_BACKBONE', os.path.join(_OSF_REPO, 'osf_backbone.pth'))
_FT_CKPT    = os.environ.get('NARNIA_OSF_FT', os.path.join(SCRIPT_DIR, 'models', 'osf_ft_best.pt'))
_THRESHOLD  = float(os.environ.get('NARNIA_OSF_THRESHOLD', '0.5'))
_MAX_EPOCHS = int(os.environ.get('NARNIA_OSF_MAX_EPOCHS', '128'))
_UNFREEZE   = int(os.environ.get('NARNIA_OSF_UNFREEZE', '1'))


def train_model(data_folder, model_folder, verbose):
    # Ship-weights: package the offline-fine-tuned checkpoint + serving config.
    package_checkpoint(_FT_CKPT, model_folder, {
        'osf_repo_path': _OSF_REPO, 'osf_backbone_pth': _OSF_BB,
        'unfreeze_last_blocks': _UNFREEZE, 'threshold': _THRESHOLD,
        'max_epochs': _MAX_EPOCHS,
    })
    if verbose:
        print(f'Packaged fine-tuned checkpoint into {model_folder} (ship-weights).')


def load_model(model_folder, verbose):
    return load_osf_mil_model(model_folder, verbose=verbose)


def run_model(model, record, data_folder, verbose):
    return run_osf_mil_record(model, record, data_folder, verbose=verbose)


def save_model(model_folder, model):
    pass  # train_model already writes osf_ft.pt + osf_ft_config.json
