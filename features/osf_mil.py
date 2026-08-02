#!/usr/bin/env python
# features/osf_mil.py — Team Narnia, canonical OSF-MIL model components.
#
# SINGLE SOURCE OF TRUTH for the fine-tuned OSF multiple-instance-learning
# model, referenced by all of:
#   team_code.py                      (submission: train/load/run)
#   tools/finetune_osf_mil.py         (offline training)
#   tools/evaluate_osf_ft.py          (clean held-out re-eval)
# so the architecture can never silently diverge between how the model is
# TRAINED and how it's SERVED (same shared-module discipline as
# features/pipeline.py's build_logreg_pipeline, which exists precisely because
# a hand-copied duplicate once let the validation harness and submission drift
# apart — see that file's header).
#
# torch is imported LAZILY inside functions on purpose: this module must be
# importable in a torch-free environment (the Entry-8 logreg submission never
# installs torch), and features/__init__.py must NOT import this module at top
# level, so importing `features` stays torch-free.

import json
import os
import re
import shutil
import numpy as np

from helper_code import (
    HEADERS, PHYSIOLOGICAL_DATA_SUBFOLDER, ALGORITHMIC_ANNOTATIONS_SUBFOLDER,
)
from features.embedding_extraction import build_osf_epoch_tensor

EMBED_DIM = 768
FINAL_NORM_STEMS = ('norm', 'fc_norm', 'ln_f', 'head_norm')
_BLOCK_RE = re.compile(r'(?:^|\.)blocks?\.?(\d+)\.')


# ── Architecture (must stay identical between train and serve) ───────────────
def build_mil_head(torch):
    """Ilse et al. 2018 gated-attention MIL pooling + classifier.
    H: [B, K, D] bag of epoch embeddings -> logits [B, 2], attn [B, K]."""
    nn = torch.nn

    class GatedAttentionMIL(nn.Module):
        def __init__(self, dim=EMBED_DIM, att_dim=128, n_classes=2, dropout=0.5):
            super().__init__()
            self.V = nn.Linear(dim, att_dim)
            self.U = nn.Linear(dim, att_dim)
            self.w = nn.Linear(att_dim, 1)
            self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(dim, n_classes))

        def forward(self, H):
            a = self.w(torch.tanh(self.V(H)) * torch.sigmoid(self.U(H)))
            a = torch.softmax(a, dim=1)
            z = (a * H).sum(dim=1)
            return self.head(z), a.squeeze(-1)

    return GatedAttentionMIL


def _block_index(name):
    m = _BLOCK_RE.search(name)
    return int(m.group(1)) if m else None


def apply_conservative_unfreeze(backbone, n_last_blocks=1, unfreeze_final_norm=True,
                                 verbose=False):
    """Freeze the whole ViT, then unfreeze ONLY the last n transformer blocks
    and the final TOP-LEVEL norm. Robust to 'block<i>.' (OSF vit1d_cls) and
    'blocks.<i>.' (timm-style). Top-level-norm match only (name stem in
    FINAL_NORM_STEMS) so per-block norms are NOT swept in. The frozen lower
    stack retains no activations (autograd keeps them only from the first
    trainable op up), so 'last block only' is both least-overfit and
    cheapest-memory. Returns the unfrozen parameter names."""
    for p in backbone.parameters():
        p.requires_grad = False
    idxs = {_block_index(n) for n, _ in backbone.named_parameters()}
    idxs.discard(None)
    if not idxs:
        raise RuntimeError(
            "No transformer-block params found (tried 'block<i>.' and "
            "'blocks.<i>.'). Inspect backbone.named_parameters() and adjust "
            "_BLOCK_RE.")
    max_b = max(idxs)
    targets = set(range(max_b - n_last_blocks + 1, max_b + 1))
    unfrozen = []
    for name, p in backbone.named_parameters():
        k = _block_index(name)
        if (k is not None and k in targets) or (
                unfreeze_final_norm and k is None
                and name.split('.')[0] in FINAL_NORM_STEMS):
            p.requires_grad = True
            unfrozen.append(name)
    if verbose:
        n_tr = sum(p.numel() for p in backbone.parameters() if p.requires_grad)
        n_tot = sum(p.numel() for p in backbone.parameters())
        print(f'  Unfroze {len(unfrozen)} tensors (last {n_last_blocks} block(s) '
              f'{sorted(targets)} + top-level norm): {100*n_tr/n_tot:.2f}% '
              f'trainable ({n_tr:,}/{n_tot:,})')
        for nm in unfrozen:
            print(f'      {nm}')
    return unfrozen


def load_backbone(torch, osf_repo_path, checkpoint_path):
    """Construct vit_base from the OSF backbone .pth's metadata and load its
    pretrained weights. Needs the OSF code package on osf_repo_path."""
    import sys
    if osf_repo_path and osf_repo_path not in sys.path:
        sys.path.insert(0, osf_repo_path)
    from osf.backbone.vit1d_cls import vit_base
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    payload = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    m = payload['metadata']
    backbone = vit_base(num_leads=m['num_leads'], seq_len=m['seq_len'],
                        patch_size=m['patch_size_time'], lead_wise=m['lead_wise'],
                        patch_size_ch=m['patch_size_ch'])
    backbone.load_state_dict(payload['state_dict'])
    backbone.to(device)
    return backbone, device


def encode_bag(torch, backbone, x, use_fp16=True):
    """x: [B, K, 12, 1920] -> per-epoch CLS [B, K, 768]."""
    B, K = x.shape[0], x.shape[1]
    flat = x.reshape(B * K, x.shape[2], x.shape[3])
    amp = use_fp16 and flat.is_cuda
    with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=amp):
        cls, _ = backbone.forward_encoding(flat, return_sequence=False)
    return cls.float().reshape(B, K, EMBED_DIM)


# ── Submission-facing helpers (train_model / load_model / run_model use these)─
def package_checkpoint(bundled_ft_ckpt, model_folder, config):
    """Ship-weights train_model: copy the offline-fine-tuned checkpoint into
    model_folder and write the config load_model needs. config must carry
    osf_repo_path, osf_backbone_pth, unfreeze_last_blocks, threshold,
    max_epochs."""
    os.makedirs(model_folder, exist_ok=True)
    if not os.path.exists(bundled_ft_ckpt):
        raise FileNotFoundError(
            f'Bundled fine-tuned checkpoint not found at {bundled_ft_ckpt}. '
            f'Fine-tune with tools/finetune_osf_mil.py and place it there.')
    shutil.copyfile(bundled_ft_ckpt, os.path.join(model_folder, 'osf_ft.pt'))
    with open(os.path.join(model_folder, 'osf_ft_config.json'), 'w') as f:
        json.dump(config, f)


def load_osf_mil_model(model_folder, verbose=False):
    """Rebuild the architecture and load the fine-tuned weights from
    model_folder (osf_ft.pt + osf_ft_config.json). Returns the serving dict."""
    import torch
    cfg = {}
    cfg_path = os.path.join(model_folder, 'osf_ft_config.json')
    if os.path.exists(cfg_path):
        cfg = json.load(open(cfg_path))
    backbone, device = load_backbone(torch, cfg['osf_repo_path'],
                                     cfg['osf_backbone_pth'])
    apply_conservative_unfreeze(backbone, cfg.get('unfreeze_last_blocks', 1))
    mil = build_mil_head(torch)().to(device)
    ck = torch.load(os.path.join(model_folder, 'osf_ft.pt'),
                    map_location=device, weights_only=False)
    backbone.load_state_dict(ck['backbone'])
    mil.load_state_dict(ck['mil'])
    backbone.eval(); mil.eval()
    if verbose:
        print(f'Loaded fine-tuned OSF-MIL model (device {device}).')
    return {'backbone': backbone, 'mil': mil, 'device': device,
            'threshold': cfg.get('threshold', 0.5),
            'max_epochs': cfg.get('max_epochs', 128)}


def run_osf_mil_record(model, record, data_folder, verbose=False):
    """Inference for one record -> (binary_output, probability_output). Never
    raises on a bad record — a single exception can fail an entire submission;
    a failed record returns (0, 0.0) and just ranks low under AUROC."""
    import torch
    backbone, mil, device = model['backbone'], model['mil'], model['device']
    threshold, max_epochs = model['threshold'], model['max_epochs']

    patient_id = record[HEADERS['bids_folder']]
    site_id    = record[HEADERS['site_id']]
    session_id = record[HEADERS['session_id']]
    phys_file = os.path.join(data_folder, PHYSIOLOGICAL_DATA_SUBFOLDER,
                             site_id, f'{patient_id}_ses-{session_id}.edf')
    algo_file = os.path.join(data_folder, ALGORITHMIC_ANNOTATIONS_SUBFOLDER,
                             site_id, f'{patient_id}_ses-{session_id}_caisr_annotations.edf')
    if not (os.path.exists(phys_file) and os.path.exists(algo_file)):
        return 0, 0.0
    try:
        from helper_code import load_signal_data
        phys_data, phys_fs = load_signal_data(phys_file)
        algo_data, _ = load_signal_data(algo_file)
        tensor, _ = build_osf_epoch_tensor(phys_data, phys_fs, algo_data,
                                           max_epochs=max_epochs)
        del phys_data, algo_data
        if tensor is None:
            return 0, 0.0
        with torch.no_grad():
            x = torch.from_numpy(tensor[None]).float().to(device)
            logits, _ = mil(encode_bag(torch, backbone, x, use_fp16=True))
            proba = float(torch.softmax(logits, 1)[0, 1].cpu())
    except Exception as e:
        if verbose:
            print(f'  ! run_model fallback for {patient_id}: {e}')
        return 0, 0.0
    return int(proba > threshold), proba
