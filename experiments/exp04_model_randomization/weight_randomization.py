#!/usr/bin/env python3
"""
Weight Randomization Module for Adebayo Model Parameter Sanity Checks (NeurIPS 2018).
Implements progressive cascading layer randomization from output decoder back to input encoder.
Operates exclusively on in-memory disposable copies, strictly preserving the original checkpoint on disk.
"""
import copy
import sys
from pathlib import Path
import torch
import torch.nn as nn

_PKG_ROOT = Path(__file__).resolve().parents[2]
_CHARM_PROJ = _PKG_ROOT / 'integrated_project' / 'charm_original' / 'project'
if str(_CHARM_PROJ) not in sys.path:
    sys.path.insert(0, str(_CHARM_PROJ))

from original.paadu_benchmark_v7 import VolumeAdaptiveRelaxer

def reinit_param(name: str, p: nn.Parameter):
    """Initializes an individual parameter tensor according to its architectural role using advancing RNG."""
    with torch.no_grad():
        if 'pos_embed' in name or 'mask_token' in name or 'cls_token' in name:
            nn.init.trunc_normal_(p, mean=0.0, std=0.02)
        elif p.dim() > 1:
            nn.init.xavier_uniform_(p)
        elif p.dim() == 1 and ('norm' in name.lower() and 'weight' in name.lower()):
            nn.init.normal_(p, mean=1.0, std=0.02)
        elif p.dim() == 1 and 'bias' in name.lower():
            nn.init.normal_(p, mean=0.0, std=0.02)
        else:
            nn.init.normal_(p, mean=0.0, std=0.02)

reinit_layer = reinit_param

def build_engine_copy(base_engine, model: nn.Module):
    """Wraps a disposable model copy inside a standalone CHARMEngine instance."""
    from charm_engine import CHARMEngine
    eng = CHARMEngine(base_engine.cfg, base_engine.features)
    eng.mean = base_engine.mean
    eng.std = base_engine.std
    eng.mse_stats = base_engine.mse_stats
    eng.dm_stats = base_engine.dm_stats
    eng.floor = base_engine.floor
    eng.mahal = base_engine.mahal
    eng.original_cfg = base_engine.original_cfg
    eng.model = model
    eng.relaxer = VolumeAdaptiveRelaxer(eng.cfg.hard_z_floor, eng.original_cfg)
    eng.relaxer.set_floor_value(eng.floor, reason='held-out benign calibration')
    return eng

def get_randomized_model_copy(orig_model: nn.Module, level_name: str, seed: int) -> nn.Module:
    """Creates an in-memory disposable copy with specified cascading layer randomization."""
    # Seed the RNG ONCE per model copy so the random generator advances continuously across all tensors
    torch.manual_seed(seed)
    disposable_model = copy.deepcopy(orig_model)
    disposable_model.eval()

    if level_name == 'Level_1_Decoder':
        target_prefixes = ('decoder.',)
    elif level_name == 'Level_2_Perception':
        target_prefixes = ('decoder.', 'perception.', 'graph_layer.')
    elif level_name == 'Level_3_Transformer':
        target_prefixes = ('decoder.', 'perception.', 'graph_layer.', 'transformer_encoder.')
    elif level_name == 'Level_4_PatchEmbed':
        target_prefixes = ('decoder.', 'perception.', 'graph_layer.', 'transformer_encoder.',
                           'patch_embed.', 'norm.', 'pos_embed', 'mask_token')
    elif level_name == 'Level_5_Complete':
        target_prefixes = tuple(name for name, _ in disposable_model.named_parameters())
    else:
        target_prefixes = ()

    orig_dict = dict(orig_model.named_parameters())
    altered_count = 0

    for name, p in disposable_model.named_parameters():
        if any(name.startswith(pre) or pre in name for pre in target_prefixes):
            reinit_param(name, p)
            # Verify actual parameter change
            orig_p = orig_dict[name]
            diff = (p - orig_p).abs().sum().item()
            if p.numel() > 0:
                assert diff > 0.0, f"Critical: Parameter {name} was not altered during {level_name} randomization!"
                altered_count += 1

    return disposable_model


def verify_parameter_changes(orig_model: nn.Module, rand_model: nn.Module, level_name: str) -> dict:
    """Audits and returns exact tensor-level parameter changes between models."""
    orig_params = dict(orig_model.named_parameters())
    rand_params = dict(rand_model.named_parameters())
    changed = []
    unchanged = []

    for name, op in orig_params.items():
        rp = rand_params[name]
        diff = (rp - op).abs().sum().item()
        if diff > 0.0:
            changed.append((name, diff))
        else:
            unchanged.append(name)

    return {
        'level_name': level_name,
        'changed_tensors_count': len(changed),
        'unchanged_tensors_count': len(unchanged),
        'total_tensors': len(orig_params),
        'changed_names': [c[0] for c in changed]
    }
