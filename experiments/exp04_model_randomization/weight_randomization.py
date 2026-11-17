#!/usr/bin/env python3
"""
Weight Randomization Module for Adebayo Model Parameter Sanity Checks (NeurIPS 2018).
Implements progressive cascading layer randomization from output decoder back to input encoder.
Operates exclusively on in-memory disposable copies, strictly preserving the original checkpoint on disk.
"""
import copy
import torch
import torch.nn as nn
from original.paadu_benchmark_v7 import VolumeAdaptiveRelaxer

def reinit_param(name: str, p: nn.Parameter, seed: int):
    """Initializes an individual parameter tensor according to its architectural role."""
    torch.manual_seed(seed)
    with torch.no_grad():
        if 'pos_embed' in name or 'mask_token' in name or 'cls_token' in name:
            nn.init.trunc_normal_(p, mean=0.0, std=0.02)
        elif p.dim() > 1:
            nn.init.xavier_uniform_(p)
        elif p.dim() == 1 and ('norm' in name.lower() and 'weight' in name.lower()):
            nn.init.ones_(p)
        elif p.dim() == 1 and 'bias' in name.lower():
            nn.init.zeros_(p)
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

    for name, p in disposable_model.named_parameters():
        if any(name.startswith(pre) or pre in name for pre in target_prefixes):
            reinit_param(name, p, seed)

    return disposable_model
