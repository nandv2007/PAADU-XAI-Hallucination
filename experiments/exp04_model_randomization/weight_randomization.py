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

def reinit_layer(module: nn.Module, seed: int):
    """Reinitializes parameters of a PyTorch module using standard initialization."""
    torch.manual_seed(seed)
    for p in module.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
        else:
            nn.init.zeros_(p)

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
        reinit_layer(disposable_model.decoder, seed)

    elif level_name == 'Level_2_Perception':
        reinit_layer(disposable_model.decoder, seed)
        reinit_layer(disposable_model.perception, seed)
        reinit_layer(disposable_model.graph_layer, seed)

    elif level_name == 'Level_3_Transformer':
        reinit_layer(disposable_model.decoder, seed)
        reinit_layer(disposable_model.perception, seed)
        reinit_layer(disposable_model.graph_layer, seed)
        reinit_layer(disposable_model.transformer_encoder, seed)

    elif level_name == 'Level_4_PatchEmbed':
        reinit_layer(disposable_model.decoder, seed)
        reinit_layer(disposable_model.perception, seed)
        reinit_layer(disposable_model.graph_layer, seed)
        reinit_layer(disposable_model.transformer_encoder, seed)
        reinit_layer(disposable_model.patch_embed, seed)
        reinit_layer(disposable_model.norm, seed)

    elif level_name == 'Level_5_Complete':
        for m in disposable_model.modules():
            if isinstance(m, (nn.Linear, nn.Conv1d, nn.Embedding, nn.LayerNorm)):
                reinit_layer(m, seed)

    return disposable_model
