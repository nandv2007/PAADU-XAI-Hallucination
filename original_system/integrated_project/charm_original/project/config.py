from dataclasses import dataclass

@dataclass(frozen=True)
class Phase1Config:
    # Defaults mirror the original CHARM paper; all can be overridden.
    seq_len: int = 10
    num_nodes: int = 1
    embed_dim: int = 64
    mask_ratio: float = 0.40
    w_recon: float = 0.20
    w_mahal: float = 0.80
    use_gnn: bool = False  # paper's primary configuration
    epochs: int = 30
    batch_size: int = 32
    learning_rate: float = 0.001
    benign_train_fraction: float = 0.7
    threshold_percentile: float = 99.0
    noise_floor_std: float = 0.015
    covariance_ridge: float = 1e-6
    hard_z_floor: float = 2.5
    volume_adaptive: bool = True
    soft_triggers: bool = True
    use_drift: bool = True
    attribution_top_k: int = 5
    vol_low_rows: int = 5000
    vol_high_rows: int = 100000
    vol_low_mult: float = 1.0
    vol_high_mult: float = 1.2
    vol_max_mult: float = 1.1
    base_leak: float = 0.06
    bucket_limit: float = 3.2
    ema_alpha: float = 0.05
    ema_multiplier: float = 6.5
    min_alert_severity: float = 0.05
    double_soft_z: float = 2.5
    double_soft_relax: float = 0.60
    latent_only_z: float = 3.0
    latent_only_relax: float = 0.70
    seed: int = 42

    def __post_init__(self):
        if not (0 < self.benign_train_fraction < 1): raise ValueError('train fraction must be in (0,1)')
        if not (0 <= self.mask_ratio < 1): raise ValueError('mask_ratio must be in [0,1)')
        if abs(self.w_recon+self.w_mahal-1)>1e-8 or min(self.w_recon,self.w_mahal)<0: raise ValueError('weights must be nonnegative and sum to 1')
        if not (0 < self.threshold_percentile < 100): raise ValueError('percentile must be in (0,100)')
        if min(self.seq_len,self.num_nodes,self.embed_dim,self.epochs,self.batch_size,self.attribution_top_k)<1: raise ValueError('positive integer parameters required')
