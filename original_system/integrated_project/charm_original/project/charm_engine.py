"""Adapter over the original CHARM classes, not a rewritten transformer.
Accepts already-normalized, numeric, ordered telemetry blocks [B,N,L,F].
The original dataset ingestion remains available in original/paadu_benchmark_v7.py.
"""
from types import SimpleNamespace
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from original.paadu_benchmark_v7 import (SemanticAnomalyMAE, MahalanobisScorer,
    hybrid_array, VolumeAdaptiveRelaxer, PerceptualDriftTracker, seed_everything)
from config import Phase1Config

class CHARMEngine:
    def __init__(self, config: Phase1Config, feature_names):
        self.cfg=config
        self.features=list(feature_names)
        if len(set(self.features)) != len(self.features) or not self.features: raise ValueError('unique feature names required')
        self.model=None
        self.mahal=MahalanobisScorer()
        self.fitted=False

    def _validate(self, blocks):
        x=np.asarray(blocks,dtype=np.float32)
        if x.ndim!=4 or x.shape[1:3]!=(self.cfg.num_nodes,self.cfg.seq_len) or x.shape[3]!=len(self.features):
            raise ValueError(f'expected [blocks,{self.cfg.num_nodes},{self.cfg.seq_len},{len(self.features)}]')
        if not np.isfinite(x).all(): raise ValueError('NaN/Inf traffic data is not allowed')
        return x

    def _adj(self, batch):
        return torch.eye(self.cfg.num_nodes).expand(batch,self.cfg.num_nodes,self.cfg.num_nodes).clone()

    def _raw(self, blocks):
        self.model.eval()
        errs=[]; lat=[]
        with torch.no_grad():
            for start in range(0,len(blocks),self.cfg.batch_size):
                x=torch.tensor(blocks[start:start+self.cfg.batch_size])
                pred,mask,_,z=self.model(x,self._adj(len(x)),return_latent=True)
                target=x.reshape(pred.shape)
                errs.extend(((pred-target)**2).mean(dim=(1,2)).numpy().tolist())
                lat.extend(z)
        return np.asarray(errs),np.asarray(lat)

    def fit(self, benign_blocks, calibration_blocks=None):
        x=self._validate(benign_blocks)
        if calibration_blocks is None:
            if len(x)<4: raise ValueError('need at least four benign blocks for disjoint training/calibration')
            cut=max(2,min(len(x)-2,int(len(x)*self.cfg.benign_train_fraction)))
            train,cal=x[:cut],x[cut:]
        else:
            train=x
            cal=self._validate(calibration_blocks)
            if len(train)<2 or len(cal)<2:
                raise ValueError('need at least two blocks in each split')
        seed_everything(self.cfg.seed)
        self.mean=train.mean(axis=(0,1,2),keepdims=True)
        self.std=train.std(axis=(0,1,2),keepdims=True)
        self.std=np.where(self.std>0,self.std,1.0)
        train=(train-self.mean)/self.std
        cal=(cal-self.mean)/self.std
        self.model=SemanticAnomalyMAE(self.cfg.num_nodes,self.cfg.seq_len,len(self.features),
            embed_dim=self.cfg.embed_dim,mask_ratio=self.cfg.mask_ratio,use_gnn=self.cfg.use_gnn)
        opt=torch.optim.Adam(self.model.parameters(),lr=self.cfg.learning_rate)
        loader=DataLoader(TensorDataset(torch.from_numpy(train)),batch_size=self.cfg.batch_size,shuffle=True)
        for _ in range(self.cfg.epochs):
            self.model.train()
            for (batch,) in loader:
                pred,mask,_=self.model(batch,self._adj(len(batch)))
                target=batch.reshape(pred.shape)
                loss=((pred-target)**2).mean()
                opt.zero_grad();loss.backward();opt.step()
        tr_err,tr_z=self._raw(train)
        self.mahal.accumulate(tr_z);self.mahal.finalize()
        tr_dm=self.mahal.score(tr_z)
        self.mse_stats=(float(tr_err.mean()),float(tr_err.std()))
        self.dm_stats=(float(tr_dm.mean()),float(tr_dm.std()))
        cal_err,cal_z=self._raw(cal)
        cal_dm=self.mahal.score(cal_z)
        self.original_cfg=SimpleNamespace(w_recon=self.cfg.w_recon,w_mahal=self.cfg.w_mahal,
            noise_floor_std=self.cfg.noise_floor_std,hard_z_floor=self.cfg.hard_z_floor,
            vol_low_rows=self.cfg.vol_low_rows,vol_high_rows=self.cfg.vol_high_rows,vol_low_mult=self.cfg.vol_low_mult,vol_high_mult=self.cfg.vol_high_mult,vol_max_mult=self.cfg.vol_max_mult,
            base_leak_default=self.cfg.base_leak,bucket_limit=self.cfg.bucket_limit,ema_alpha=self.cfg.ema_alpha,ema_multiplier=self.cfg.ema_multiplier,
            min_alert_severity=self.cfg.min_alert_severity,drift_requires_current_severity=True,
            double_soft_z=self.cfg.double_soft_z,double_soft_relax=self.cfg.double_soft_relax,latent_only_z=self.cfg.latent_only_z,latent_only_relax=self.cfg.latent_only_relax)
        cal_h,_,_=hybrid_array(cal_err,cal_dm,self.mse_stats,self.dm_stats,self.original_cfg)
        self.floor=max(self.cfg.hard_z_floor,float(np.percentile(cal_h,self.cfg.threshold_percentile)))
        self.relaxer=VolumeAdaptiveRelaxer(self.cfg.hard_z_floor,self.original_cfg)
        self.relaxer.set_floor_value(self.floor,reason='held-out benign calibration')
        self.fitted=True
        return self

    def score(self, blocks, n_rows=None):
        if not self.fitted: raise RuntimeError('fit benign blocks before scoring')
        x=self._validate(blocks)
        norm=(x-self.mean)/self.std
        err,z=self._raw(norm)
        dm=self.mahal.score(z)
        hybrid,zr,zm=hybrid_array(err,dm,self.mse_stats,self.dm_stats,self.original_cfg)
        threshold=(self.relaxer.get_z_threshold(n_rows or len(x)*self.cfg.seq_len*self.cfg.num_nodes,len(x))
                   if self.cfg.volume_adaptive else self.floor)
        drift=PerceptualDriftTracker(base_leak=self.original_cfg.base_leak_default,
            limit=self.original_cfg.bucket_limit,ema_alpha=self.original_cfg.ema_alpha,
            ema_multiplier=self.original_cfg.ema_multiplier,min_alert_severity=self.original_cfg.min_alert_severity,
            drift_requires_current_severity=True)
        out=[]
        for i in range(len(x)):
            causes=[]
            if hybrid[i]>threshold: causes.append('hybrid_threshold')
            if self.cfg.soft_triggers:
                if zr[i]>self.original_cfg.double_soft_z and zm[i]>self.original_cfg.double_soft_z and hybrid[i]>threshold*self.original_cfg.double_soft_relax: causes.append('dual_evidence')
                if zm[i]>self.original_cfg.latent_only_z and hybrid[i]>threshold*self.original_cfg.latent_only_relax: causes.append('latent_evidence')
            if self.cfg.use_drift:
                fired,reason=drift.update(max(0.,float(hybrid[i]-threshold)),float(err[i]))
                if fired: causes.append(reason)
            out.append(dict(index=i,anomaly=bool(causes),causes=causes,hybrid_score=float(hybrid[i]),
                threshold=float(threshold),reconstruction_error=float(err[i]),mahalanobis=float(dm[i]),
                reconstruction_z=float(zr[i]),latent_z=float(zm[i])))
        return out

    def score_one_components(self, normalized_block):
        err,z=self._raw(normalized_block[None])
        dm=self.mahal.score(z)
        h,_,_=hybrid_array(err,dm,self.mse_stats,self.dm_stats,self.original_cfg)
        return float(h[0]),float(err[0]),float(dm[0])
