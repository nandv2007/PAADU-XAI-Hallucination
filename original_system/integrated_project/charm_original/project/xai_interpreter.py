"""Deterministic feature occlusion. Contributions are score differences, not attack labels.
Occlusion baseline is the benign training mean; score changes can be negative.
"""
import numpy as np

class XAIInterpreter:
    def __init__(self, engine): self.engine=engine

    def explain(self, blocks, scores, metadata=None):
        x=self.engine._validate(blocks)
        if len(scores)!=len(x): raise ValueError('score count does not match blocks')
        if metadata is not None and len(metadata)!=len(x): raise ValueError('metadata length mismatch')
        results=[]
        for i,decision in enumerate(scores):
            if not decision['anomaly']:
                results.append({**decision,'explanation':None,'feature_attributions':[]});continue
            normalized=(x[i]-self.engine.mean[0])/self.engine.std[0]
            base_h,base_e,base_m=self.engine.score_one_components(normalized)
            attributes=[]
            for j,name in enumerate(self.engine.features):
                occluded=normalized.copy()
                occluded[:,:,j]=0.0
                h,e,m=self.engine.score_one_components(occluded)
                attributes.append(dict(feature=name,hybrid_contribution=base_h-h,
                    reconstruction_contribution=base_e-e,latent_contribution=base_m-m,
                    observed_mean=float(x[i,:,:,j].mean()),benign_mean=float(self.engine.mean[0,0,0,j]),
                    standardized_deviation=float(normalized[:,:,j].mean())))
            attributes.sort(key=lambda item:abs(item['hybrid_contribution']),reverse=True)
            selected=attributes[:self.engine.cfg.attribution_top_k]
            names=', '.join(a['feature'] for a in selected)
            context=metadata[i] if metadata else {}
            if not isinstance(context,dict): raise ValueError('metadata entries must be dictionaries')
            results.append({**decision,'feature_attributions':selected,
                'metadata':context,'explanation':(
                    f'Anomalous traffic block: hybrid score {base_h:.3f} exceeded or triggered the calibrated alert policy '
                    f'(threshold {decision["threshold"]:.3f}; causes: {", ".join(decision["causes"])}). '
                    f'Highest absolute leave-one-feature-out score changes: {names}. '
                    'These measurements indicate deviations from learned benign traffic, not a verified attack family.')})
        return results
