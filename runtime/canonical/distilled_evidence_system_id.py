from __future__ import annotations
import torch
from torch import nn
from .separated_system_id import TransitionEncoder, MLP

class DistilledNodeEvidence(nn.Module):
    """Per-transition node-factor evidence distilled from frozen decoder likelihood.

    The head predicts centered *per-node-bit* log evidence. History posterior logits
    are a learned prior plus the sum of evidence multiplied by the number of
    observed node bits in each transition. This preserves the additive structure
    of raw log likelihood while keeping the regression target scale stable across
    graph sizes.
    """
    def __init__(self,nfactor=6,te=96):
        super().__init__()
        self.trans=TransitionEncoder(mode='node',out=te)
        self.evidence=MLP(te,nfactor,128,2)
        self.prior=nn.Parameter(torch.zeros(nfactor))
    def transition_evidence_per_bit(self,hist):
        B,H,N=hist['nodes_before'].shape
        flat={k:v.reshape(B*H,*v.shape[2:]) for k,v in hist.items()}
        z=self.trans(flat).reshape(B,H,-1)
        e=self.evidence(z)
        return e-e.mean(-1,keepdim=True)
    def forward(self,hist):
        e=self.transition_evidence_per_bit(hist)
        bits=hist['node_mask'].sum(-1).clamp_min(1.0)
        return self.prior[None,:]+(e*bits[:,:,None]).sum(1)
