from __future__ import annotations
import torch
from torch import nn

class MLP(nn.Module):
    def __init__(self,di,do,h=128,depth=2):
        super().__init__(); ls=[]; d=di
        for _ in range(depth): ls += [nn.Linear(d,h),nn.SiLU()]; d=h
        ls += [nn.Linear(d,do)]; self.net=nn.Sequential(*ls)
    def forward(self,x): return self.net(x)

def rel_pool(rel,m):
    outw=rel[:,:,:,None]*m[:,None,:,:]
    inw=rel.transpose(1,2)[:,:,:,None]*m[:,None,:,:]
    return outw.sum(2),inw.sum(2),outw.amax(2),inw.amax(2)

class TransitionEncoder(nn.Module):
    """Permutation-equivariant transition encoder.

    mode='edge' masks node-after information so edge-ID cannot shortcut through
    the partner node mechanism. mode='node' receives the complete observed
    transition because node dynamics legitimately depend on A and post-edge R.
    """
    def __init__(self,mode='node',d=64,out=96,rounds=3):
        super().__init__(); assert mode in ('edge','node'); self.mode=mode; self.rounds=rounds
        self.init=MLP(13,d,128,2); self.msg=MLP(d+13,d,128,2); self.fuse=MLP(9*d+13,d,192,2)
        self.gru=nn.GRUCell(d,d); self.norm=nn.LayerNorm(d); self.out=MLP(3*d+8,out,192,2)
    def forward(self,x):
        xb=x['nodes_before']; xa=x['nodes_after']; A=x['edges_before']; R=x['edges_after']; action=x['action']; nm=x['node_mask']; B,N=xb.shape
        if self.mode=='edge': xa=xb
        eye=torch.eye(N,device=xb.device)[None]
        valid=(nm[:,:,None]*nm[:,None,:])*(1-eye); A=A*valid; R=R*valid
        Aout,Ain=A.sum(2),A.sum(1); Rout,Rin=R.sum(2),R.sum(1)
        act_b=(xb*action).sum(1,keepdim=True).expand(B,N); act_a=(xa*action).sum(1,keepdim=True).expand(B,N)
        local=torch.stack([xb,xa,(xa-xb).abs(),action,nm,Aout,Ain,Rout,Rin,act_b,act_a,xb*action,xa*action],-1)
        h=self.init(local)*nm[:,:,None]
        for _ in range(self.rounds):
            m=self.msg(torch.cat([h,local],-1))*nm[:,:,None]
            ao,ai,amxo,amxi=rel_pool(A,m); ro,ri,rmxo,rmxi=rel_pool(R,m)
            u=self.fuse(torch.cat([h,ao,ai,amxo,amxi,ro,ri,rmxo,rmxi,local],-1))
            h2=self.gru(u.reshape(B*N,-1),h.reshape(B*N,-1)).reshape(B,N,-1)
            h=self.norm(h+h2)*nm[:,:,None]
        denom=nm.sum(1,keepdim=True).clamp_min(1)
        mean=(h*nm[:,:,None]).sum(1)/denom
        mx=h.masked_fill(nm[:,:,None]==0,-1e9).amax(1)
        sm=(h*nm[:,:,None]).sum(1)/10.0
        edge_bits=valid.sum((1,2)).clamp_min(1)
        stats=torch.stack([
            nm.sum(1)/10.0,
            A.sum((1,2))/edge_bits,
            R.sum((1,2))/edge_bits,
            ((A-R).abs()*valid).sum((1,2))/edge_bits,
            ((xa-xb).abs()*nm).sum(1)/denom.squeeze(1),
            (xb*nm).sum(1)/denom.squeeze(1),
            (xa*nm).sum(1)/denom.squeeze(1),
            (action*nm).sum(1)
        ],-1)
        return self.out(torch.cat([mean,mx,sm,stats],-1))

class HistoryBranch(nn.Module):
    def __init__(self,mode,nfactor=6,te=96,hd=128):
        super().__init__(); self.trans=TransitionEncoder(mode=mode,out=te); self.gru=nn.GRU(te,hd,batch_first=True)
        self.shared=MLP(hd+te+1,128,128,1); self.head=nn.Linear(128,nfactor)
    def forward(self,hist):
        B,H,N=hist['nodes_before'].shape
        flat={k:v.reshape(B*H,*v.shape[2:]) for k,v in hist.items()}
        z=self.trans(flat).reshape(B,H,-1); _,hn=self.gru(z); mean=z.mean(1)
        hfrac=torch.full((B,1),float(H)/8.0,device=z.device,dtype=z.dtype)
        s=self.shared(torch.cat([hn[-1],mean,hfrac],-1)); return self.head(s)

class FactorSeparatedSystemID(nn.Module):
    def __init__(self,nedge=6,nnode=6):
        super().__init__(); self.edge=HistoryBranch('edge',nedge); self.node=HistoryBranch('node',nnode)
    def forward(self,hist): return {'edge_logits':self.edge(hist),'node_logits':self.node(hist)}
