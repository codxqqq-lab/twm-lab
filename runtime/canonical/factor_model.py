from __future__ import annotations
import torch
from torch import nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self,di,do,h=128,depth=2):
        super().__init__(); ls=[]; d=di
        for _ in range(depth): ls += [nn.Linear(d,h),nn.SiLU()]; d=h
        ls += [nn.Linear(d,do)]; self.net=nn.Sequential(*ls)
    def forward(self,x): return self.net(x)

class FactorizedRelationalTransitionModel(nn.Module):
    """R10.2 repair: edge and node factor identities are separated.

    The edge factor may affect node dynamics only through the learned post-edge
    relation R. The global action broadcast is reconstructed from observable
    state/action + A/R summaries, never from edge-factor-conditioned hidden state.
    """
    def __init__(self,nedge=6,nnode=6,ctxd=16,d=64,rounds=4):
        super().__init__(); self.ectx=nn.Embedding(nedge,ctxd); self.nctx=nn.Embedding(nnode,ctxd); self.rounds=rounds; self.d=d
        self.einit=MLP(5+ctxd,d,128,2); self.emsg=MLP(d+5+ctxd,d,128,2); self.efuse=MLP(5*d+5+ctxd,d,192,2); self.egru=nn.GRUCell(d,d); self.enorm=nn.LayerNorm(d); self.ehead=MLP(2*d+13+ctxd,3,192,2)
        # Factor-invariant action context: action-node state and degrees in current A / predicted R.
        self.action_enc=MLP(5,d,128,2)
        self.ninit=MLP(8+ctxd+d,d,128,2); self.nmsg=MLP(d+ctxd+4+d,d,128,2); self.nfuse=MLP(9*d+8+ctxd+d,d,192,2); self.ngru=nn.GRUCell(d,d); self.nnorm=nn.LayerNorm(d); self.nhead=MLP(d+8+ctxd+d,3,128,2)
    @staticmethod
    def pool(rel,m):
        outw=rel[:,:,:,None]*m[:,None,:,:]; inw=rel.transpose(1,2)[:,:,:,None]*m[:,None,:,:]
        return outw.sum(2),inw.sum(2),outw.amax(2),inw.amax(2)
    def action_context(self,nodes,action,Aout,Ain,Rout,Rin):
        astate=(nodes*action).sum(1)
        aaout=(Aout*action).sum(1); aain=(Ain*action).sum(1)
        arout=(Rout*action).sum(1); arin=(Rin*action).sum(1)
        return self.action_enc(torch.stack([astate,aaout,aain,arout,arin],-1))
    def forward(self,b):
        nodes,A,action,nm=b['nodes'],b['edges'],b['action'],b['node_mask']; B,N=nodes.shape
        eye=torch.eye(N,device=nodes.device)[None]; valid=(nm[:,:,None]*nm[:,None,:])*(1-eye); A=A*valid
        ec=self.ectx(b['edge_factor']); ew=ec[:,None,:].expand(B,N,-1)
        outd=A.sum(2); ind=A.sum(1); elocal=torch.stack([nodes,action,nm,outd,ind],-1)
        eh=self.einit(torch.cat([elocal,ew],-1))*nm[:,:,None]
        for _ in range(self.rounds):
            m=self.emsg(torch.cat([eh,elocal,ew],-1))*nm[:,:,None]; so,si,mo,mi=self.pool(A,m)
            u=self.efuse(torch.cat([eh,so,si,mo,mi,elocal,ew],-1)); h2=self.egru(u.reshape(B*N,-1),eh.reshape(B*N,-1)).reshape(B,N,-1); eh=self.enorm(eh+h2)*nm[:,:,None]
        hi=eh[:,:,None,:].expand(B,N,N,-1); hj=eh[:,None,:,:].expand(B,N,N,-1); li=elocal[:,:,None,:].expand(B,N,N,-1); lj=elocal[:,None,:,:].expand(B,N,N,-1); ecc=ec[:,None,None,:].expand(B,N,N,-1)
        pair=torch.stack([A,action[:,:,None].expand(B,N,N),action[:,None,:].expand(B,N,N)],-1)
        elog=self.ehead(torch.cat([hi,hj,li,lj,pair,ecc],-1)); ep=torch.softmax(elog,-1); R=(ep[:,:,:,0]*A+ep[:,:,:,2])*valid

        # Protected semantic interface: node path may read learned R, but its loss cannot update the edge subsystem.
        R_node=R.detach()
        Aout,Ain=A.sum(2),A.sum(1); Rout,Rin=R_node.sum(2),R_node.sum(1)
        acth=self.action_context(nodes,action,Aout,Ain,Rout,Rin); aw=acth[:,None,:].expand(B,N,-1)
        nc=self.nctx(b['node_factor']); nw=nc[:,None,:].expand(B,N,-1)
        nlocal=torch.stack([nodes,action,nodes*action,nm,Aout,Ain,Rout,Rin],-1)
        nh=self.ninit(torch.cat([nlocal,nw,aw],-1))*nm[:,:,None]
        for _ in range(self.rounds):
            m=self.nmsg(torch.cat([nh,nw,nodes[:,:,None],action[:,:,None],Aout[:,:,None],Ain[:,:,None],aw],-1))*nm[:,:,None]
            ao,ai,amxo,amxi=self.pool(A,m); ro,ri,rmxo,rmxi=self.pool(R_node,m)
            u=self.nfuse(torch.cat([nh,ao,ai,amxo,amxi,ro,ri,rmxo,rmxi,nlocal,nw,aw],-1)); h2=self.ngru(u.reshape(B*N,-1),nh.reshape(B*N,-1)).reshape(B,N,-1); nh=self.nnorm(nh+h2)*nm[:,:,None]
        nlog=self.nhead(torch.cat([nh,nlocal,nw,aw],-1)); np=torch.softmax(nlog,-1); NXT=(np[:,:,0]*nodes+np[:,:,2])*nm
        return {'edge_event_logits':elog,'next_edges_prob':R,'node_event_logits':nlog,'next_nodes_prob':NXT,'valid':valid}

def delta(before,after):
    same=before==after; return torch.where(same,torch.zeros_like(before,dtype=torch.long),torch.where(after<.5,torch.ones_like(before,dtype=torch.long),torch.full_like(before,2,dtype=torch.long)))

def edge_weights(pool):
    B,N=pool['nodes'].shape; valid=(pool['node_mask'][:,:,None]*pool['node_mask'][:,None,:])*(1-torch.eye(N)[None]); lab=delta(pool['edges'],pool['next_edges']); cnt=torch.stack([((lab==c).float()*valid).sum() for c in range(3)]).clamp_min(1); w=cnt.rsqrt(); return w/w.mean()

def loss_fn(o,b,ew):
    valid=o['valid']; el=delta(b['edges'],b['next_edges']); ece=F.cross_entropy(o['edge_event_logits'].reshape(-1,3),el.reshape(-1),weight=ew.to(el.device),reduction='none').reshape_as(valid); ece=(ece*valid).sum()/valid.sum().clamp_min(1); eps=1e-4; er=o['next_edges_prob']*(1-2*eps)+eps; eb=(F.binary_cross_entropy(er,b['next_edges'],reduction='none')*valid).sum()/valid.sum().clamp_min(1)
    nm=b['node_mask']; nl=delta(b['nodes'],b['next_nodes']); nce=F.cross_entropy(o['node_event_logits'].reshape(-1,3),nl.reshape(-1),reduction='none').reshape_as(nm); nce=(nce*nm).sum()/nm.sum().clamp_min(1); nr=o['next_nodes_prob']*(1-2*eps)+eps; nb=(F.binary_cross_entropy(nr,b['next_nodes'],reduction='none')*nm).sum()/nm.sum().clamp_min(1)
    return ece+.25*eb+nce+.25*nb
