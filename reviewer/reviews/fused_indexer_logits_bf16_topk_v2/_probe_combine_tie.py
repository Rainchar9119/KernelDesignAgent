"""Reviewer probe: boundary tie through the split-KV COMBINE path.
AC-D (plan) mandates: 'construct a case where the true top-K boundary score
equals the running threshold / top-K elements spread across split segments;
must be set-equal.' The LONG table uses only random data, where exact ties at
the 512-boundary don't occur, so this case is UNTESTED. Here the top `ntop`
KV positions are exactly tied high (top-512 boundary lands INSIDE the tie group
when ntop>512); the rest are lower. A correct kernel returns 512 valid indices
from the tied set. Run through the harness's own check_correctness oracle.
Read-only w.r.t. target dir."""
import sys, os
TGT='/root/paddlejob/inference-public/yuanzihang/KernelDesignAgent/kernel-agent/kernels/fused_indexer_logits_bf16_topk_v2'
sys.path.insert(0,TGT); os.chdir(TGT)
import torch, harness as H
r=H.Runner()
block=64
def build(B,S,ntop,seed=0):
    torch.manual_seed(seed)
    npt=(S+block-1)//block; nb=B*npt
    kv=torch.randn(nb,block,1,128,dtype=torch.bfloat16,device='cuda')*0.02
    kv.view(nb*block,128)[:ntop]=torch.ones(128,dtype=torch.bfloat16,device='cuda')*0.2
    q=torch.ones(B,1,64,128,dtype=torch.bfloat16,device='cuda')*0.1
    w=torch.ones(B,64,dtype=torch.float32,device='cuda')
    sl=torch.full((B,),S,dtype=torch.int32,device='cuda')
    pt=torch.arange(nb,dtype=torch.int32,device='cuda').view(B,npt).contiguous()
    return {'batch':B,'max_seq_len':S,'heads':64,'head_dim':128,'block':block,'np_total':npt,
       'q':q,'kv_packed':kv.view(torch.uint8).view((-1,block,1,256)),'weight':w,
       'seq_lens':sl,'page_table':pt,'q_bhd':q.view(B,64,128),'kv_bf16':kv.view(nb,block,128)}
def split_of(B,npt): return max(1,min(npt,(152+B//2)//B))
for B,S,ntop in [(1,4096,512),(1,4096,513),(1,4096,600),(64,1024,600)]:
    c=build(B,S,ntop); sp=split_of(B,c['np_total'])
    print(f"\n### B={B} S={S} ntop_tied={ntop} split={sp} "
          f"({'combine' if sp>1 else 'stage1'} path)")
    ok=H.check_correctness(r,c)
    print(f"    HARNESS ORACLE ok={ok}")
