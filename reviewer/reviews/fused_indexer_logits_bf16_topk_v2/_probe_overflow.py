"""Reviewer probe: force the radix overflow/de-clamp path (a single coarse bin
> CAND_CAP=4096) that Round 8 added to replace v1's silent clamp. Random data
never triggers it (~64/bin at 16K), so no passing test covers it. Build a case
whose logits collapse into one bin: identical K rows + identical Q => every KV
position gets the SAME score. Then top-512 is an all-tie set; the oracle passes
iff the kernel returns 512 valid distinct indices with matching score multiset.
Read-only w.r.t. target dir."""
import sys, os
TGT='/root/paddlejob/inference-public/yuanzihang/KernelDesignAgent/kernel-agent/kernels/fused_indexer_logits_bf16_topk_v2'
sys.path.insert(0,TGT); os.chdir(TGT)
import torch, harness as H

B, S = 1, 8192          # variant 8192 -> CAND_CAP=4096; 8192 pos all in one bin
dev='cuda'
block=64
np_total=(S+block-1)//block
num_blocks=B*np_total
# identical KV across every block/pos, identical Q across heads -> constant logits
kv=torch.ones(num_blocks, block, 1, 128, dtype=torch.bfloat16, device=dev)*0.05
q=torch.ones(B,1,64,128,dtype=torch.bfloat16,device=dev)*0.05
weight=torch.ones(B,64,dtype=torch.float32,device=dev)
seq_lens=torch.full((B,),S,dtype=torch.int32,device=dev)
perm=torch.randperm(num_blocks,dtype=torch.int32,device=dev)
page_table=perm.view(B,np_total).contiguous()
kv_packed=kv.view(torch.uint8).view((-1,block,1,256))
c={"batch":B,"max_seq_len":S,"heads":64,"head_dim":128,"block":block,
   "np_total":np_total,"q":q,"kv_packed":kv_packed,"weight":weight,
   "seq_lens":seq_lens,"page_table":page_table,
   "q_bhd":q.view(B,64,128),"kv_bf16":kv.view(num_blocks,block,128)}
r=H.Runner()
logits=r.logits(c); torch.cuda.synchronize()
row=logits[0,:S]
uniq=torch.unique(row)
print(f"logits over {S} valid pos: {uniq.numel()} unique values "
      f"(min={float(row.min()):.4f} max={float(row.max()):.4f}) "
      f"-> {'ALL-TIE (forces overflow)' if uniq.numel()<=4 else 'not collapsed enough'}")
ok=H.check_correctness(r,c)
gp,gr=r.golden(c,logits=logits)
fp,fw=r.fused_forward(c)
torch.cuda.synchronize()
nvalid=int((fp[0]>=0).sum()); ndistinct=int(torch.unique(fw[0][fw[0]>=0]).numel())
print(f"candidate returned {nvalid}/512 valid, {ndistinct} distinct raw idx")
print(f"OVERFLOW-PATH correctness (set+multiset+finite): {ok}")
