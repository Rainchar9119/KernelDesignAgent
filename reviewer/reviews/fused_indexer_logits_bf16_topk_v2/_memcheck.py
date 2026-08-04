import sys, os
TGT='/root/paddlejob/inference-public/yuanzihang/KernelDesignAgent/kernel-agent/kernels/fused_indexer_logits_bf16_topk_v2'
sys.path.insert(0,TGT); os.chdir(TGT)
import torch, harness as H
r=H.Runner()
# shapes hitting two-level combine + a GROUP remainder of 1 segment (ncand==TOPK path)
for b,a in [(1,16*1024),(1,64*1024),(1,256*1024)]:
    c=H.make_long_inputs(b,a,seed=0); r.fused_forward(c); torch.cuda.synchronize()
print("OK no launch error")
