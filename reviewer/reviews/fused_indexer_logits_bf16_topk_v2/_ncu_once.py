import sys, os
TGT='/root/paddlejob/inference-public/yuanzihang/KernelDesignAgent/kernel-agent/kernels/fused_indexer_logits_bf16_topk_v2'
sys.path.insert(0,TGT); os.chdir(TGT)
import torch, harness as H
b,s=int(sys.argv[1]),int(sys.argv[2])
r=H.Runner(); c=H.make_inputs(b,s,seed=0)
for _ in range(3):
    r.two_step(c); r.fused_forward(c)
torch.cuda.synchronize()
torch.cuda.cudart().cudaProfilerStart()
for _ in range(3):
    r.two_step(c)
    r.fused_forward(c)
torch.cuda.synchronize()
torch.cuda.cudart().cudaProfilerStop()
