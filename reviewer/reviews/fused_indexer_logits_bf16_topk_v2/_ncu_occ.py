import sys, os
TGT='/root/paddlejob/inference-public/yuanzihang/KernelDesignAgent/kernel-agent/kernels/fused_indexer_logits_bf16_topk_v2'
sys.path.insert(0,TGT); os.chdir(TGT)
import torch, harness as H
r=H.Runner()
c=H.make_long_inputs(64,16*1024,seed=0)
for _ in range(3): r.fused_forward(c)
torch.cuda.synchronize()
torch.cuda.cudart().cudaProfilerStart()
r.fused_forward(c)
torch.cuda.synchronize()
torch.cuda.cudart().cudaProfilerStop()
