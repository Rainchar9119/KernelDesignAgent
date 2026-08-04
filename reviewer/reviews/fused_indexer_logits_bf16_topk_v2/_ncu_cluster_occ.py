import sys, os
TGT='/root/paddlejob/inference-public/yuanzihang/KernelDesignAgent/kernel-agent/kernels/fused_indexer_logits_bf16_topk_v2'
sys.path.insert(0,TGT); os.chdir(TGT)
os.environ['FUSED_ENABLE_CLUSTER']='1'; os.environ['FUSED_CLUSTER']='1'
import torch, harness as H
r=H.Runner()
c=H.make_long_inputs(18,16*1024,seed=0)
for _ in range(5): r.fused_forward(c)
torch.cuda.synchronize()
torch.cuda.cudart().cudaProfilerStart(); r.fused_forward(c); torch.cuda.synchronize(); torch.cuda.cudart().cudaProfilerStop()
