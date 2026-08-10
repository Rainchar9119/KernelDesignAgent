import argparse, os, sys
HERE=os.path.dirname(os.path.abspath(__file__))
KERNEL_DIR=os.path.abspath(os.path.join(HERE,"..","..",".."))
sys.path.insert(0,KERNEL_DIR)
import harness as H, torch
ap=argparse.ArgumentParser()
ap.add_argument("--dtype",required=True); ap.add_argument("--num-tokens",type=int,required=True)
ap.add_argument("--num-q-heads",type=int,default=64); ap.add_argument("--pos-dtype",default="int32")
a=ap.parse_args()
tdt=H._TORCH_DTYPE[a.dtype](); pdt=torch.int32 if a.pos_dtype=="int32" else torch.int64
mod=H._load_candidate_module(tdt,cuh_path=os.path.join(KERNEL_DIR,"candidate","main_norm_rope.cuh"),lineinfo=True)
inp=H.make_inputs(a.num_tokens,a.num_q_heads,a.dtype,pos_dtype=pdt,seed=0)
H.run_kernel(mod,inp); torch.cuda.synchronize()
print("launched",a.dtype,a.num_tokens,file=sys.stderr)
