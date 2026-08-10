import argparse, os, sys
HERE=os.path.dirname(os.path.abspath(__file__)); KERNEL_DIR=os.path.abspath(os.path.join(HERE,"..","..",".."))
sys.path.insert(0,KERNEL_DIR)
import harness as H, torch
ap=argparse.ArgumentParser(); ap.add_argument("--dtype",required=True); ap.add_argument("--num-tokens",type=int,required=True); ap.add_argument("--num-q-heads",type=int,default=64)
a=ap.parse_args(); tdt=H._TORCH_DTYPE[a.dtype]()
mod=H._load_candidate_module(tdt,cuh_path=os.path.join(KERNEL_DIR,"dev","main_norm_rope_r7_blockpertoken.cuh"),lineinfo=True)
inp=H.make_inputs(a.num_tokens,a.num_q_heads,a.dtype,pos_dtype=torch.int32,seed=0)
H.run_kernel(mod,inp); torch.cuda.synchronize(); print("launched",file=sys.stderr)
