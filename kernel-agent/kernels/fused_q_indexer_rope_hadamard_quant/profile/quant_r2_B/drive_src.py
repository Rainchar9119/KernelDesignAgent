import argparse, os, sys
d = os.path.dirname(os.path.abspath(__file__))
while d != "/" and not os.path.exists(os.path.join(d, "harness.py")):
    d = os.path.dirname(d)
sys.path.insert(0, d)
import harness as H
import torch
ap = argparse.ArgumentParser()
ap.add_argument("--batch", type=int, default=256)
ap.add_argument("--cuh", required=True)
a = ap.parse_args()
H._load_elementwise()
module = H._load_candidate_module(torch.bfloat16, os.path.abspath(a.cuh), lineinfo=True)
run = H.make_direct_forward(H.make_inputs(a.batch, heads=64, seed=0), module)
run(); torch.cuda.synchronize(); run(); torch.cuda.synchronize()
