import os, sys
HD = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, HD)
import harness as H
import torch
which, B, src = sys.argv[1], int(sys.argv[2]), sys.argv[3]
inputs = H.make_inputs(B, heads=64, seed=0)
H._load_elementwise()
if which == "baseline":
    run = H.make_direct_forward(inputs)
else:
    mod = H._load_candidate_module(torch.bfloat16, src)
    run = H.make_direct_forward(inputs, mod)
for _ in range(30): run()
torch.cuda.synchronize(); run(); torch.cuda.synchronize()
