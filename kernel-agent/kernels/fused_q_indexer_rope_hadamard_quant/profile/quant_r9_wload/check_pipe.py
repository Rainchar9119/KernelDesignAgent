"""Correctness (bitwise) + settled-clock direct HOT/COLD timing for the
software-pipelined grid-stride candidate vs repo baseline.

Usage: python check_pipe.py <src.cuh> <B> [B2 ...]
"""
import os
import sys

HD = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, HD)
import harness as H  # noqa: E402
import torch  # noqa: E402


def check_bitwise(inputs, cand_mod):
    base = H.make_direct_forward(inputs)
    cand = H.make_direct_forward(inputs, cand_mod)
    bq, bw = base()
    cq, cw = cand()
    torch.cuda.synchronize()
    q_eq = torch.equal(bq.view(torch.uint8), cq.view(torch.uint8))
    w_eq = torch.equal(bw, cw)
    nan = bool(torch.isnan(cw).any() or torch.isinf(cw).any())
    return q_eq, w_eq, nan


def main():
    src = sys.argv[1]
    batches = [int(x) for x in sys.argv[2:]] or [256, 512]
    H._load_elementwise()
    cand_mod = H._load_candidate_module(torch.bfloat16, os.path.abspath(src))

    flush, _ = H.make_l2_flusher()
    for B in batches:
        inputs = H.make_inputs(B, heads=64, seed=0)
        q_eq, w_eq, nan = check_bitwise(inputs, cand_mod)
        base = H.make_direct_forward(inputs)
        cand = H.make_direct_forward(inputs, cand_mod)
        for _ in range(400):
            base(); cand()
        torch.cuda.synchronize()
        bh = H.cuda_time_ms(base, 100, 200)
        ch = H.cuda_time_ms(cand, 100, 200)
        bc = H.cuda_time_ms(base, 100, 200, flush=flush)
        cc = H.cuda_time_ms(cand, 100, 200, flush=flush)
        print(f"B={B}: q_bitwise={q_eq} w_equal={w_eq} nan={nan}  "
              f"HOT base={bh*1e3:.2f}us cand={ch*1e3:.2f}us ({ch/bh:.3f})  "
              f"COLD base={bc*1e3:.2f}us cand={cc*1e3:.2f}us ({cc/bc:.3f})")


if __name__ == "__main__":
    main()
