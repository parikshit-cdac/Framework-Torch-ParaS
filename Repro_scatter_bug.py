"""
Standalone repro for the scatter_ undercount seen in Param2MoESparseMoeBlock.moe_infer:

    cnts = topk_ids.new_zeros((num_tokens, num_experts))
    cnts.scatter_(1, topk_ids, 1)
    tokens_per_expert = cnts.sum(dim=0)   # should sum to num_tokens * top_k

On the crash: num_tokens=16, top_k=6 -> expected sum=96, actual sum=31 on paras:1.

Run:  $PTSYCL_PYTHON repro_scatter_bug.py
"""
import sys
sys.path.insert(0, "/home/aicloud/parikshit/torch_paras/Framework-Torch-ParaS/python")

import torch
import torch_paras  # noqa: F401 -- registers the paras backend

NUM_TOKENS = 16
NUM_EXPERTS = 64      # set to your model's actual config.num_experts if different
TOP_K = 6             # set to your model's actual config.num_experts_per_tok
SEED = 0

DEVICES = ["cpu", "cuda:0", "paras:1"]


def run_case(device_str):
    device = torch.device(device_str)
    torch.manual_seed(SEED)

    # Same shape/dtype pattern as topk_idx in Param2MoEGate.forward:
    # torch.topk(...) over experts, per token -> (num_tokens, top_k) int64, values in [0, num_experts)
    topk_ids = torch.stack([
        torch.randperm(NUM_EXPERTS, device=device)[:TOP_K] for _ in range(NUM_TOKENS)
    ])

    cnts = topk_ids.new_zeros((NUM_TOKENS, NUM_EXPERTS))
    cnts.scatter_(1, topk_ids, 1)
    total = cnts.sum().item()
    expected = NUM_TOKENS * TOP_K

    # cross-check: count nonzero entries directly instead of via scatter_,
    # to make sure the *indices themselves* aren't the problem
    manual_total = 0
    cnts_cpu = topk_ids.cpu()
    for row in cnts_cpu.tolist():
        manual_total += len(set(row))  # randperm slice -> guaranteed unique per row

    status = "OK" if total == expected else "MISMATCH"
    print(f"[{device_str:10s}] scatter_ sum={total:4d} | expected={expected:4d} | "
          f"unique-index cross-check={manual_total:4d} | {status}")

    return total == expected


if __name__ == "__main__":
    print(f"NUM_TOKENS={NUM_TOKENS}  NUM_EXPERTS={NUM_EXPERTS}  TOP_K={TOP_K}  "
          f"(expected sum = {NUM_TOKENS * TOP_K})")
    print("-" * 80)

    results = {}
    for dev in DEVICES:
        try:
            results[dev] = run_case(dev)
        except Exception as e:
            print(f"[{dev:10s}] FAILED TO RUN: {e}")
            results[dev] = None

    print("-" * 80)
    if results.get("paras:1") is False and (results.get("cuda:0") or results.get("cpu")):
        print("CONFIRMED: scatter_(dim=1, index, scalar) undercounts on paras:1 "
              "while cpu/cuda are correct. This is a backend kernel bug, not a model bug.")
