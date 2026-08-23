"""
Follow-up repro: scatter_ itself is confirmed correct on paras:1 (previous test).
This isolates the actual upstream suspect: torch.topk() on masked_scores where
most entries are -inf (exactly what Param2MoEGate.group_limited_topk produces
before calling torch.topk(masked_scores, k=top_k, dim=-1)).

Hypothesis: a naive topk kernel implemented as iterative argmax + mask-out can
return DUPLICATE indices within a row when many values are tied (e.g. all -inf),
if it fails to correctly invalidate the exact winning position each iteration.
Duplicate indices collapse under scatter_ (last-write-wins on the same cell),
which would explain the undercount without scatter_ itself being at fault.

Run:  $PTSYCL_PYTHON repro_topk_duplicate_bug.py
"""
import sys
sys.path.insert(0, "/home/aicloud/parikshit/torch_paras/Framework-Torch-ParaS/python")

import torch
import torch_paras  # noqa: F401

NUM_TOKENS = 16
NUM_EXPERTS = 64
TOP_K = 6

# How many experts survive group masking (i.e. how many finite, non -inf
# candidates each token has to choose top_k from). Set this to match your
# real config: topk_group * (num_experts // n_group). If unsure, this sweep
# covers the two interesting regimes: barely enough candidates, and lots.
CANDIDATE_COUNTS_TO_TEST = [TOP_K, TOP_K + 2, NUM_EXPERTS // 2]

SEED = 0
DEVICES = ["cpu", "cuda:0", "paras:1"]


def build_masked_scores(num_candidates, device):
    torch.manual_seed(SEED)
    scores = torch.rand(NUM_TOKENS, NUM_EXPERTS, device=device)
    # mask out everything except `num_candidates` random positions per row,
    # mirroring group_limited_topk's masked_scores.masked_fill(~mask, -inf)
    mask = torch.zeros(NUM_TOKENS, NUM_EXPERTS, dtype=torch.bool, device=device)
    for row in range(NUM_TOKENS):
        keep = torch.randperm(NUM_EXPERTS, device=device)[:num_candidates]
        mask[row, keep] = True
    masked_scores = scores.masked_fill(~mask, float("-inf"))
    return masked_scores


def run_case(device_str, num_candidates):
    device = torch.device(device_str)
    masked_scores = build_masked_scores(num_candidates, device)

    _, top_indices = torch.topk(masked_scores, k=TOP_K, dim=-1)

    top_indices_cpu = top_indices.cpu()
    dup_rows = 0
    total_dupes = 0
    for row in top_indices_cpu.tolist():
        n_unique = len(set(row))
        if n_unique != len(row):
            dup_rows += 1
            total_dupes += (len(row) - n_unique)

    status = "OK" if dup_rows == 0 else "DUPLICATES FOUND"
    print(
        f"[{device_str:10s}] candidates/row={num_candidates:3d} | "
        f"rows_with_dupes={dup_rows:3d}/{NUM_TOKENS} | total_dupe_slots={total_dupes:3d} | {status}"
    )
    return dup_rows == 0


if __name__ == "__main__":
    print(f"NUM_TOKENS={NUM_TOKENS}  NUM_EXPERTS={NUM_EXPERTS}  TOP_K={TOP_K}")
    print("-" * 88)

    any_paras_dupe = False
    for num_candidates in CANDIDATE_COUNTS_TO_TEST:
        for dev in DEVICES:
            try:
                ok = run_case(dev, num_candidates)
                if dev == "paras:1" and not ok:
                    any_paras_dupe = True
            except Exception as e:
                print(f"[{dev:10s}] candidates/row={num_candidates:3d} | FAILED TO RUN: {e}")
        print("-" * 88)

    if any_paras_dupe:
        print("CONFIRMED: torch.topk returns duplicate indices per row on paras:1 "
              "under tie/masked conditions. This is the root cause feeding into "
              "the scatter_ undercount seen in moe_infer.")
    else:
        print("No duplicates found in this sweep. If the real crash still reproduces, "
              "the candidate counts here may not match the real n_group/topk_group "
              "config -- check config.json for exact values and adjust "
              "CANDIDATE_COUNTS_TO_TEST, or the bug may be in argsort/fancy-indexing "
              "further down in moe_infer instead.")
