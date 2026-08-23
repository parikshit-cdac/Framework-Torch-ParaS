"""
Third-pass repro. scatter_ (confirmed OK) and topk with sorted=True and
candidates >= k (confirmed OK) are ruled out. This targets the two regimes
from the real code path that weren't covered yet:

  1. group_limited_topk's group-selection topk uses sorted=False:
         torch.topk(group_scores, k=self.topk_group, dim=-1, sorted=False)[1]
     Unsorted topk is often a genuinely different kernel path.

  2. The final topk over masked_scores can be forced to pick MORE indices
     than there are finite (non -inf) candidates, e.g. if upstream group
     masking (bugged or not) leaves fewer than top_k unmasked experts for
     some row. torch.topk must then return -inf "padding" entries, which
     is the heaviest possible tie condition -- and wasn't tested before.

Run:  $PTSYCL_PYTHON repro_topk_edge_cases.py
"""
import sys
sys.path.insert(0, "/home/aicloud/parikshit/torch_paras/Framework-Torch-ParaS/python")

import torch
import torch_paras  # noqa: F401

NUM_TOKENS = 16
NUM_EXPERTS = 64
TOP_K = 6
N_GROUP = 8            # adjust to match real config if known
TOPK_GROUP = 4          # adjust to match real config if known
SEED = 0
DEVICES = ["cpu", "cuda:0", "paras:1"]


def check_dupes(indices_2d):
    idx_cpu = indices_2d.cpu().tolist()
    dup_rows = sum(1 for row in idx_cpu if len(set(row)) != len(row))
    total_dupes = sum(len(row) - len(set(row)) for row in idx_cpu)
    return dup_rows, total_dupes


# ---------------------------------------------------------------------------
# Case 1: sorted=False topk, mirroring group selection
# ---------------------------------------------------------------------------
def case_unsorted_group_topk(device_str):
    device = torch.device(device_str)
    torch.manual_seed(SEED)
    group_scores = torch.rand(NUM_TOKENS, N_GROUP, device=device)
    _, group_idx = torch.topk(group_scores, k=TOPK_GROUP, dim=-1, sorted=False)
    dup_rows, total_dupes = check_dupes(group_idx)
    status = "OK" if dup_rows == 0 else "DUPLICATES FOUND"
    print(f"[{device_str:10s}] sorted=False group topk | rows_with_dupes={dup_rows:3d}/{NUM_TOKENS} "
          f"| total_dupe_slots={total_dupes:3d} | {status}")
    return dup_rows == 0


# ---------------------------------------------------------------------------
# Case 2: final topk forced to pad with -inf when candidates < top_k
# ---------------------------------------------------------------------------
def case_topk_forced_padding(device_str, num_candidates):
    device = torch.device(device_str)
    torch.manual_seed(SEED)
    scores = torch.rand(NUM_TOKENS, NUM_EXPERTS, device=device)
    mask = torch.zeros(NUM_TOKENS, NUM_EXPERTS, dtype=torch.bool, device=device)
    for row in range(NUM_TOKENS):
        keep = torch.randperm(NUM_EXPERTS, device=device)[:num_candidates]
        mask[row, keep] = True
    masked_scores = scores.masked_fill(~mask, float("-inf"))

    # top_k > num_candidates -> guaranteed to pull in tied -inf padding
    _, top_indices = torch.topk(masked_scores, k=TOP_K, dim=-1)
    dup_rows, total_dupes = check_dupes(top_indices)
    status = "OK" if dup_rows == 0 else "DUPLICATES FOUND"
    print(f"[{device_str:10s}] candidates/row={num_candidates:2d} < top_k={TOP_K} (forced -inf padding) | "
          f"rows_with_dupes={dup_rows:3d}/{NUM_TOKENS} | total_dupe_slots={total_dupes:3d} | {status}")
    return dup_rows == 0


if __name__ == "__main__":
    print(f"NUM_TOKENS={NUM_TOKENS}  NUM_EXPERTS={NUM_EXPERTS}  TOP_K={TOP_K}  "
          f"N_GROUP={N_GROUP}  TOPK_GROUP={TOPK_GROUP}")
    print("-" * 96)

    any_paras_dupe = False

    print("Case 1: sorted=False group-selection topk")
    for dev in DEVICES:
        try:
            ok = case_unsorted_group_topk(dev)
            if dev == "paras:1" and not ok:
                any_paras_dupe = True
        except Exception as e:
            print(f"[{dev:10s}] FAILED TO RUN: {e}")
    print("-" * 96)

    print("Case 2: final topk forced to pad with -inf ties (candidates < top_k)")
    for num_candidates in [1, 2, 4, TOP_K - 1]:
        for dev in DEVICES:
            try:
                ok = case_topk_forced_padding(dev, num_candidates)
                if dev == "paras:1" and not ok:
                    any_paras_dupe = True
            except Exception as e:
                print(f"[{dev:10s}] candidates={num_candidates} FAILED TO RUN: {e}")
        print("-" * 96)

    if any_paras_dupe:
        print("CONFIRMED: paras:1 topk produces duplicate indices in at least one of "
              "these regimes (sorted=False, and/or forced -inf padding). This is the "
              "root cause -- fix belongs in the topk kernel's handling of ties/unsorted "
              "mode, not in scatter_ or the MoE routing code.")
    else:
        print("Still no duplicates. Next step: get the REAL n_group/topk_group/num_experts "
              "values from config.json and re-run with exact values, since a real run may "
              "hit a candidate count this sweep didn't cover. If it still doesn't reproduce, "
              "shift focus to argsort + fancy-indexing (idxs = topk_ids.view(-1).argsort(); "
              "new_x[idxs] = outs) as the next suspect.")
