"""
Direct diagnostic: instead of guessing at num_experts/n_group/topk_group and
building synthetic repros, capture the REAL topk_ids tensor from the actual
model on the actual crashing input (seq_len=16, first MoE layer), and compare:

  (a) tokens_per_expert as computed ON DEVICE via cnts.scatter_(...).sum(dim=0)
  (b) tokens_per_expert recomputed on CPU from the *same* topk_ids tensor
      (moved to cpu), using the exact same scatter_+sum logic

If (a) != (b) for paras:1, the bug is confirmed in the device scatter_/sum
with the REAL data (contradicting the earlier synthetic test -> means the
real topk_ids has some property my synthetic tests didn't reproduce: e.g.
non-contiguous strides, a different dtype, out-of-range values, or NaNs
propagating into the index tensor).

If (a) == (b), the corruption is happening earlier than moe_infer even
runs -- i.e. topk_ids itself is already wrong once it lands in this
function (which would point back at the gate: F.linear/sigmoid/masked_fill/
topk chain, or something upstream in the decoder layer). This script prints
enough to distinguish those cases without needing a second run.

Run:  $PTSYCL_PYTHON diagnose_real_moe_infer.py
"""
import sys
sys.path.insert(0, "/home/aicloud/parikshit/torch_paras/Framework-Torch-ParaS/python")

import torch
import torch_paras  # noqa: F401
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_PATH = "/home/aicloud/parikshit/models/Param2-17B-A2.4B-Thinking"
TARGET_DEVICE = torch.device("paras:1")
SEQ_LEN = 16

# We only need this to fire ONCE, on the first MoE layer, then we can stop.
_diagnosed = {"done": False}


def patched_moe_infer(self, x, topk_ids, topk_weight):
    if _diagnosed["done"]:
        # Fall through to a safe CPU-side computation for any later calls
        # so the rest of the forward pass doesn't also crash.
        topk_ids_cpu = topk_ids.cpu()
        x_cpu = x.cpu()
        return _cpu_moe_infer(self, x_cpu, topk_ids_cpu, topk_weight.cpu()).to(x.device)

    print("=" * 100)
    print(f"[Diagnostic] moe_infer called | x.shape={tuple(x.shape)} x.dtype={x.dtype} "
          f"x.is_contiguous={x.is_contiguous()}")
    print(f"[Diagnostic] topk_ids.shape={tuple(topk_ids.shape)} dtype={topk_ids.dtype} "
          f"device={topk_ids.device} is_contiguous={topk_ids.is_contiguous()} "
          f"stride={topk_ids.stride()}")

    num_experts = len(self.experts)
    print(f"[Diagnostic] num_experts={num_experts} "
          f"topk_ids.min()={topk_ids.min().item()} topk_ids.max()={topk_ids.max().item()}")

    has_nan = torch.isnan(topk_ids.float()).any().item() if topk_ids.is_floating_point() else False
    print(f"[Diagnostic] topk_ids has NaN: {has_nan}")

    # (a) on-device computation, exactly as the real code does it
    cnts_device = topk_ids.new_zeros((topk_ids.shape[0], num_experts))
    cnts_device.scatter_(1, topk_ids, 1)
    tokens_per_expert_device = cnts_device.sum(dim=0)
    total_device = tokens_per_expert_device.sum().item()

    # (b) recompute on CPU from the SAME topk_ids tensor (just moved), to
    # isolate whether the bug is in the device scatter_/sum or in topk_ids
    # itself having already-wrong values by the time it reaches this function
    topk_ids_cpu = topk_ids.cpu()
    cnts_cpu = topk_ids_cpu.new_zeros((topk_ids_cpu.shape[0], num_experts))
    cnts_cpu.scatter_(1, topk_ids_cpu, 1)
    tokens_per_expert_cpu = cnts_cpu.sum(dim=0)
    total_cpu = tokens_per_expert_cpu.sum().item()

    expected = topk_ids.numel()

    print(f"[Diagnostic] expected total (topk_ids.numel())      = {expected}")
    print(f"[Diagnostic] total from ON-DEVICE scatter_+sum      = {total_device}")
    print(f"[Diagnostic] total from CPU recompute (same tensor) = {total_cpu}")

    print(f"[Diagnostic] tokens_per_expert (device): {tokens_per_expert_device.cpu().tolist()}")
    print(f"[Diagnostic] tokens_per_expert (cpu)   : {tokens_per_expert_cpu.tolist()}")

    print(f"[Diagnostic] topk_ids raw values (per token row):")
    for row_i, row in enumerate(topk_ids_cpu.tolist()):
        dup_note = " <-- HAS DUPLICATE EXPERT IDS" if len(set(row)) != len(row) else ""
        print(f"    token {row_i:2d}: {row}{dup_note}")

    if total_device != expected and total_cpu == expected:
        print("[VERDICT] CONFIRMED: device scatter_/sum undercounts on REAL data even "
              "though topk_ids itself is correct (no duplicates, in-range). "
              "Bug is in the on-device scatter_ or sum(dim=0) kernel with this "
              "specific real shape/stride/dtype -- not reproduced by earlier "
              "synthetic tests, so likely stride/contiguity or size-dependent.")
    elif total_device != expected and total_cpu != expected:
        print("[VERDICT] topk_ids ALREADY has duplicate/out-of-range values before "
              "reaching moe_infer (both device and cpu recompute undercount from the "
              "same source tensor). Bug is upstream in the gate (F.linear/sigmoid/"
              "group_limited_topk chain on paras:1), not in scatter_ itself.")
    else:
        print("[VERDICT] Both device and cpu computations match expected count. "
              "moe_infer's own count logic is fine for this real input -- if the "
              "crash still reproduces elsewhere, check a LATER layer/token, or the "
              "argsort/fancy-indexing lines further down in moe_infer.")

    _diagnosed["done"] = True

    # Fall back to a safe CPU computation so we don't crash and can still
    # observe a clean forward pass completing (optional but nice to have).
    return _cpu_moe_infer(self, x.cpu(), topk_ids_cpu, topk_weight.cpu()).to(x.device)


def _cpu_moe_infer(self, x, topk_ids, topk_weight):
    """Reference implementation run entirely on CPU, used as a safe fallback
    after the diagnostic fires so the rest of the script can still finish."""
    num_experts = len(self.experts)
    cnts = topk_ids.new_zeros((topk_ids.shape[0], num_experts))
    cnts.scatter_(1, topk_ids, 1)
    tokens_per_expert = cnts.sum(dim=0)
    idxs = topk_ids.view(-1).argsort()
    sorted_tokens = x[idxs // topk_ids.shape[1]]
    tokens_per_expert = tokens_per_expert.numpy()
    outputs = []
    start_idx = 0
    for i, num_tokens in enumerate(tokens_per_expert):
        end_idx = start_idx + num_tokens
        if num_tokens == 0:
            continue
        expert = self.experts[i].to("cpu")
        tokens_for_this_expert = sorted_tokens[start_idx:end_idx]
        expert_out = expert(tokens_for_this_expert)
        outputs.append(expert_out)
        start_idx = end_idx
        self.experts[i].to(x.device if x.device.type != "cpu" else "cpu")
    outs = torch.cat(outputs, dim=0) if len(outputs) else sorted_tokens.new_empty(0)
    new_x = torch.empty_like(outs)
    new_x[idxs] = outs
    final_out = (
        new_x.view(*topk_ids.shape, -1)
        .type(topk_weight.dtype)
        .mul_(topk_weight.unsqueeze(dim=-1))
        .sum(dim=1)
        .type(new_x.dtype)
    )
    return final_out


def main():
    print("[Setup] Patching Param2MoESparseMoeBlock.moe_infer for diagnostics...")
    import importlib
    # Import the actual model module the same way transformers loads it, so we
    # patch the exact class instance the model uses.
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, dtype=torch.bfloat16, trust_remote_code=True)

    moe_block_cls = None
    for module in model.modules():
        if type(module).__name__ == "Param2MoESparseMoeBlock":
            moe_block_cls = type(module)
            break
    assert moe_block_cls is not None, "Could not find Param2MoESparseMoeBlock in the model"

    moe_block_cls.moe_infer = patched_moe_infer
    print(f"[Setup] Patched {moe_block_cls}")

    model = model.to(TARGET_DEVICE)
    model.eval()

    prompt = "The quick brown fox jumps over the lazy dog and explores the neighboring forest. " * 40
    encoded = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=SEQ_LEN)
    input_ids = encoded["input_ids"]
    if input_ids.shape[1] < SEQ_LEN:
        pad_id = tokenizer.pad_token_id
        pad = torch.full((1, SEQ_LEN - input_ids.shape[1]), pad_id, dtype=input_ids.dtype)
        input_ids = torch.cat([input_ids, pad], dim=1)
    attention_mask = torch.ones_like(input_ids)

    input_ids = input_ids.to(TARGET_DEVICE)
    attention_mask = attention_mask.to(TARGET_DEVICE)

    print(f"[Run] Forward pass, seq_len={SEQ_LEN}, device={TARGET_DEVICE}...")
    with torch.no_grad():
        model(input_ids=input_ids, attention_mask=attention_mask)

    print("[Done] Diagnostic forward pass completed without crashing "
          "(fallback CPU path was used after the first MoE layer's diagnosis).")


if __name__ == "__main__":
    main()
