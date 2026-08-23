"""
Param2-17B Unified Benchmark Suite: Torch-ParaS vs Native PyTorch CUDA
========================================================================

Single-file, reproducible benchmark that covers:

  1. Correctness / Parity   - shape & finiteness checks, greedy-decode
                               cross-backend token/text parity
  2. Latency                - TTFT (prefill) and ITL (decode) sweeps
  3. Throughput              - prompt tok/s and decode tok/s
  4. Memory usage             - allocated / reserved / peak / KV-cache growth
  5. Numerical fidelity        - max-abs-error, MAE, cosine sim, top1/top5
                                 agreement, KL divergence (ParaS vs CUDA)
  6. Stability                - multi-turn generation, NaN/Inf watch
  7. Backward pass            - gradient sanity check (bonus signal)
  8. Baseline comparison      - CUDA is the baseline; every ParaS metric is
                                 reported alongside its CUDA counterpart with
                                 a speedup/parity column

Usage:
    python param2_benchmark.py

Outputs:
    param2_benchmark_results.json   -- raw machine-readable results
    PARAM2_BENCHMARK_REPORT.md      -- human-readable tables/report

Reproducibility:
    - Fixed SEED used for torch.manual_seed before every generation call
    - Config block below fully controls sweep sizes / iteration counts
    - Deterministic (greedy) decoding used for correctness/parity checks;
      sampling (fixed seed) used only for the stability stress test
"""

import sys
import os
import gc
import time
import json
import datetime

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

sys.path.insert(0, "/home/aicloud/parikshit/torch_paras/Framework-Torch-ParaS/python")

import torch
import torch.nn.functional as F
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM

try:
    import torch_paras
except ImportError:
    torch_paras = None


# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------

MODEL_PATH = "/home/aicloud/parikshit/models/Param2-17B-A2.4B-Thinking"
EXPECTED_VOCAB_SIZE = 128008

PREFILL_SEQ_LENGTHS = [16, 64, 128, 256, 512]
DECODE_TOKEN_BUDGETS = [16, 32, 64, 128]
WARMUP_ITERS = 2
MEASURE_ITERS = 5

CORRECTNESS_NEW_TOKENS = 16        # greedy decode length for parity check
STABILITY_TURNS = 3
STABILITY_BUDGET = 64
STABILITY_PROMPTS = [
    "What is the capital of France?",
    "What is its approximate population?",
    "Name one famous landmark there.",
]

SEED = 42
JSON_OUTPUT_PATH = "param2_benchmark_results.json"
MARKDOWN_OUTPUT_PATH = "PARAM2_BENCHMARK_REPORT.md"

FIDELITY_THRESHOLDS = {
    "max_abs_error": 1.0,
    "cosine_similarity": 0.99,
    "top1_agreement_pct": 95.0,
    "kl_divergence": 0.05,
}

# CUDA is treated as the baseline backend throughout the report.
BACKENDS = [
    {"key": "paras", "name": "Torch-ParaS", "device_str": "paras:1"},
    {"key": "cuda", "name": "Native PyTorch CUDA", "device_str": "cuda:0"},
]
BASELINE_KEY = "cuda"


# --------------------------------------------------------------------------
# Compatibility patch (shared cache_utils quirk on both backends)
# --------------------------------------------------------------------------

def apply_dynamic_layer_patch():
    try:
        import transformers.cache_utils as cache_utils

        DynamicLayer = getattr(cache_utils, "DynamicLayer", None)
        if DynamicLayer is None:
            print("[Patch] DynamicLayer not found; no cache patch applied.")
            return

        def _safe_dynamic_layer_update(self, key_states, value_states, *args, **kwargs):
            if self.keys is None or self.keys.numel() == 0 or self.keys.ndim != key_states.ndim:
                self.keys = key_states
                self.values = value_states
            else:
                self.keys = torch.cat([self.keys, key_states], dim=-2)
                self.values = torch.cat([self.values, value_states], dim=-2)
            return self.keys, self.values

        DynamicLayer.update = _safe_dynamic_layer_update
        print("[Patch] Transformers DynamicLayer.update patched.")
    except Exception as e:
        print(f"[Patch] DynamicLayer patch notice: {e}")


# --------------------------------------------------------------------------
# Backend-agnostic helpers
# --------------------------------------------------------------------------

def load_model(device):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, dtype=torch.bfloat16, trust_remote_code=True)
    model = model.to(device)
    return tokenizer, model


def free_model(model, device):
    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
        return
    if torch_paras is not None:
        empty_cache_fn = getattr(torch_paras, "empty_cache", None)
        if empty_cache_fn is not None:
            empty_cache_fn()


def make_sync_fn(device):
    if device.type == "cuda":
        def _sync():
            torch.cuda.synchronize(device)
        return _sync

    def _sync():
        try:
            torch_paras.synchronize(device)
        except TypeError:
            torch_paras.synchronize()
    return _sync


def reset_memory_stats(device):
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        return
    if torch_paras is None:
        return
    reset_fn = getattr(torch_paras, "reset_peak_memory_stats", None)
    if reset_fn is None:
        return
    try:
        reset_fn(device)
    except TypeError:
        reset_fn()


def get_memory_stats(device):
    if device.type == "cuda":
        return {
            "allocated_mb": torch.cuda.memory_allocated(device) / 1e6,
            "reserved_mb": torch.cuda.memory_reserved(device) / 1e6,
            "max_allocated_mb": torch.cuda.max_memory_allocated(device) / 1e6,
            "max_reserved_mb": torch.cuda.max_memory_reserved(device) / 1e6,
        }

    stats = {"allocated_mb": None, "reserved_mb": None, "max_allocated_mb": None, "max_reserved_mb": None}
    if torch_paras is None:
        return stats
    fn_map = {
        "allocated_mb": "memory_allocated",
        "reserved_mb": "memory_reserved",
        "max_allocated_mb": "max_memory_allocated",
        "max_reserved_mb": "max_memory_reserved",
    }
    for key, fn_name in fn_map.items():
        fn = getattr(torch_paras, fn_name, None)
        if fn is None:
            continue
        try:
            value = fn(device)
        except TypeError:
            value = fn()
        stats[key] = value / 1e6
    return stats


def build_prompt_input(tokenizer, seq_len, device):
    filler = "The quick brown fox jumps over the lazy dog and explores the neighboring forest. " * 40
    encoded = tokenizer(filler, return_tensors="pt", truncation=True, max_length=seq_len)
    input_ids = encoded["input_ids"]
    if input_ids.shape[1] < seq_len:
        pad_id = tokenizer.pad_token_id
        pad_len = seq_len - input_ids.shape[1]
        pad = torch.full((1, pad_len), pad_id, dtype=input_ids.dtype)
        input_ids = torch.cat([input_ids, pad], dim=1)
    attention_mask = torch.ones_like(input_ids)
    return input_ids.to(device), attention_mask.to(device)


def build_reference_input(tokenizer, device):
    messages = [{"role": "user", "content": "What is the capital of France?"}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    encoded = tokenizer(prompt, return_tensors="pt")
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)
    return input_ids, attention_mask


# --------------------------------------------------------------------------
# 1. Correctness / Parity
# --------------------------------------------------------------------------

def run_correctness_check(model, tokenizer, device):
    """Shape/finiteness sanity + deterministic greedy decode for later
    cross-backend parity comparison."""
    model.eval()
    input_ids, attention_mask = build_reference_input(tokenizer, device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits

    expected_shape = (1, input_ids.shape[1], EXPECTED_VOCAB_SIZE)
    shape_ok = tuple(logits.shape) == expected_shape
    finite_ok = not torch.isnan(logits).any().item() and not torch.isinf(logits).any().item()

    next_token_id = torch.argmax(logits[0, -1, :]).item()

    torch.manual_seed(SEED)
    with torch.no_grad():
        output_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=CORRECTNESS_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True,
        )
    generated_ids = output_ids[0, input_ids.shape[1]:].tolist()
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    length_ok = len(generated_ids) == CORRECTNESS_NEW_TOKENS

    passed = shape_ok and finite_ok and length_ok
    assert shape_ok, f"Unexpected logits shape: {tuple(logits.shape)}; expected {expected_shape}"
    assert finite_ok, "Logits contain NaN/Inf!"
    assert length_ok, f"Expected {CORRECTNESS_NEW_TOKENS} tokens, got {len(generated_ids)}"

    return {
        "shape_ok": shape_ok,
        "finite_ok": finite_ok,
        "length_ok": length_ok,
        "passed": passed,
        "prefill_top1_token_id": next_token_id,
        "greedy_generated_ids": generated_ids,
        "greedy_generated_text": generated_text,
    }


def compute_cross_backend_parity(correctness_results):
    a = correctness_results["paras"]
    b = correctness_results["cuda"]
    token_ids_match = a["greedy_generated_ids"] == b["greedy_generated_ids"]
    text_match = a["greedy_generated_text"] == b["greedy_generated_text"]
    prefill_top1_match = a["prefill_top1_token_id"] == b["prefill_top1_token_id"]
    return {
        "prefill_top1_match": prefill_top1_match,
        "greedy_token_ids_match": token_ids_match,
        "greedy_text_match": text_match,
    }


# --------------------------------------------------------------------------
# 2 & 3. Latency + Throughput: Prefill (TTFT)
# --------------------------------------------------------------------------

def profile_prefill(model, tokenizer, device, sync_fn):
    results = []
    model.eval()

    for seq_len in PREFILL_SEQ_LENGTHS:
        input_ids, attention_mask = build_prompt_input(tokenizer, seq_len, device)

        warmup_logits = None
        for _ in range(WARMUP_ITERS):
            with torch.no_grad():
                warmup_logits = model(input_ids=input_ids, attention_mask=attention_mask).logits

        assert warmup_logits.shape[-1] == EXPECTED_VOCAB_SIZE, f"Unexpected vocab size: {warmup_logits.shape[-1]}"
        assert not torch.isnan(warmup_logits).any(), f"NaN logits at prefill S={seq_len}"
        assert not torch.isinf(warmup_logits).any(), f"Inf logits at prefill S={seq_len}"

        times = []
        for _ in range(MEASURE_ITERS):
            sync_fn()
            t0 = time.perf_counter()
            with torch.no_grad():
                model(input_ids=input_ids, attention_mask=attention_mask)
            sync_fn()
            times.append(time.perf_counter() - t0)

        mean_time = sum(times) / len(times)
        std_time = (sum((t - mean_time) ** 2 for t in times) / len(times)) ** 0.5

        results.append({
            "seq_len": seq_len,
            "ttft_ms": mean_time * 1000,
            "ttft_std_ms": std_time * 1000,
            "prompt_throughput_tok_s": seq_len / mean_time,
        })

        print(
            f"  S={seq_len:4d} | TTFT={mean_time * 1000:8.2f} ms (+/-{std_time * 1000:.2f}) | "
            f"throughput={seq_len / mean_time:9.1f} tok/s"
        )

    return results


# --------------------------------------------------------------------------
# 2 & 3. Latency + Throughput: Decode (ITL)
# --------------------------------------------------------------------------

def decode_run(model, input_ids, attention_mask, n_tokens, sync_fn, device, check_finite):
    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=True)

    if check_finite:
        assert not torch.isnan(out.logits).any(), "NaN logits in decode prefill step"
        assert not torch.isinf(out.logits).any(), "Inf logits in decode prefill step"

    past = out.past_key_values
    cur_id = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)
    cur_mask = torch.cat([attention_mask, torch.ones_like(cur_id)], dim=-1)

    step_times = []
    for _ in range(n_tokens):
        sync_fn()
        t0 = time.perf_counter()
        with torch.no_grad():
            out = model(input_ids=cur_id, attention_mask=cur_mask, past_key_values=past, use_cache=True)
        sync_fn()
        step_times.append(time.perf_counter() - t0)

        if check_finite:
            assert not torch.isnan(out.logits).any(), "NaN logits during decode"
            assert not torch.isinf(out.logits).any(), "Inf logits during decode"

        past = out.past_key_values
        cur_id = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)
        cur_mask = torch.cat([cur_mask, torch.ones_like(cur_id)], dim=-1)

    end_mem_mb = get_memory_stats(device)["allocated_mb"]
    return step_times, end_mem_mb


def profile_decode(model, tokenizer, device, sync_fn):
    results = []
    model.eval()
    input_ids, attention_mask = build_prompt_input(tokenizer, 128, device)

    for n in DECODE_TOKEN_BUDGETS:
        for _ in range(WARMUP_ITERS):
            decode_run(model, input_ids, attention_mask, n, sync_fn, device, check_finite=True)

        step_times, end_mem_mb = decode_run(model, input_ids, attention_mask, n, sync_fn, device, check_finite=False)

        mean_step = sum(step_times) / len(step_times)
        throughput = len(step_times) / sum(step_times)

        results.append({
            "token_budget": n,
            "itl_ms": mean_step * 1000,
            "decode_throughput_tok_s": throughput,
            "allocated_at_budget_mb": end_mem_mb,
        })

        print(f"  N={n:4d} | ITL={mean_step * 1000:7.2f} ms/tok | throughput={throughput:7.2f} tok/s")

    return results


def estimate_kv_cache_growth_mb_per_token(decode_results):
    valid = [r for r in decode_results if r["allocated_at_budget_mb"] is not None]
    if len(valid) < 2:
        return None
    lo, hi = valid[0], valid[-1]
    token_delta = hi["token_budget"] - lo["token_budget"]
    if token_delta <= 0:
        return None
    return (hi["allocated_at_budget_mb"] - lo["allocated_at_budget_mb"]) / token_delta


# --------------------------------------------------------------------------
# 5. Numerical fidelity (ParaS vs CUDA)
# --------------------------------------------------------------------------

def compute_numerical_fidelity(logits_a, logits_b):
    a = logits_a.float().flatten()
    b = logits_b.float().flatten()

    max_abs_error = (a - b).abs().max().item()
    mae = (a - b).abs().mean().item()
    cosine_sim = F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()

    top1_a = logits_a.argmax(dim=-1)
    top1_b = logits_b.argmax(dim=-1)
    top1_agree = (top1_a == top1_b).float().mean().item() * 100

    top5_a = logits_a.topk(5, dim=-1).indices.reshape(-1, 5)
    top5_b = logits_b.topk(5, dim=-1).indices.reshape(-1, 5)
    matches = sum(
        1 for i in range(top5_a.shape[0])
        if len(set(top5_a[i].tolist()) & set(top5_b[i].tolist())) > 0
    )
    top5_agree = (matches / top5_a.shape[0]) * 100

    log_p = F.log_softmax(logits_a.float(), dim=-1)
    q = F.softmax(logits_b.float(), dim=-1)
    kl = F.kl_div(log_p, q, reduction="batchmean").item()

    return {
        "max_abs_error": max_abs_error,
        "mae": mae,
        "cosine_similarity": cosine_sim,
        "top1_agreement_pct": top1_agree,
        "top5_agreement_pct": top5_agree,
        "kl_divergence": kl,
    }


# --------------------------------------------------------------------------
# 6. Backward pass gradient sanity (bonus signal)
# --------------------------------------------------------------------------

def run_backward_verification(model, tokenizer, device):
    model.train()
    input_ids, attention_mask = build_reference_input(tokenizer, device)
    labels = input_ids.clone()

    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    loss = outputs.loss
    loss_val = loss.item()

    assert not torch.isnan(loss), "Loss is NaN!"
    assert not torch.isinf(loss), "Loss is Inf!"

    loss.backward()

    total_params = 0
    grad_count = 0
    nan_grads = 0
    inf_grads = 0

    for _, param in model.named_parameters():
        total_params += 1
        if param.grad is not None:
            grad_count += 1
            if torch.isnan(param.grad).any():
                nan_grads += 1
            if torch.isinf(param.grad).any():
                inf_grads += 1

    model.zero_grad(set_to_none=True)

    assert grad_count > 0, "No gradients were computed!"
    assert nan_grads == 0, f"Found {nan_grads} NaN gradients!"
    assert inf_grads == 0, f"Found {inf_grads} Inf gradients!"

    return {
        "loss": loss_val,
        "total_params": total_params,
        "grad_count": grad_count,
        "nan_grads": nan_grads,
        "inf_grads": inf_grads,
    }


# --------------------------------------------------------------------------
# 7. Stability (multi-turn sampling stress test)
# --------------------------------------------------------------------------

def run_stability_test(model, tokenizer, device):
    model.eval()
    messages = []
    turn_results = []
    total_nan = 0
    total_inf = 0

    for i, user_msg in enumerate(STABILITY_PROMPTS[:STABILITY_TURNS]):
        messages.append({"role": "user", "content": user_msg})
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        encoded = tokenizer(prompt, return_tensors="pt")
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)

        torch.manual_seed(SEED + i)
        with torch.no_grad():
            output_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=STABILITY_BUDGET,
                do_sample=True,
                temperature=0.6,
                top_p=0.95,
                repetition_penalty=1.1,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            hidden_out = model(input_ids=output_ids, output_hidden_states=True)

        reply_ids = output_ids[0, input_ids.shape[1]:]
        reply_text = tokenizer.decode(reply_ids, skip_special_tokens=True)
        messages.append({"role": "assistant", "content": reply_text})

        nan_layers = sum(int(torch.isnan(h).any().item()) for h in hidden_out.hidden_states)
        inf_layers = sum(int(torch.isinf(h).any().item()) for h in hidden_out.hidden_states)
        total_nan += nan_layers
        total_inf += inf_layers

        turn_results.append({
            "turn": i + 1,
            "prompt": user_msg,
            "generated_tokens": int(reply_ids.shape[0]),
            "nan_layers": nan_layers,
            "inf_layers": inf_layers,
        })

        print(f"  Turn {i + 1}: generated {reply_ids.shape[0]} tokens | NaN layers={nan_layers} | Inf layers={inf_layers}")

    passed = total_nan == 0 and total_inf == 0
    assert passed, f"Stability test failed: {total_nan} NaN layers, {total_inf} Inf layers detected"

    return {"turns": turn_results, "total_nan_layers": total_nan, "total_inf_layers": total_inf, "passed": passed}


# --------------------------------------------------------------------------
# Markdown report
# --------------------------------------------------------------------------

def build_markdown_report(results):
    lines = []
    lines.append("# Param2-17B Benchmark Report: Torch-ParaS vs Native PyTorch CUDA")
    lines.append("")
    lines.append(f"Generated: {results['timestamp']}")
    lines.append(f"Model: {results['model']}")
    lines.append(f"PyTorch: {results['pytorch_version']} | Transformers: {results['transformers_version']}")
    lines.append(f"Baseline backend: {results['baseline_key']}")
    lines.append("")

    verdict = "PASSED" if results["overall_passed"] else "FAILED"
    lines.append(f"## Overall Result: {verdict}")
    lines.append("")

    paras = results["backends"]["paras"]
    cuda = results["backends"]["cuda"]

    # 1. Correctness / Parity
    lines.append("## 1. Correctness / Parity")
    lines.append("")
    lines.append("| Check | Torch-ParaS | Native CUDA |")
    lines.append("|---|---|---|")
    for label, key in [
        ("Logits shape OK", "shape_ok"),
        ("Logits finite (no NaN/Inf)", "finite_ok"),
        ("Greedy decode length OK", "length_ok"),
        ("Overall correctness passed", "passed"),
    ]:
        pv = paras["correctness"][key]
        cv = cuda["correctness"][key]
        lines.append(f"| {label} | {'PASS' if pv else 'FAIL'} | {'PASS' if cv else 'FAIL'} |")
    lines.append("")

    parity = results["cross_backend_parity"]
    lines.append("### Cross-backend parity (ParaS vs CUDA, greedy decoding)")
    lines.append("")
    lines.append("| Parity check | Match |")
    lines.append("|---|---|")
    lines.append(f"| Prefill top-1 token matches | {'YES' if parity['prefill_top1_match'] else 'NO'} |")
    lines.append(f"| Greedy generated token IDs match | {'YES' if parity['greedy_token_ids_match'] else 'NO'} |")
    lines.append(f"| Greedy generated text matches | {'YES' if parity['greedy_text_match'] else 'NO'} |")
    lines.append("")

    # 2 & 3. Latency & Throughput - Prefill
    lines.append("## 2. Latency & Throughput -- Prefill (TTFT)")
    lines.append("")
    lines.append(
        "| Seq Len | TTFT ParaS (ms) | TTFT CUDA (ms, baseline) | Speedup (CUDA/ParaS) | "
        "Throughput ParaS (tok/s) | Throughput CUDA (tok/s) |"
    )
    lines.append("|---|---|---|---|---|---|")
    for p, c in zip(paras["prefill"], cuda["prefill"]):
        speedup = c["ttft_ms"] / p["ttft_ms"] if p["ttft_ms"] else float("nan")
        lines.append(
            f"| {p['seq_len']} | {p['ttft_ms']:.2f} | {c['ttft_ms']:.2f} | {speedup:.2f}x | "
            f"{p['prompt_throughput_tok_s']:.1f} | {c['prompt_throughput_tok_s']:.1f} |"
        )
    lines.append("")

    # 2 & 3. Latency & Throughput - Decode
    lines.append("## 3. Latency & Throughput -- Decode (ITL)")
    lines.append("")
    lines.append(
        "| Token Budget | ITL ParaS (ms/tok) | ITL CUDA (ms/tok, baseline) | Speedup (CUDA/ParaS) | "
        "Throughput ParaS (tok/s) | Throughput CUDA (tok/s) |"
    )
    lines.append("|---|---|---|---|---|---|")
    for p, c in zip(paras["decode"], cuda["decode"]):
        speedup = c["itl_ms"] / p["itl_ms"] if p["itl_ms"] else float("nan")
        lines.append(
            f"| {p['token_budget']} | {p['itl_ms']:.2f} | {c['itl_ms']:.2f} | {speedup:.2f}x | "
            f"{p['decode_throughput_tok_s']:.1f} | {c['decode_throughput_tok_s']:.1f} |"
        )
    lines.append("")

    # 4. Memory
    lines.append("## 4. Memory Usage")
    lines.append("")
    lines.append("| Metric | Torch-ParaS | Native CUDA (baseline) |")
    lines.append("|---|---|---|")
    for label, key in [
        ("Base allocated (MB)", "base_allocated_mb"),
        ("Base reserved (MB)", "base_reserved_mb"),
        ("Peak allocated (MB)", "peak_allocated_mb"),
        ("Peak reserved (MB)", "peak_reserved_mb"),
        ("KV-cache growth (MB/token)", "kv_cache_growth_mb_per_token"),
    ]:
        pv = paras["memory"][key]
        cv = cuda["memory"][key]
        pv_str = f"{pv:.2f}" if pv is not None else "N/A"
        cv_str = f"{cv:.2f}" if cv is not None else "N/A"
        lines.append(f"| {label} | {pv_str} | {cv_str} |")
    lines.append("")

    # 5. Numerical fidelity
    lines.append("## 5. Numerical Fidelity (Torch-ParaS vs Native CUDA)")
    lines.append("")
    lines.append("| Metric | Value | Threshold | Status |")
    lines.append("|---|---|---|---|")
    fid = results["numerical_fidelity"]
    th = results["fidelity_thresholds"]
    checks = [
        ("Max Absolute Error", fid["max_abs_error"], th["max_abs_error"], fid["max_abs_error"] <= th["max_abs_error"]),
        ("Mean Absolute Error", fid["mae"], None, None),
        ("Cosine Similarity", fid["cosine_similarity"], th["cosine_similarity"], fid["cosine_similarity"] >= th["cosine_similarity"]),
        ("Top-1 Agreement (%)", fid["top1_agreement_pct"], th["top1_agreement_pct"], fid["top1_agreement_pct"] >= th["top1_agreement_pct"]),
        ("Top-5 Agreement (%)", fid["top5_agreement_pct"], None, None),
        ("KL Divergence", fid["kl_divergence"], th["kl_divergence"], fid["kl_divergence"] <= th["kl_divergence"]),
    ]
    for name, value, threshold, ok in checks:
        threshold_str = f"{threshold}" if threshold is not None else "-"
        status_str = "PASS" if ok else ("FAIL" if ok is not None else "-")
        lines.append(f"| {name} | {value:.6f} | {threshold_str} | {status_str} |")
    lines.append("")

    # 6. Stability
    lines.append("## 6. Multi-Turn Stability")
    lines.append("")
    lines.append("| Backend | Turn | Prompt | Generated Tokens | NaN Layers | Inf Layers |")
    lines.append("|---|---|---|---|---|---|")
    for key, label in [("paras", "Torch-ParaS"), ("cuda", "Native CUDA")]:
        for turn in results["backends"][key]["stability"]["turns"]:
            lines.append(
                f"| {label} | {turn['turn']} | {turn['prompt']} | {turn['generated_tokens']} | "
                f"{turn['nan_layers']} | {turn['inf_layers']} |"
            )
    lines.append("")

    # 7. Backward pass
    lines.append("## 7. Backward Pass Gradient Verification")
    lines.append("")
    lines.append("| Backend | Loss | Valid Grads | Total Params | NaN Grads | Inf Grads |")
    lines.append("|---|---|---|---|---|---|")
    for key, label in [("paras", "Torch-ParaS"), ("cuda", "Native CUDA")]:
        bw = results["backends"][key]["backward"]
        lines.append(
            f"| {label} | {bw['loss']:.4f} | {bw['grad_count']} | {bw['total_params']} | "
            f"{bw['nan_grads']} | {bw['inf_grads']} |"
        )
    lines.append("")

    # 8. Load time / baseline summary
    lines.append("## 8. Load Time & Baseline Summary")
    lines.append("")
    lines.append("| Backend | Load Time (s) | Role |")
    lines.append("|---|---|---|")
    lines.append(f"| Torch-ParaS | {paras['load_time_s']:.1f} | Candidate |")
    lines.append(f"| Native CUDA | {cuda['load_time_s']:.1f} | Baseline |")
    lines.append("")

    lines.append("## Reproducibility / Test Configuration")
    lines.append("")
    for k, v in results["config"].items():
        lines.append(f"- {k}: {v}")
    lines.append("")

    return "\n".join(lines)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    torch.manual_seed(SEED)
    apply_dynamic_layer_patch()

    print("=" * 88)
    print("  Param2-17B Unified Benchmark: Torch-ParaS vs Native PyTorch CUDA")
    print("=" * 88)
    print(f"PyTorch version : {torch.__version__}")
    print(f"Transformers    : {transformers.__version__}")
    print(f"Model           : {MODEL_PATH}")
    print(f"Baseline backend: {BASELINE_KEY}")

    backend_results = {}
    reference_logits = {}
    correctness_results = {}

    for backend in BACKENDS:
        key, name, device_str = backend["key"], backend["name"], backend["device_str"]
        device = torch.device(device_str)

        print("\n" + "=" * 88)
        print(f"  BACKEND: {name} ({device_str})")
        print("=" * 88)

        print("\n[Load] Loading tokenizer and model...")
        t0 = time.time()
        tokenizer, model = load_model(device)
        load_time = time.time() - t0
        print(f"Model ready on {device_str} in {load_time:.1f}s")

        sync_fn = make_sync_fn(device)

        reset_memory_stats(device)
        base_mem = get_memory_stats(device)

        print("\n[Correctness/Parity] Shape/finiteness + greedy decode...")
        correctness_result = run_correctness_check(model, tokenizer, device)
        correctness_results[key] = correctness_result
        print(f"  Correctness passed: {correctness_result['passed']}")

        print("\n[Latency/Throughput] Profiling prefill (TTFT)...")
        prefill_results = profile_prefill(model, tokenizer, device, sync_fn)

        print("\n[Latency/Throughput] Profiling decode (ITL)...")
        decode_results = profile_decode(model, tokenizer, device, sync_fn)

        peak_mem = get_memory_stats(device)
        kv_growth_mb_per_token = estimate_kv_cache_growth_mb_per_token(decode_results)

        print("\n[Fidelity] Capturing reference logits...")
        ref_input_ids, ref_attention_mask = build_reference_input(tokenizer, device)
        with torch.no_grad():
            ref_logits = model(input_ids=ref_input_ids, attention_mask=ref_attention_mask).logits.float().cpu()
        reference_logits[key] = ref_logits

        print("\n[Stability] Running multi-turn conversational test...")
        stability_result = run_stability_test(model, tokenizer, device)

        print("\n[Backward] Running gradient verification...")
        backward_result = run_backward_verification(model, tokenizer, device)
        print(
            f"Gradients: {backward_result['grad_count']}/{backward_result['total_params']} valid | "
            f"NaN={backward_result['nan_grads']} | Inf={backward_result['inf_grads']}"
        )

        backend_results[key] = {
            "name": name,
            "device": device_str,
            "load_time_s": load_time,
            "correctness": correctness_result,
            "memory": {
                "base_allocated_mb": base_mem["allocated_mb"],
                "base_reserved_mb": base_mem["reserved_mb"],
                "peak_allocated_mb": peak_mem["max_allocated_mb"],
                "peak_reserved_mb": peak_mem["max_reserved_mb"],
                "kv_cache_growth_mb_per_token": kv_growth_mb_per_token,
            },
            "prefill": prefill_results,
            "decode": decode_results,
            "stability": stability_result,
            "backward": backward_result,
        }

        print(f"\n[Cleanup] Freeing {name} model...")
        free_model(model, device)

    print("\n" + "=" * 88)
    print("  CROSS-BACKEND PARITY (greedy decoding)")
    print("=" * 88)
    cross_backend_parity = compute_cross_backend_parity(correctness_results)
    for k, v in cross_backend_parity.items():
        print(f"  {k:28s}: {v}")

    print("\n" + "=" * 88)
    print("  NUMERICAL FIDELITY: Torch-ParaS vs Native PyTorch CUDA")
    print("=" * 88)
    fidelity = compute_numerical_fidelity(reference_logits["paras"], reference_logits["cuda"])
    for k, v in fidelity.items():
        print(f"  {k:24s}: {v:.6f}")

    fidelity_passed = (
        fidelity["max_abs_error"] <= FIDELITY_THRESHOLDS["max_abs_error"]
        and fidelity["cosine_similarity"] >= FIDELITY_THRESHOLDS["cosine_similarity"]
        and fidelity["top1_agreement_pct"] >= FIDELITY_THRESHOLDS["top1_agreement_pct"]
        and fidelity["kl_divergence"] <= FIDELITY_THRESHOLDS["kl_divergence"]
    )

    overall_passed = (
        fidelity_passed
        and backend_results["paras"]["correctness"]["passed"]
        and backend_results["cuda"]["correctness"]["passed"]
        and backend_results["paras"]["stability"]["passed"]
        and backend_results["cuda"]["stability"]["passed"]
        and backend_results["paras"]["backward"]["nan_grads"] == 0
        and backend_results["paras"]["backward"]["inf_grads"] == 0
        and backend_results["cuda"]["backward"]["nan_grads"] == 0
        and backend_results["cuda"]["backward"]["inf_grads"] == 0
    )

    final_results = {
        "model": MODEL_PATH,
        "expected_vocab_size": EXPECTED_VOCAB_SIZE,
        "timestamp": datetime.datetime.now().isoformat(),
        "pytorch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "baseline_key": BASELINE_KEY,
        "config": {
            "prefill_seq_lengths": PREFILL_SEQ_LENGTHS,
            "decode_token_budgets": DECODE_TOKEN_BUDGETS,
            "warmup_iters": WARMUP_ITERS,
            "measure_iters": MEASURE_ITERS,
            "correctness_new_tokens": CORRECTNESS_NEW_TOKENS,
            "stability_turns": STABILITY_TURNS,
            "stability_budget": STABILITY_BUDGET,
            "seed": SEED,
        },
        "backends": backend_results,
        "cross_backend_parity": cross_backend_parity,
        "numerical_fidelity": fidelity,
        "fidelity_thresholds": FIDELITY_THRESHOLDS,
        "fidelity_passed": fidelity_passed,
        "overall_passed": overall_passed,
    }

    with open(JSON_OUTPUT_PATH, "w") as f:
        json.dump(final_results, f, indent=2)
    print(f"\n[Export] JSON results written to {JSON_OUTPUT_PATH}")

    markdown_report = build_markdown_report(final_results)
    with open(MARKDOWN_OUTPUT_PATH, "w") as f:
        f.write(markdown_report)
    print(f"[Export] Markdown report written to {MARKDOWN_OUTPUT_PATH}")

    print("\n" + "=" * 88)
    verdict = "PASSED" if overall_passed else "FAILED"
    print(f"  OVERALL BENCHMARK RESULT: {verdict}")
    print("=" * 88)


if __name__ == "__main__":
    main()
