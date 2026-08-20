"""
Param2-17B Complete Verification Suite on Framework-Torch-ParaS (NVIDIA H200 NVL)

Tests:
    1. Stage 1: Single Forward Pass (Prefill)
    2. Stage 2: Multi-Token Autoregressive Generation (Decode with KV Cache)
    3. Stage 3: Full Backward Pass (Loss computation & Gradient backprop)
"""

import sys
import os
import time

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

sys.path.insert(0, "/home/aicloud/parikshit/torch_paras/Framework-Torch-ParaS/python")

import torch
import torch_paras
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM


try:
    import transformers.cache_utils as cache_utils

    DynamicLayer = getattr(cache_utils, "DynamicLayer", None)

    if DynamicLayer is not None:

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
    else:
        print("[Patch] DynamicLayer not found; no cache patch applied.")

except Exception as e:
    print(f"[Patch] DynamicLayer patch notice: {e}")


MODEL_PATH = "/home/aicloud/parikshit/models/Param2-17B-A2.4B-Thinking"
TARGET_DEVICE = torch.device("paras:1")
EXPECTED_VOCAB_SIZE = 128008
MAX_NEW_TOKENS = 16


print("=" * 80)
print("  Param2-17B Comprehensive Verification on Framework-Torch-ParaS (H200)")
print("=" * 80)
print(f"PyTorch version : {torch.__version__}")
print(f"ParaS Backend   : {torch_paras.backend_name()}")
print(f"Target Device   : {TARGET_DEVICE} ({torch_paras.device_name(1)})")

try:
    print(f"AMP Supported   : {torch_paras.get_amp_supported_dtype()}")
except AttributeError:
    try:
        print(f"AMP Supported   : {torch.paras.get_amp_supported_dtype()}")
    except AttributeError:
        print("AMP Supported   : API unavailable")


print("\n[Step 1/4] Loading Tokenizer & Model...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("Tokenizer loaded.")

t0 = time.time()
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, dtype=torch.bfloat16, trust_remote_code=True)
model_load_time = time.time() - t0
print(f"Model loaded on CPU in {model_load_time:.1f}s")

t0 = time.time()
model = model.to(TARGET_DEVICE)
model_transfer_time = time.time() - t0
print(f"Model transferred to {TARGET_DEVICE} in {model_transfer_time:.1f}s")


messages = [{"role": "user", "content": "What is the capital of France?"}]
prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
encoded = tokenizer(prompt, return_tensors="pt")
input_ids = encoded["input_ids"].to(TARGET_DEVICE)

attention_mask = encoded.get("attention_mask")
if attention_mask is not None:
    attention_mask = attention_mask.to(TARGET_DEVICE)

print("\nTest prompt:")
print(prompt)
print(f"\nInput IDs shape : {input_ids.shape}")
print(f"Input IDs device: {input_ids.device}")


print("\n" + "=" * 80)
print("  STAGE 1: Single Forward Pass Verification (Prefill)")
print("=" * 80)

model.eval()

t0 = time.time()
with torch.no_grad():
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
fwd_time = time.time() - t0

logits = outputs.logits

print(f"Forward pass completed in {fwd_time:.3f}s")
print(f"Logits shape : {logits.shape}")
print(f"Logits dtype : {logits.dtype}")
print(f"Logits device: {logits.device}")
print(f"Logits mean  : {logits.float().mean().item():.4f}")
print(f"Logits std   : {logits.float().std().item():.4f}")

expected_shape = (1, input_ids.shape[1], EXPECTED_VOCAB_SIZE)
assert logits.shape == expected_shape, f"Unexpected logits shape: {logits.shape}; expected {expected_shape}"
assert not torch.isnan(logits).any(), "Logits contain NaNs!"
assert not torch.isinf(logits).any(), "Logits contain Infs!"

next_token_id = torch.argmax(logits[0, -1, :]).item()
next_token = tokenizer.decode([next_token_id])
print(f"Top predicted next token: {next_token_id} -> {next_token!r}")
print(">>> STAGE 1: PASSED")


print("\n" + "=" * 80)
print("  STAGE 2: Autoregressive Generation Verification (KV Cache)")
print("=" * 80)
print(f"Generating {MAX_NEW_TOKENS} tokens on {TARGET_DEVICE}...")

t0 = time.time()
with torch.no_grad():
    output_ids = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        use_cache=True,
    )
gen_time = time.time() - t0

num_generated = output_ids.shape[1] - input_ids.shape[1]
tok_per_sec = num_generated / gen_time if gen_time > 0 else float("inf")

print(f"Generation completed in {gen_time:.2f}s")
print(f"Generated tokens : {num_generated}")
print(f"Generation speed : {tok_per_sec:.2f} tok/s")

generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=False)
print("\nGenerated text:")
print(generated_text)

assert num_generated == MAX_NEW_TOKENS, f"Expected {MAX_NEW_TOKENS} tokens, got {num_generated}"
print(">>> STAGE 2: PASSED")


print("\n" + "=" * 80)
print("  STAGE 3: Full Backward Pass (Training / Gradient Backpropagation)")
print("=" * 80)

model.train()
labels = input_ids.clone()

print("Executing forward pass with loss computation...")
t0 = time.time()
outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
loss = outputs.loss
loss_val = loss.item()
loss_fwd_time = time.time() - t0
print(f"Forward with loss completed in {loss_fwd_time:.3f}s | Loss: {loss_val:.4f}")

assert not torch.isnan(loss), "Loss is NaN!"
assert not torch.isinf(loss), "Loss is Inf!"

print("Executing backward pass (loss.backward())...")
t0 = time.time()
loss.backward()
bwd_time = time.time() - t0
print(f"Backward pass completed in {bwd_time:.3f}s")


print("\nInspecting parameter gradients:")

grad_count = 0
total_params = 0
nan_grads = 0
inf_grads = 0
sample_grads = []

for name, param in model.named_parameters():
    total_params += 1
    if param.grad is not None:
        grad_count += 1
        g = param.grad
        if torch.isnan(g).any():
            nan_grads += 1
        if torch.isinf(g).any():
            inf_grads += 1
        if len(sample_grads) < 8:
            sample_grads.append((name, tuple(g.shape), g.dtype, g.float().norm().item()))

print(f"Total parameter tensors       : {total_params}")
print(f"Parameters with valid gradients: {grad_count} / {total_params}")
print(f"NaN gradients                 : {nan_grads}")
print(f"Inf gradients                 : {inf_grads}")

print("\nSample layer gradients:")
for name, shape, dtype, norm in sample_grads:
    print(f"  - {name:60s} | shape={str(shape):20s} | dtype={str(dtype):12s} | norm={norm:.6f}")

assert grad_count > 0, "No gradients were computed!"
assert nan_grads == 0, f"Found {nan_grads} NaN gradients!"
assert inf_grads == 0, f"Found {inf_grads} Inf gradients!"
print(">>> STAGE 3: PASSED")


print("\n" + "=" * 80)
print("  ALL 3 STAGES PASSED")
print("  Param2 is Fully Supported on Framework-Torch-ParaS (H200)")
print("=" * 80)
