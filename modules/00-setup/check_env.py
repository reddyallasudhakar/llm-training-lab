"""Check that the learning environment is ready: Python, PyTorch on MPS, MLX, and a tiny model."""

import sys

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"  # ~1 GB download, cached after the first run


def ok(msg):
    print(f"✓ {msg}")


def fail(msg):
    print(f"✗ {msg}")
    sys.exit(1)


if sys.version_info < (3, 11):
    fail(f"Python {sys.version.split()[0]} — need 3.11+ (run this with `uv run`)")
ok(f"Python {sys.version.split()[0]}")

import torch

if not torch.backends.mps.is_available():
    fail("PyTorch MPS backend not available")
x = torch.randn(1024, 1024, device="mps")
(x @ x).sum().item()
ok(f"PyTorch {torch.__version__} running on MPS")

import mlx.core as mx

y = mx.random.normal((1024, 1024))
mx.eval(y @ y)
ok(f"MLX {mx.__version__} running on {mx.default_device()}")

from transformers import AutoModelForCausalLM, AutoTokenizer

tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16).to("mps")
messages = [{"role": "user", "content": "In one sentence, what is a token?"}]
inputs = tok.apply_chat_template(
    messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
).to("mps")
out = model.generate(**inputs, max_new_tokens=40, do_sample=False)
reply = tok.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
ok(f"{MODEL} generated: {reply.strip()!r}")

print("\nAll set. On to Module 1.")
