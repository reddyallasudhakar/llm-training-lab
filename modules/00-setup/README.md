# M0. Environment setup

Your system Python is 3.9, which is too old for current PyTorch, MLX and TRL. We'll use
[uv](https://docs.astral.sh/uv/) to get Python 3.11 and keep one environment for the whole plan.

## Steps

From the `ai-learning/` folder:

1. Install uv (one-off): `brew install uv`
2. Create the environment and install the core libraries: `uv sync`
3. Check everything works: `uv run python modules/00-setup/check_env.py`
4. Make a free [Hugging Face](https://huggingface.co/join) account and create a read token, then
   run `uv run hf auth login` and paste it in. Some models (Llama) need you to accept a licence
   on their model page first; Qwen models don't.

## Done when

`check_env.py` prints a ✓ on every line, including a small model generating text on MPS.
