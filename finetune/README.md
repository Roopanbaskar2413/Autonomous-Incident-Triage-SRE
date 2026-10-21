# Fine-tuning the RCA model

The base `llama3.2` model, when prompted directly, sometimes returns a
verbose sentence in the `remediation_action` field (e.g. "Perform a rolling
restart of...") instead of a short, machine-usable slug (e.g.
`rolling_restart`). This directory LoRA fine-tunes the model on a small set
of incident -> RCA/remediation examples so it reliably emits the schema
`graph.py`'s `RCAOutput` expects.

The resulting model is imported into Ollama as `incident-triage-llama` and
used by `graph.py` in place of the stock `llama3.2`.

## Reproducing

Requires an Apple Silicon Mac (uses [MLX](https://github.com/ml-explore/mlx))
and [Ollama](https://ollama.com) running locally.

```bash
pip install mlx-lm gguf

# 1. (Re)generate the training data
python finetune/make_dataset.py

# 2. LoRA fine-tune Llama-3.2-3B-Instruct (a few minutes on an M-series Mac)
mlx_lm.lora --model mlx-community/Llama-3.2-3B-Instruct-4bit \
  --train --data finetune/data --iters 120 --batch-size 1 \
  --val-batches 1 --steps-per-eval 30 --adapter-path finetune/adapters

# 3. Fuse the LoRA adapter into the base model (de-quantized, fp16)
mlx_lm.fuse --model mlx-community/Llama-3.2-3B-Instruct-4bit \
  --adapter-path finetune/adapters --save-path finetune/fused-model \
  --dequantize

# 4. Convert to GGUF via llama.cpp's converter (mlx_lm's own --export-gguf
#    produces a vocab format Ollama's llama.cpp runtime currently rejects)
git clone --depth 1 https://github.com/ggml-org/llama.cpp.git finetune/llama.cpp
python finetune/llama.cpp/convert_hf_to_gguf.py finetune/fused-model \
  --outfile finetune/fused-model/incident-triage-llama-f16.gguf --outtype f16

# 5. Import into Ollama, quantized back down to ~2GB
cd finetune && ollama create incident-triage-llama -f Modelfile -q q4_K_M
```

`finetune/adapters/` (the trained LoRA weights, ~54MB) is kept in the repo so
step 2 can be skipped by resuming from it
(`--resume-adapter-file finetune/adapters/adapters.safetensors`). The fused
model and the llama.cpp clone (steps 3-4, several GB) are intermediate build
artifacts, not checked in — Ollama keeps its own copy of the final quantized
model under `~/.ollama` once step 5 completes.

## Caveat

The training set is intentionally tiny (7 examples) — enough to teach the
output *format*, not incident-diagnosis expertise. The model still relies on
the real telemetry/runbook context passed in the prompt for the actual RCA
content; fine-tuning here only disciplines how that reasoning gets
formatted into fields `graph.py` can parse.
