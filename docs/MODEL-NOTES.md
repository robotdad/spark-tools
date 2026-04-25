# Model Compatibility Notes

Running new models on DGX Spark (GB10 aarch64) is not plug-and-play. Each model
generation brings a different set of container image requirements, vLLM flags,
and tool-call parsers. This document records what works, what doesn't, and why.

Last updated: 2026-04-25

## The Three Things That Break

Every time you try a new model, check these three dimensions:

1. **Container image** — Does the vLLM version in the container have native
   support for the model's architecture? If not, vLLM falls back to the generic
   Transformers backend, which often crashes on MoE models.

2. **vLLM flags** — Newer architectures use different attention mechanisms
   (hybrid Mamba/attention, MoE routing) that conflict with flags like
   `--block-size` and `--attention-backend` that worked fine on older models.

3. **Tool-call parser** — Each model family has its own function-calling format.
   Using the wrong parser means tool calls silently fail or produce garbage.

## Image Sources

Three container image sources exist. They are NOT interchangeable:

| Source | Registry | Entrypoint | When to use |
|--------|----------|------------|-------------|
| **NGC NVIDIA** | `nvcr.io/nvidia/vllm:YY.MM-py3` | `nvidia_entrypoint.sh` → needs `vllm serve` in command | Stable, Blackwell-optimized, but ships older vLLM versions. Good for established models (Qwen 3.x, Llama 4). |
| **Docker Hub (tagged)** | `vllm/vllm-openai:<tag>` | `["vllm", "serve"]` → pass model name directly, no `vllm serve` prefix | Purpose-built for specific models. Use when NGC is too old. Has arm64 manifests. |
| **Docker Hub (nightly)** | `vllm/vllm-openai:latest` | Same as tagged | Bleeding edge. Use as last resort. |

**Entrypoint matters.** NGC images need `vllm serve "$MODEL_NAME"` in the docker
command. Docker Hub images already have `vllm serve` as the entrypoint — pass
only `"$MODEL_NAME"`. Getting this wrong produces:
```
vllm: error: unrecognized arguments: serve <model>
```

When switching between NGC and Docker Hub images, the systemd service file
(`/etc/systemd/system/spark-vllm-standalone.service`) must be updated to match.

## Tested Model Configurations

### google/gemma-4-26B-A4B-it (Gemma 4 MoE)

| Setting | Value | Notes |
|---------|-------|-------|
| **Container** | `vllm/vllm-openai:gemma4` | **Required.** NGC 26.03 (vLLM 0.17.1) does NOT have native Gemma4 support — falls back to Transformers MoE path and crashes on `assert top_k is not None`. Docker Hub `:gemma4` tag ships vLLM 0.19.1.dev6 with native `Gemma4ForConditionalGeneration`. |
| **Entrypoint** | Docker Hub style | Service file must pass `"$MODEL_NAME"` directly, not `vllm serve "$MODEL_NAME"`. |
| **tool-call-parser** | `gemma4` | Not `hermes`, not `qwen3_coder`. |
| **attention-backend** | *omit* (auto-select) | `--attention-backend flashinfer` crashes: `head_size not supported`. vLLM auto-selects `TRITON_ATTN` which works. |
| **block-size** | *omit* | `--block-size 128` crashes with Mamba cache alignment errors on hybrid architectures. |
| **max-num-batched-tokens** | 4096 | Default 2048 is too small for MoE prefill. |
| **kv-cache-dtype** | fp8 | Works. Warning about uncalibrated scales is cosmetic. |
| **Weight format** | BF16 (49 GB) | No working NVFP4 MoE variant as of April 2026 (vLLM issue #39000). |
| **Load time** | ~7 min | 49 GB BF16 on GB10 unified memory. |
| **Capabilities** | Multimodal (text + image), 256K native ctx | Vision via `--mm-processor-kwargs '{"max_soft_tokens": 560}'` for OCR quality. |

Working node.env:
```
VLLM_EXTRA_ARGS="--enable-prefix-caching --kv-cache-dtype fp8 --enable-auto-tool-choice --tool-call-parser gemma4 --max-num-batched-tokens 4096 --max-num-seqs 64 --trust-remote-code"
```

### Qwen/Qwen3.6-35B-A3B-FP8 (Qwen 3.6 MoE)

| Setting | Value | Notes |
|---------|-------|-------|
| **Container** | `nvcr.io/nvidia/vllm:26.03-py3` | Works. vLLM 0.17.1 has native `Qwen3_5MoeForConditionalGeneration`. |
| **Entrypoint** | NGC style | Service file uses `vllm serve "$MODEL_NAME"`. |
| **tool-call-parser** | `qwen3_coder` | Qwen family standard. |
| **attention-backend** | `flashinfer` | Works for Qwen 3.6. |
| **block-size** | *omit* | `--block-size 128` crashes: `In Mamba cache align mode, block_size (2096) must be <= max_num_batched_tokens`. Qwen 3.6 uses hybrid Gated DeltaNet (3/4 linear attention + 1/4 full attention). |
| **max-num-batched-tokens** | 4096 | Required because of Mamba cache alignment. |
| **kv-cache-dtype** | fp8 | Works. |
| **Weight format** | FP8 (35 GB) | Official `Qwen/Qwen3.6-35B-A3B-FP8`. |
| **Load time** | ~5 min | 35 GB FP8 on GB10 unified memory. |
| **Capabilities** | Coding, tool calling, 262K ctx | Best agentic coding model in this size class. |

Working node.env:
```
VLLM_EXTRA_ARGS="--enable-prefix-caching --attention-backend flashinfer --kv-cache-dtype fp8 --enable-auto-tool-choice --tool-call-parser qwen3_coder --max-num-seqs 32 --max-num-batched-tokens 4096 --trust-remote-code"
```

### nvidia/Qwen3-14B-FP4 (previous default)

| Setting | Value | Notes |
|---------|-------|-------|
| **Container** | `nvcr.io/nvidia/vllm:25.09-py3` or any NGC | Older model, broad support. |
| **Entrypoint** | NGC style | |
| **tool-call-parser** | `qwen3_coder` | |
| **attention-backend** | `flashinfer` | Works. |
| **block-size** | 128 | Works — this is a standard transformer, no Mamba layers. |
| **max-num-batched-tokens** | default | |
| **Weight format** | FP4 (10 GB) | NVIDIA-optimized. |

### nvidia/Qwen3-235B-A22B-FP4 (flagship)

| Setting | Value | Notes |
|---------|-------|-------|
| **Container** | `nvcr.io/nvidia/vllm:25.09-py3` or newer NGC | |
| **Weight format** | FP4 (125 GB) | Fits on single GB10 node. |
| **Notes** | Untested with 26.03 container. Likely works since it's standard Qwen3 MoE arch. |

## Flags That Break on Hybrid Architectures

Models after ~March 2026 increasingly use **hybrid attention** (mixing full
attention with linear attention / Mamba / DeltaNet). These architectures break
flags that assumed pure transformer attention:

| Flag | Safe on pure transformers | Breaks on hybrid | Why |
|------|--------------------------|-------------------|-----|
| `--block-size 128` | Yes | **Yes** | Mamba cache alignment requires block_size <= max_num_batched_tokens. The model's internal block size (e.g., 2096) clashes with vLLM's default 2048 batched token limit. |
| `--attention-backend flashinfer` | Yes | **Sometimes** | FlashInfer doesn't support all head sizes. Gemma 4 uses a head size FlashInfer can't handle. Qwen 3.6 works fine with FlashInfer. |
| `--max-num-batched-tokens` (default) | Fine at 2048 | **Too small** | Hybrid models need >= the internal block size. Set to 4096 as a safe default for new models. |

**Safe defaults for unknown new models:**
```
VLLM_EXTRA_ARGS="--enable-prefix-caching --kv-cache-dtype fp8 --enable-auto-tool-choice --tool-call-parser <model-specific> --max-num-batched-tokens 4096 --max-num-seqs 32 --trust-remote-code"
```

Omit `--block-size` and `--attention-backend` until you know they work.

## Tool-Call Parser Reference

| Parser name | Models |
|-------------|--------|
| `qwen3_coder` | Qwen 3.x, Qwen 3.5, Qwen 3.6 |
| `gemma4` | Gemma 4 family |
| `hermes` | Generic (Hermes-format models, Mistral, some fine-tunes) |
| `llama3_json` | Llama 3.x, Llama 4 |
| `glm45` | GLM 4.5+ |

## Multi-Model Split Topology

The proxy (`vllm_proxy.py`) supports running **different models on different
nodes** with model-aware routing. `spark-set-model` does NOT support this — it
syncs the same model.env to both nodes.

To run different models:

1. Edit `~/.config/spark-tools/model.env` on monad (MODEL_NAME=model-A)
2. SSH to dyad, edit its `model.env` (MODEL_NAME=model-B)
3. Edit `~/.config/spark-tools/node.env` on each node with model-specific
   VLLM_EXTRA_ARGS (tool-call-parser, attention-backend, etc.)
4. If models need different container images, edit
   `/etc/systemd/system/spark-vllm-standalone.service` on each node
   (`sudo` required, then `sudo systemctl daemon-reload`)
5. Restart each node's service independently
6. The proxy auto-discovers both models within 10 seconds

Clients specify which model via `"model": "google/gemma-4-26B-A4B-it"` or
`"model": "Qwen/Qwen3.6-35B-A3B-FP8"` in their request body. The proxy
routes to the correct backend.

## Diagnosing New Model Failures

When a new model fails to load, check the logs in this order:

```bash
# 1. Is the container running?
docker ps --format '{{.Names}} {{.Image}} {{.Status}}' | grep vllm

# 2. What does the log say?
docker logs spark-vllm --tail 20

# 3. If it crashed, get the root cause from journalctl
journalctl -u spark-vllm-standalone --no-pager -n 30 | grep "ERROR" | tail -10
```

Common failure signatures:

| Error | Cause | Fix |
|-------|-------|-----|
| `model type "xxx" but Transformers does not recognize this architecture` | Container's transformers too old | Use a newer container image or one tagged for the model |
| `TransformersMultiModalMoEForCausalLM has no vLLM implementation, falling back` followed by `assert top_k is not None` | vLLM doesn't have native support, generic fallback crashes on MoE | Use model-specific tagged container (e.g., `vllm/vllm-openai:gemma4`) |
| `In Mamba cache align mode, block_size (N) must be <= max_num_batched_tokens` | `--block-size` incompatible with hybrid architecture | Remove `--block-size`, add `--max-num-batched-tokens 4096` |
| `Selected backend FLASHINFER is not valid... head_size not supported` | Model's attention head size unsupported by FlashInfer | Remove `--attention-backend flashinfer`, let vLLM auto-select |
| `vllm: error: unrecognized arguments: serve` | Docker Hub image has `vllm serve` as entrypoint, service file also passes `vllm serve` | Remove `vllm serve` from ExecStart, pass only `"$MODEL_NAME"` |
| `pull access denied for vllm-gemma4` | Node doesn't have the custom image | Use the correct image for each node's service file |

## Container Image Versions on This Cluster

| Image | vLLM Version | Transformers | Gemma 4 | Qwen 3.6 | Notes |
|-------|-------------|-------------|---------|----------|-------|
| `nvcr.io/nvidia/vllm:25.09-py3` | 0.12.0 | 4.57.1 | No | No | Original cluster image |
| `nvcr.io/nvidia/vllm:26.03-py3` | 0.17.1 | 4.57.1 | No (crashes) | **Yes** (native) | Good for Qwen 3.x family |
| `vllm/vllm-openai:gemma4` | 0.19.1.dev6 | 5.6.0 | **Yes** (native) | Untested | Purpose-built for Gemma 4 |
| `vllm/vllm-openai:latest` | nightly | latest | Likely | Likely | Untested on GB10 |

## Node.env vs Node.env.monad

The systemd service loads `~/.config/spark-tools/node.env` — NOT
`node.env.monad` or `node.env.dyad`. The named files are reference copies.
When editing node overrides, edit `node.env` on each machine directly.

Load order (last wins): `model.env` → `node.env`