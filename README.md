# spark-tools

A small collection of scripts I use to poke at NVIDIA DGX Spark nodes.

This includes utilities for running LLM inference on DGX Spark hardware using both [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) (multi-node clusters) and [vLLM](https://github.com/vllm-project/vllm) (single-node inference with multimodal support).

## Layout

```text
spark-tools/
  README.md          # this file

  trtllm/            # TensorRT-LLM cluster helpers (multi-node)
    README.md
    trtllm-container.sh
    trtllm-download.sh
    cluster-download.sh
    trtllm-model-status.sh
    trtllm-serve.sh
    trtllm-validate.sh
    trtllm-bench.sh
    trtllm-stop.sh

  vllm/              # vLLM single-node helpers (multimodal/vision)
    README.md
    vllm-serve.sh
    vllm-stop.sh
    vllm-validate.sh
    vllm-test.py
```

## Tool Selection Guide

### Use TensorRT-LLM (`trtllm/`) when:
- Running **multi-node** inference (tensor parallelism across 2+ DGX Sparks)
- Need absolute minimum latency for production workloads
- Working with very large models (200B+ parameters)
- Have existing TensorRT-LLM setup/infrastructure

### Use vLLM (`vllm/`) when:
- Running **single-node** inference
- Need **multimodal/vision** capabilities (image analysis)
- Want OpenAI-compatible API out of the box
- Need to experiment with different models quickly
- Prefer higher throughput over absolute minimum latency

## Quick Start Examples

### vLLM (Vision Model)
```bash
export PATH="$HOME/spark-tools/vllm:$PATH"
docker pull nvcr.io/nvidia/vllm:25.09-py3
vllm-serve.sh
vllm-validate.sh
vllm-test.py photo.jpg
```

### TensorRT-LLM (Multi-Node)
```bash
export PATH="$HOME/spark-tools/trtllm:$PATH"
export HF_TOKEN=hf_...
trtllm-new-model.sh nvidia/Qwen3-235B-A22B-FP4 8355 2
trtllm-model-status.sh nvidia/Qwen3-235B-A22B-FP4
trtllm-serve.sh nvidia/Qwen3-235B-A22B-FP4 8355 2
```

## TensorRT-LLM Setup (Multi-Node)

These scripts are written around a two-node DGX Spark setup, but are configurable:

- Two Spark nodes:
  - **Primary node** (Swarm manager) – defaults to the local hostname
  - **Secondary node** (Swarm worker) – defaults to `dyad`

- Hostnames:
  - By default, the scripts assume:
    - `PRIMARY_HOST = $(hostname)`
    - `SECONDARY_HOST = dyad`
  - You can override this via environment variables:
    - `export SPARK_PRIMARY_HOST=<your-primary-host>`
    - `export SPARK_SECONDARY_HOST=<your-secondary-host>`

- Layout on each node:
  - Scripts live in: `~/spark-tools/trtllm`
  - (Recommended) add to your PATH on **both** nodes:
    - `export PATH="$HOME/spark-tools/trtllm:$PATH"`

- Docker / Swarm expectations:
  - Docker Swarm configured with a TRT-LLM multinode stack:
    - Stack name: `trtllm-multinode`
    - Service name: `trtllm`
  - The TRT-LLM service is the one running the container we exec into.

- Connectivity:
  - Passwordless SSH from **primary → secondary** is required
    - e.g. from `$SPARK_PRIMARY_HOST` you can run:
      - `ssh $SPARK_SECONDARY_HOST hostname`

- Hugging Face auth:
  - `HF_TOKEN` must be exported in your shell on **both** nodes:
    - `export HF_TOKEN=hf_...`
  - The download scripts pass this into the TRT-LLM container.

If your topology is different (more nodes, different hostnames, different stack name), you can still use these scripts as a starting point — just update the host env vars, stack name, and paths to match your environment.

## Documentation

For detailed documentation on each tool:

- **TensorRT-LLM**: See [`trtllm/README.md`](trtllm/README.md)
- **vLLM**: See [`vllm/README.md`](vllm/README.md)
