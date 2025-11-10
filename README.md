# spark-tools

A small collection of scripts I use to poke at NVIDIA DGX Spark nodes.

Right now this is mostly focused on standing up multi-node [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) on a two-node Docker Swarm (my “monad” and “dyad” hosts), downloading huge models from Hugging Face, and validating that everything is actually running.

Over time I expect to throw more Spark-related utilities in here.

## Layout

```text
spark-tools/
  README.md          # this file

  trtllm/            # TensorRT-LLM cluster helpers
    README.md
    trtllm-container.sh
    trtllm-download.sh
    cluster-download.sh
    trtllm-model-status.sh
    trtllm-serve.sh
    trtllm-validate.sh
    trtllm-bench.sh
    trtllm-stop.sh
```

## Basic expectations

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


For details on the TensorRT-LLM workflow itself, see [`trtllm/README.md`](trtllm/README.md).
