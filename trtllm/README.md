# TensorRT-LLM for DGX Spark

Scripts for running TensorRT-LLM inference on DGX Spark - both **single-node** and **cluster** (two-node) setups.

**What these scripts solve:**
- Background model downloads that survive SSH disconnects
- Automated cluster-wide downloads
- Pre-flight validation before serving
- Easy health checks (GPU memory, endpoint validation, benchmarking)
- Graceful stop/cleanup without memorizing Docker commands

---

## Quick Start Workflows

### Single-Node Workflow

**One DGX Spark, serving models locally (TP_SIZE=1)**

```bash
# 1. Setup (once)
export HF_TOKEN=hf_...
export PATH="$HOME/spark-tools/trtllm:$PATH"

# 2. Download model
trtllm-download.sh download nvidia/Qwen3-14B-FP4

# 3. Check download progress
trtllm-download.sh status nvidia/Qwen3-14B-FP4

# 4. Start serving (once download complete)
trtllm-serve.sh nvidia/Qwen3-14B-FP4 8355 1

# 5. Validate and benchmark
trtllm-validate.sh nvidia/Qwen3-14B-FP4 8355
trtllm-bench.sh nvidia/Qwen3-14B-FP4 8355

# 6. Stop when done
trtllm-stop.sh
```

### Cluster Workflow (Two Sparks)

**Two DGX Sparks connected via QSFP, tensor parallel across both (TP_SIZE=2)**

```bash
# 1. Setup on BOTH nodes (once, or in .bashrc)
export HF_TOKEN=hf_...
export PATH="$HOME/spark-tools/trtllm:$PATH"
export SPARK_PRIMARY_HOST=monad      # your manager node
export SPARK_SECONDARY_HOST=dyad     # your worker node

# 2. Download model on BOTH nodes (run from primary)
trtllm-new-model.sh nvidia/Qwen3-235B-A22B-FP4 8355 2

# 3. Monitor download status on both nodes
trtllm-model-status.sh nvidia/Qwen3-235B-A22B-FP4

# 4. Start serving (once both show "Incomplete files: 0")
trtllm-serve.sh nvidia/Qwen3-235B-A22B-FP4 8355 2

# 5. Validate across cluster
trtllm-validate.sh nvidia/Qwen3-235B-A22B-FP4 8355
trtllm-bench.sh nvidia/Qwen3-235B-A22B-FP4 8355

# 6. Stop when done
trtllm-stop.sh --stack
```

---

## Prerequisites

### Hardware Requirements

**Single-node:** 1x DGX Spark
- 128 GB unified memory
- Supports models up to ~200B parameters

**Cluster:** 2x DGX Spark
- Connected via QSFP cables (ConnectX-7 ports)
- Supports models up to 405B parameters
- Both nodes must be on same network segment

### Software Prerequisites

**On ALL nodes:**

1. **Docker** with NVIDIA Container Toolkit
   ```bash
   # Verify Docker is running
   docker ps
   
   # Verify NVIDIA Container Toolkit
   docker run --rm --gpus all nvidia/cuda:12.0.0-base-ubuntu22.04 nvidia-smi
   ```

2. **NVIDIA drivers**
   ```bash
   nvidia-smi
   # Should show GPU info
   ```

3. **TRT-LLM container deployed**
   - For cluster: Docker Swarm must be initialized and TRT-LLM stack deployed
   - Stack name: `trtllm-multinode` (or configure scripts)
   - See [NVIDIA Stacked Sparks docs](https://build.nvidia.com/spark/trt-llm/stacked-sparks)

### Network Prerequisites (Cluster Only)

1. **QSFP cable connection** between nodes (ConnectX-7 ports)

2. **Network interfaces configured**
   ```bash
   # Check interfaces are up
   ip addr show enp1s0f0np0
   ip addr show enp1s0f1np1
   ```

3. **Passwordless SSH** from primary → secondary
   ```bash
   # Test from primary node
   ssh $SPARK_SECONDARY_HOST hostname
   # Should return secondary hostname without password prompt
   ```

4. **Docker Swarm initialized** and both nodes joined
   ```bash
   # Check on primary (manager)
   docker node ls
   # Should show both nodes as "Ready"
   ```

5. **GPU resources advertised** to Docker Swarm
   - Each node must have `/etc/docker/daemon.json` configured with GPU UUID
   - NVIDIA Container Runtime configured in `/etc/nvidia-container-runtime/config.toml`
   - See NVIDIA docs for details

### Environment Setup

**Required on ALL nodes:**

```bash
export HF_TOKEN=hf_...  # Your Hugging Face token
```

**Recommended on ALL nodes** (add to `~/.bashrc`):

```bash
export PATH="$HOME/spark-tools/trtllm:$PATH"
export SPARK_PRIMARY_HOST=<your-primary-hostname>
export SPARK_SECONDARY_HOST=<your-secondary-hostname>
```

### Directory Structure

These scripts expect to be installed at:
```
~/spark-tools/trtllm/
```

On **both nodes** for cluster setups.

---

## System Health Checks

Before using these scripts, verify your setup:

### Single-Node Checks

```bash
# 1. Docker and GPU access
docker run --rm --gpus all nvidia/cuda:12.0.0-base-ubuntu22.04 nvidia-smi

# 2. TRT-LLM container running
docker ps | grep trtllm

# 3. GPU memory available
nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits
```

### Cluster Checks

Run these from your **primary** node:

```bash
# 1. Docker Swarm healthy
docker node ls
# Both nodes should show "Ready" and "Active"

# 2. SSH connectivity
ssh $SPARK_SECONDARY_HOST 'echo "Connection OK"'

# 3. Network interfaces on both nodes
ip addr show enp1s0f0np0
ssh $SPARK_SECONDARY_HOST 'ip addr show enp1s0f0np0'

# 4. GPU visible on both nodes
nvidia-smi
ssh $SPARK_SECONDARY_HOST 'nvidia-smi'

# 5. TRT-LLM stack deployed
docker stack ls | grep trtllm-multinode

# 6. Containers running on both nodes
docker stack ps trtllm-multinode
# Should show 2 tasks in "Running" state
```

---

## Script Reference

### Core Scripts

#### `trtllm-container.sh`
Find the local TRT-LLM container ID.

```bash
trtllm-container.sh
# Output: container ID (e.g., 97dc1d9ac139)
```

Used internally by other scripts.

---

#### `trtllm-download.sh`
Download model inside TRT-LLM container (local node only).

```bash
# Start background download
trtllm-download.sh download nvidia/Qwen3-14B-FP4

# Check download status
trtllm-download.sh status nvidia/Qwen3-14B-FP4
```

**Features:**
- Runs in background via `nohup`
- Survives SSH disconnects
- Logs to `/tmp/hf_download_<model>.log` (inside container)

---

#### `cluster-download.sh`
Kick off model download on **both** primary and secondary nodes.

```bash
cluster-download.sh nvidia/Qwen3-235B-A22B-FP4
```

**Requires:** `HF_TOKEN` set, passwordless SSH to secondary.

Starts background downloads on both nodes simultaneously.

---

#### `trtllm-model-status.sh`
Check model download status on **both** nodes.

```bash
trtllm-model-status.sh nvidia/Qwen3-14B-FP4
```

**Output per node:**
- `COMPLETE` - Model fully downloaded, ready to serve
- `IN_PROGRESS` - Download still running (shows incomplete file count)
- `NOT_STARTED` - Model cache directory doesn't exist

---

#### `trtllm-new-model.sh`
**High-level helper:** Download + status for cluster workflow.

```bash
trtllm-new-model.sh nvidia/Qwen3-235B-A22B-FP4 8355 2
```

**Parameters:**
- `MODEL` - Hugging Face model ID
- `PORT` - Serving port (default: 8355)
- `TP_SIZE` - Tensor parallel size (default: 2)

Does:
1. Calls `cluster-download.sh` to start downloads
2. Tells you to monitor with `trtllm-model-status.sh`
3. Tells you the serve command once downloads complete

---

#### `trtllm-serve.sh`
Start TensorRT-LLM server.

```bash
trtllm-serve.sh nvidia/Qwen3-14B-FP4 8355 2 0.9
```

**Parameters:**
- `MODEL` - Hugging Face model ID (required)
- `PORT` - HTTP port (default: 8355)
- `TP_SIZE` - Tensor parallel size (default: 2)
- `GPU_MEM_FRACTION` - KV cache memory fraction (default: 0.9)

**Features:**
- Validates model is fully downloaded before serving
- Configures KV cache memory
- Uses MPI for multi-node (TP_SIZE > 1)

**Blocks until you Ctrl+C.** Run in a `tmux` or `screen` session.

---

#### `trtllm-validate.sh`
Health check: GPU memory + HTTP endpoint test.

```bash
trtllm-validate.sh nvidia/Qwen3-14B-FP4 8355
```

**Checks:**
1. TRT-LLM container exists on primary
2. `nvidia-smi` output on both nodes (cluster only)
3. HTTP `/v1/chat/completions` endpoint responds with valid completion

**Example output:**
```
✅ Endpoint healthy (HTTP 200)
Sample content: "Paris is a beautiful city known for..."
```

---

#### `trtllm-bench.sh`
Simple throughput benchmark.

```bash
trtllm-bench.sh nvidia/Qwen3-14B-FP4 8355 3 256
```

**Parameters:**
- `MODEL` - Model ID
- `PORT` - HTTP port
- `RUNS` - Number of requests (default: 3)
- `MAX_TOKENS` - Max tokens per request (default: 256)

**Output:**
- Per-run: latency, token count, tokens/sec
- Summary: average latency, average tokens/sec

---

#### `trtllm-stop.sh`
Stop serving processes and/or tear down infrastructure.

```bash
# Just kill TRT-LLM processes (default)
trtllm-stop.sh

# Kill processes + remove Docker stack
trtllm-stop.sh --stack

# Kill processes + remove stack + leave swarm
trtllm-stop.sh --swarm
```

**Modes:**
- `--kill-only` (default): `pkill` TRT-LLM processes on both nodes
- `--stack`: Above + `docker stack rm`
- `--swarm`: Above + both nodes leave swarm

---

## Configuration

### Host Names

Scripts default to:
- Primary: `$(hostname)` on the node you run from
- Secondary: `dyad`

**Override with environment variables:**
```bash
export SPARK_PRIMARY_HOST=spark-001
export SPARK_SECONDARY_HOST=spark-002
```

### Stack Name

Scripts expect Docker stack named: `trtllm-multinode`

To change: edit `STACK_NAME` in `trtllm-stop.sh` or container name pattern in `trtllm-container.sh`.

### GPU Memory Fraction

Default: `0.9` (NVIDIA recommended)

Override per-invocation:
```bash
trtllm-serve.sh nvidia/Qwen3-14B-FP4 8355 2 0.75
```

---

## Troubleshooting

### "Could not find TRT-LLM container"

**Check:**
```bash
docker ps | grep trtllm
```

**Fix:**
- Single-node: Ensure TRT-LLM container is running
- Cluster: Ensure stack is deployed: `docker stack ps trtllm-multinode`

### Model download hangs or fails

**Check logs:**
```bash
trtllm-download.sh status nvidia/Qwen3-14B-FP4
```

**Common causes:**
- Invalid `HF_TOKEN`
- No disk space: `df -h`
- Network issues

### "Model not ready" when serving

**Check status:**
```bash
trtllm-model-status.sh nvidia/Qwen3-14B-FP4
```

Wait until both nodes show: `Incomplete files: 0`

### MPI/network warnings during serve

```
UCX WARN network device 'enp1s0f0np0' is not available
```

**This is usually OK** - means only one of two CX-7 ports is used. Ignore if inference works.

### Secondary node unreachable

**Test SSH:**
```bash
ssh $SPARK_SECONDARY_HOST hostname
```

**Fix:**
- Ensure passwordless SSH is set up
- Check `SPARK_SECONDARY_HOST` is correct
- Verify secondary node is powered on and networked

### Low throughput

**Check:**
1. GPU memory not saturated:
   ```bash
   nvidia-smi
   ```
   If memory is near full, reduce `GPU_MEM_FRACTION`

2. Both GPUs active (cluster):
   ```bash
   trtllm-validate.sh <model> <port>
   ```
   Should show GPU usage on both nodes

3. Model size appropriate for hardware:
   - Single Spark: Up to ~200B parameters
   - Two Sparks: Up to 405B parameters

---

## Model Size Guidelines

**Single DGX Spark (128GB):**
- Up to ~200B parameters
- Examples: Qwen3-14B, Llama-3-70B, Mixtral-8x22B

**Two DGX Sparks (256GB total):**
- Up to 405B parameters
- Examples: Qwen3-235B, Llama-3.3-405B

**TP_SIZE recommendations:**
- Single node: `TP_SIZE=1`
- Two nodes: `TP_SIZE=2`

---

## Additional Resources

- [DGX Spark Hardware Docs](https://docs.nvidia.com/dgx/dgx-spark/hardware.html)
- [NVIDIA Stacked Sparks Setup](https://build.nvidia.com/spark/trt-llm/stacked-sparks)
- [TensorRT-LLM Documentation](https://nvidia.github.io/TensorRT-LLM/)

---

## Contributing

This toolset is designed for DGX Spark but can be adapted for other NVIDIA hardware with unified memory architectures.

**To adapt:**
1. Adjust host names via environment variables
2. Update Docker stack/container names in scripts
3. Modify GPU memory fractions based on your hardware
