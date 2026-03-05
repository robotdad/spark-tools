# spark-tools — Architecture

## Cluster Overview

| Node  | Role   | Default Hostname | Services                                     |
|-------|--------|------------------|----------------------------------------------|
| monad | Head   | `monad`          | Ray Head / Swarm Manager, vLLM API, Proxy, WebUI |
| dyad  | Worker | `dyad`           | Ray Worker / Swarm Worker                    |

IP addresses and the QSFP interface are auto-detected by `install.sh` and stored in
`~/.config/spark-tools/cluster.env`. Hostnames are written into `/etc/hosts` on both
nodes so that `monad` and `dyad` resolve without a DNS server.

---

## Network Topology

```text
         ┌─────────────────────────────────────────────┐
         │                 monad                       │
         │          (Head / Swarm Manager)             │
         │                                             │
 LAN     │  spark-proxy   :9000  (auth + LB proxy)    │
─────────┤  vLLM / TRT-LLM :8000  (inference API)     │
         │  Open WebUI    :8080  (chat UI)             │
         │                                             │
 QSFP    │  NCCL / UCX / MPI over QSFP interface      │
═════════╡  (200 Gb/s, link-local 169.254.x.x)        │
         └─────────────────────────────────────────────┘

         ┌─────────────────────────────────────────────┐
         │                  dyad                       │
         │             (Worker Node)                   │
         │                                             │
 LAN     │  vLLM / TRT-LLM :8000  (split mode only)  │
─────────┤                                             │
         │  NCCL / UCX / MPI over QSFP interface      │
 QSFP    │  (200 Gb/s, link-local 169.254.x.x)        │
═════════╡                                             │
         └─────────────────────────────────────────────┘
```

- **LAN** is used for management, SSH, Ray coordination, and Docker Swarm control.
- **QSFP** (usually `enp1s0f0np0`) carries GPU-to-GPU traffic: NCCL all-reduce for Ray/vLLM,
  MPI collective ops for TRT-LLM. Link-local addressing (169.254.x.x) means it works with
  no DHCP and survives router changes.
- In **split mode** dyad also exposes its own inference port on LAN so the proxy can
  reach it.

---

## Operating Modes

spark-tools supports three distinct operating modes. Switch with `spark-mode`.

### Mode 1: Ray Cluster (vLLM TP=2)

Two nodes form a single Ray cluster. vLLM runs with tensor-parallelism across both GB10s
as one logical device.

```
monad                              dyad
┌────────────────────────────┐    ┌────────────────────────────┐
│ spark-ray-head.service     │    │ spark-ray-worker.service   │
│  └─ Ray Head (port 6379)   │◄──►│  └─ Ray Worker             │
│                            │    │                            │
│ spark-ray-vllm.service     │    │  (GPU worker managed by    │
│  └─ vLLM (TP=2 via Ray)   │    │   Ray; no separate API)    │
│     port 8000              │    │                            │
└────────────────────────────┘    └────────────────────────────┘
         ▲
         │ NCCL/UCX over QSFP (200 Gb/s)
         ▼
       clients → monad:8000
```

**When to use:**
- Unified model weights across both GPUs (large models > single-GPU VRAM)
- You prefer a single API endpoint with no client-side awareness of nodes
- vLLM engine only (`SPARK_ENGINE=vllm`)

**Known limitation:** `--enforce-eager` is required on GB10 at TP=2 to avoid a Triton
allocator crash during CUDA graph capture. This disables CUDA graphs, reducing throughput.
See [TROUBLESHOOTING.md](TROUBLESHOOTING.md#triton-allocator-crash).

---

### Mode 2: Docker Swarm (TRT-LLM or vLLM)

Docker Swarm manages a multi-node stack. For TRT-LLM, MPI is used for inter-node
tensor parallelism. For vLLM in swarm mode, each node runs independently (effectively
a swarm-managed split).

```
monad                              dyad
┌────────────────────────────┐    ┌────────────────────────────┐
│ spark-swarm-stack.service  │    │ (Swarm worker; services    │
│  └─ Docker Swarm Manager   │◄──►│   scheduled here by        │
│     Stack: trtllm-multi    │    │   Swarm manager)           │
│                            │    │                            │
│  TRT-LLM container         │    │  TRT-LLM container         │
│   (MPI rank 0)             │    │   (MPI rank 1)             │
│   port 8355                │    │                            │
└────────────────────────────┘    └────────────────────────────┘
         ▲
         │ MPI over QSFP (200 Gb/s)
         ▼
       clients → monad:8355
```

**When to use:**
- TRT-LLM engine (`SPARK_ENGINE=trtllm`) — requires swarm mode
- Highest throughput for production batch workloads
- Docker Swarm already configured with `trtllm-multinode` stack

**Note:** `ray + trtllm` is explicitly unsupported. TRT-LLM requires Swarm/MPI.

---

### Mode 3: Split (Independent TP=1 Instances)

Each node runs a **completely independent** vLLM instance at TP=1. No Ray, no MPI,
no inter-node GPU communication. The proxy on monad load-balances across both.

```
monad                              dyad
┌────────────────────────────┐    ┌────────────────────────────┐
│ spark-vllm-standalone      │    │ spark-vllm-standalone      │
│  └─ vLLM (TP=1)           │    │  └─ vLLM (TP=1)           │
│     port 8000              │    │     port 8000              │
│     node.env.monad overlay │    │     node.env.dyad overlay  │
└────────────────────────────┘    └────────────────────────────┘
         ▲                                    ▲
         └──────────────┬─────────────────────┘
                        │ model-aware routing + least-connections
                  monad:9000 (spark-proxy)
                        ▲
                     clients
```

**When to use:**
- Best latency and throughput for models that fit in a single GB10 (e.g., Qwen3-235B-FP4)
- CUDA graphs work correctly at TP=1 — no `--enforce-eager` needed
- Want per-node tuning (different `GPU_MEM_UTIL`, `MAX_MODEL_LEN`, or `VLLM_EXTRA_ARGS`)
- Each node can serve different models simultaneously

**Performance impact (vs. untuned Ray cluster):**

| Metric                   | Ray cluster (untuned) | Split mode (tuned) |
|--------------------------|-----------------------|--------------------|
| Short prompt latency     | 11.6 s                | 0.3 s              |
| Long generation (500 tok)| 30.6 s                | ~11 s              |
| Throughput               | ~16 tok/s             | ~45 tok/s          |

Key optimizations that split mode enables: CUDA graphs (no `--enforce-eager`), FP8 KV
cache, `FLASHINFER_MOE_BACKEND=latency`, `vm.swappiness=1`, periodic `drop_caches`.

**Switching modes:**
```bash
spark-mode split    # switch to independent TP=1
spark-mode ray      # switch to Ray TP=2 cluster
spark-mode swarm    # switch to Docker Swarm (TRT-LLM default)
```

---

## Backend Dispatch Architecture

All user-facing commands in `bin/` share a common library (`lib/spark-common.sh`) and
delegate to the active backend via a consistent dispatch pattern:

```
bin/spark-serve
bin/spark-stop         ──► lib/spark-common.sh ──► backends/{SPARK_MODE}/{action}.sh
bin/spark-status            spark_dispatch()
bin/spark-bench
```

### Command → Backend Flow

```
User runs: spark-serve
    │
    ├─ sources lib/spark-common.sh
    │     ├─ spark_load_config()     loads cluster.env → model.env → node.env.*
    │     ├─ spark_validate_mode()   checks SPARK_MODE / SPARK_ENGINE combination
    │     └─ spark_dispatch("serve") resolves backends/${SPARK_MODE}/serve.sh
    │
    └─ exec backends/swarm/serve.sh   (or ray/serve.sh)
           └─ mode-specific logic
```

### Backend Directory Layout

```
backends/
  ray/
    serve.sh       # Start Ray Head + vLLM worker (monad) / Ray worker (dyad)
    stop.sh        # Stop Ray services
    status.sh      # Check Ray cluster state
  swarm/
    serve.sh       # Deploy Swarm stack (TRT-LLM or vLLM)
    stop.sh        # Remove Swarm stack
    status.sh      # Check Swarm service state
```

### Supported Combinations

| `SPARK_MODE` | `SPARK_ENGINE` | Backend Used         | Notes                        |
|--------------|----------------|----------------------|------------------------------|
| `swarm`      | `trtllm`       | `backends/swarm/`    | Default; MPI tensor-parallel |
| `swarm`      | `vllm`         | `backends/swarm/`    | vLLM via Swarm               |
| `ray`        | `vllm`         | `backends/ray/`      | Ray tensor-parallel (TP=2)   |
| `ray`        | `trtllm`       | ❌ Not supported     | TRT-LLM requires Swarm/MPI   |

---

## Proxy Architecture

The auth proxy (`proxy/vllm_proxy.py`, managed by `spark-proxy.service`) provides:

1. **Bearer-token authentication** — all `/v1/*` requests require `Authorization: Bearer <SECRET>`
2. **Model-aware routing** — requests specifying a `model` are forwarded to the backend(s)
   currently serving that model
3. **Least-connections load balancing** — ties (or unknown model) broken by active request count
4. **Health monitoring** — each backend polled every 10 s; unhealthy backends removed from rotation
5. **Model aggregation** — `GET /v1/models` returns the union of models from all healthy backends

```
Client
  │  Authorization: Bearer <SECRET>
  │  {"model": "nvidia/Qwen3-...", "messages": [...]}
  ▼
spark-proxy (:9000)
  │
  ├── model in routing table?
  │     YES → pick least-busy backend serving that model
  │     NO  → pick least-busy healthy backend (or 404 if model unknown)
  │
  ├── backend.healthy == True?
  │     YES → forward (buffered or streaming)
  │     NO  → 503 All backends down
  │
  └── stream: True?
        YES → StreamingResponse (SSE pass-through)
        NO  → buffered JSONResponse
```

**Configuration:**

| Variable               | Default                              | Purpose                        |
|------------------------|--------------------------------------|--------------------------------|
| `VLLM_PROXY_SECRET`    | (required)                           | Bearer token value             |
| `VLLM_PROXY_SECRET_FILE` | `~/.config/vllm-proxy/secret.env` | File containing secret         |
| `VLLM_BACKENDS`        | `http://monad:8000,http://dyad:8000` | Comma-separated backend URLs   |
| `VLLM_HEALTH_INTERVAL` | `10`                                 | Health check interval (seconds)|

**Secret management:**
```bash
# Generate a new secret
openssl rand -hex 32 > ~/.config/vllm-proxy/secret.env

# Rotate (generates new secret + restarts proxy)
~/spark-tools/scripts/rotate-proxy-secret.sh
```

---

## Configuration Hierarchy

Config is layered: later files override earlier ones. All files live in
`~/.config/spark-tools/` by default (override with `$SPARK_CONFIG_DIR`).

```
cluster.env          ← cluster topology, mode, engine, network, images
     ↓ (overrides)
model.env            ← model identity, parallelism, memory, vLLM/TRT-LLM args
     ↓ (overrides)
node.env.<hostname>  ← per-node tuning (loaded on that node only)
```

### cluster.env

Controls what gets started and how nodes find each other:

| Key                   | Example            | Purpose                                |
|-----------------------|--------------------|----------------------------------------|
| `SPARK_MODE`          | `swarm`            | Orchestration backend (swarm / ray)    |
| `SPARK_ENGINE`        | `trtllm`           | Inference engine (trtllm / vllm)       |
| `SPARK_PRIMARY_HOST`  | `monad`            | Head node hostname                     |
| `SPARK_SECONDARY_HOST`| `dyad`             | Worker node hostname                   |
| `SPARK_QSFP_IFACE`    | `enp1s0f0np0`      | High-speed inter-node interface        |
| `SPARK_PORT`          | `8000`             | vLLM API port                          |
| `SPARK_PROXY_PORT`    | `9000`             | Auth proxy port                        |
| `SPARK_PROXY_ENABLED` | `false`            | Start proxy service automatically      |

### model.env

Controls what model runs and how it's configured:

| Key                     | Example                         | Purpose                          |
|-------------------------|---------------------------------|----------------------------------|
| `MODEL_NAME`            | `nvidia/Qwen3-235B-A22B-FP4`   | HuggingFace model repo           |
| `SERVED_MODEL_NAME`     | `qwen3-coder`                   | API alias (optional)             |
| `TP_SIZE`               | `2`                             | Tensor-parallel degree           |
| `MAX_MODEL_LEN`         | `32768`                         | Max context length (tokens)      |
| `GPU_MEM_UTIL`          | `0.85`                          | GPU memory fraction for KV cache |
| `VLLM_EXTRA_ARGS`       | `--enforce-eager ...`           | Additional vLLM CLI args         |
| `TRTLLM_PORT`           | `8355`                          | TRT-LLM listen port              |
| `TRTLLM_MAX_BATCH_SIZE` | `4`                             | TRT-LLM batch size               |

### node.env.\<hostname\>

Loaded after `model.env` on each node. Any key here overrides the shared value.
Primary use is split mode tuning:

**`node.env.monad`** (moderate profile — monad hosts other services too):
- `GPU_MEM_UTIL=0.80`, `MAX_MODEL_LEN=131072`, `--max-num-seqs 64`
- No `--enforce-eager` (TP=1 doesn't need it)

**`node.env.dyad`** (aggressive profile — dyad is dedicated compute):
- `GPU_MEM_UTIL=0.90`, `MAX_MODEL_LEN=262144`, `--max-num-seqs 32`
- Higher context ceiling thanks to more VRAM headroom

---

## Systemd Service Layout

Services are generated from templates in `systemd/` by `install.sh`:

| Template                            | Installed As                      | Role                        |
|-------------------------------------|-----------------------------------|-----------------------------|
| `spark-ray-head.service.template`   | `spark-ray-head.service`          | Ray Head daemon (monad)     |
| `spark-ray-worker.service.template` | `spark-ray-worker.service`        | Ray Worker daemon (dyad)    |
| `spark-ray-vllm.service.template`   | `spark-ray-vllm.service`          | vLLM on top of Ray (monad)  |
| `spark-swarm-stack.service.template`| `spark-swarm-stack.service`       | Docker Swarm stack lifecycle|
| `spark-proxy.service.template`      | `spark-proxy.service`             | Auth proxy (monad)          |
| `spark-open-webui.service.template` | `spark-open-webui.service`        | Open WebUI (monad)          |

Template placeholders resolved by `install.sh`:

| Placeholder       | Value                        |
|-------------------|------------------------------|
| `{{USERNAME}}`    | Linux username               |
| `{{HOME_DIR}}`    | User home directory          |
| `{{HEAD_IP}}`     | monad IP on QSFP network     |
| `{{WORKER_IP}}`   | dyad IP on QSFP network      |
| `{{QSFP_IFACE}}`  | QSFP interface name          |

---

## Tiktoken Encodings

GPT-OSS compatible models require tiktoken vocab files mounted into the container:

```
encodings/
  cl100k_base.tiktoken    # GPT-4 tokenizer
  o200k_base.tiktoken     # GPT-4o tokenizer
```

`install.sh` copies these to `/etc/encodings/` on both nodes and ensures containers
mount `-v /etc/encodings:/etc/encodings`.

---

## Design Philosophy

- **systemd manages everything** — services start on boot, restart on failure, log to journald
- **Docker isolates runtimes** — GPU drivers stay on host; engine versions are container images
- **One config directory** — `~/.config/spark-tools/` holds all runtime state
- **Templates enable portability** — swap IPs/users without editing service files manually
- **Modes are explicit** — no magic auto-detection at runtime; `SPARK_MODE` + `SPARK_ENGINE` determine everything
- **Backends are pluggable** — adding a new orchestration backend is `backends/newmode/{serve,stop,status}.sh`
