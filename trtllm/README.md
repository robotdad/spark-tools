# trtllm scripts

Helpers for running TensorRT-LLM in multi-node mode on a two-node DGX Spark cluster using Docker Swarm, **and** for poking at models on a single node.

The goal is to make it **much harder** to screw up things like:

- “Did I download the model on both nodes?”
- “Is the swarm actually healthy?”
- “Is the model serving and answering requests?”
- “How much GPU memory is this thing using?”
- “Will this long HF download die when my SSH session drops?”

…and to give me a repeatable “new model” workflow instead of a giant pile of copy-pasted commands.

---

## Host / environment assumptions

These scripts are written around a two-node Spark setup but are configurable.

### Host naming

On each node, the scripts compute:

- `PRIMARY_HOST` – defaults to `$(hostname)`  
  Override with:  
  ```bash
  export SPARK_PRIMARY_HOST=<your-primary-host>
  ```
- `SECONDARY_HOST` – defaults to `dyad`  
  Override with:  
  ```bash
  export SPARK_SECONDARY_HOST=<your-secondary-host>
  ```

In my setup:

- Primary: `monad` (Swarm manager)
- Secondary: `dyad` (Swarm worker)

### Script layout / PATH

I keep everything here **on both nodes**:

```bash
~/spark-tools/trtllm
```

Recommended: add to your `PATH` (on both nodes):

```bash
export PATH="$HOME/spark-tools/trtllm:$PATH"
```

That lets you just run:

```bash
trtllm-new-model.sh nvidia/Qwen3-14B-FP4
```

from anywhere, instead of cd’ing into the folder all the time.

### Docker / Swarm expectations

These scripts assume:

- Docker Swarm is initialized
- You’ve deployed the TRT-LLM multinode stack using NVIDIA’s `docker-compose.yml`

By default they expect:

- Stack name: `trtllm-multinode`
- Service name: `trtllm`

The helper script `trtllm-container.sh` looks up the running TRT-LLM service container and returns its ID. Everything else builds on top of that.

### Connectivity

- Passwordless SSH from **primary → secondary**:
  - From `$SPARK_PRIMARY_HOST` you should be able to run:
    ```bash
    ssh $SPARK_SECONDARY_HOST hostname
    ```
- For **single-node** use (no remote host), just don’t use the cluster scripts, or set:
  ```bash
  export SPARK_SECONDARY_HOST=$SPARK_PRIMARY_HOST
  ```
  so remote checks don’t complain.

### Hugging Face token

On **both nodes**:

```bash
export HF_TOKEN=hf_...
```

The download scripts will pass this into the TRT-LLM container and use `huggingface-cli download`.

---

## Script overview

### 1. `trtllm-container.sh`

> Find the TRT-LLM container ID on the **local** node.

```bash
trtllm-container.sh
```

Outputs a single container ID, e.g.:

```text
97dc1d9ac139
```

Every other script uses this to `docker exec` into the right place.

---

### 2. `trtllm-download.sh`

> Download (or resume) a Hugging Face model **inside the TRT-LLM container on this node**, in the background.

Usage:

```bash
trtllm-download.sh start  nvidia/Qwen3-14B-FP4
trtllm-download.sh status nvidia/Qwen3-14B-FP4
```

- `start`:
  - Runs `huggingface-cli download <model>` inside the container via `nohup`.
  - Writes a log: `/tmp/hf_download_<sanitized-model-name>.log` (inside the container).
  - Keeps running even if your SSH session disconnects.
- `status`:
  - Shows the tail of that log so you can see what it’s doing.

This script is **local-only**. It only talks to the TRT-LLM container on whatever node you’re on.

Works fine for both:

- Single-node (just run this on your one host)
- Swarm/multi-node (used indirectly by `cluster-download.sh`)

---

### 3. `cluster-download.sh`

> Kick off the same model download on **both** primary and secondary nodes in one shot.

Usage (run on the **primary** node):

```bash
cluster-download.sh nvidia/Qwen3-14B-FP4
```

What it does:

- Determines:
  - `PRIMARY_HOST` (from `SPARK_PRIMARY_HOST` or `hostname`)
  - `SECONDARY_HOST` (from `SPARK_SECONDARY_HOST` or default `dyad`)
- On the **primary** host:
  - Calls `trtllm-download.sh start <model>`
- On the **secondary** host:
  - SSHes into `$SECONDARY_HOST`
  - Ensures `PATH` includes `~/spark-tools/trtllm`
  - Calls `trtllm-download.sh start <model>` there too

It prints a reminder at the end of how to check status.

---

### 4. `trtllm-model-status.sh`

> Check whether a model’s cache is **complete** on primary and secondary.

Usage:

```bash
trtllm-model-status.sh nvidia/Qwen3-14B-FP4
```

Example output:

```text
Primary host:   monad
Secondary host: dyad

============================================================
Node: monad (local)
Container: 97dc1d9ac139
Model cache path: /root/.cache/huggingface/hub/models--nvidia--Qwen3-14B-FP4
Status: COMPLETE
  Total files:      17
  Incomplete files: 0
  Approx size:      7.0G

============================================================
Node: dyad (via ssh)
Container: 83b4a5c91a0e
Model cache path: /root/.cache/huggingface/hub/models--nvidia--Qwen3-14B-FP4
Status: COMPLETE
  Total files:      17
  Incomplete files: 0
  Approx size:      7.0G
```

Under the hood (inside each container) it:

- Checks for `/root/.cache/huggingface/hub/models--<org>--<name>`
- Counts:
  - All files
  - `*.incomplete` blobs
- Computes `du -sh` on the model directory
- Classifies status as:
  - `COMPLETE` – no `*.incomplete`, and files exist
  - `IN_PROGRESS` – some `*.incomplete`
  - `NOT_STARTED` / `UNKNOWN` – no cache dir or nothing useful

On single-node, this is overkill; you can still use it, but the “remote” part is only useful if `SPARK_SECONDARY_HOST` points at a second machine.

---

### 5. `trtllm-serve.sh`

> Start the TensorRT-LLM multi-node server for a given model (from the primary node).

Usage:

```bash
trtllm-serve.sh nvidia/Qwen3-14B-FP4 8355 2
```

Parameters:

- `MODEL` – Hugging Face model id (e.g. `nvidia/Qwen3-14B-FP4`)
- `PORT` – HTTP port (inside the container) for Uvicorn (default `8355`)
- `TP_SIZE` – tensor parallel size (default `2` for a two-node setup)

What it does:

- Finds the local TRT-LLM container
- Runs `mpirun` inside it, using the OpenMPI hostfile (`/etc/openmpi-hostfile`) set up during stack deploy
- Starts:
  - Rank 0: `trtllm-serve` (HTTP server) + management node
  - Rank 1: worker node on the other host

If all goes well, you’ll eventually see:

```text
INFO:     Uvicorn running on http://localhost:8355 (Press CTRL+C to quit)
```

On a **single node**, you can still use this pattern (with `tp_size = 1`), but the multi-node bits (OpenMPI hostfile etc.) don’t matter.

---

### 6. `trtllm-validate.sh`

> Sanity-check that the model is serving and look at GPU memory usage on both nodes.

Usage:

```bash
trtllm-validate.sh nvidia/Qwen3-14B-FP4 8355
```

It does:

1. **Container sanity check** on the primary:
   - Uses `trtllm-container.sh` and errors if nothing is running.
2. **GPU usage**:
   - Runs `nvidia-smi` on the primary host.
   - SSHes to the secondary host and runs `nvidia-smi` there as well.
3. **HTTP endpoint check**:
   - Sends a small OpenAI-style `chat/completions` request to:
     ```text
     http://localhost:<PORT>/v1/chat/completions
     ```
   - Verifies:
     - HTTP 2xx
     - Response JSON has a `choices[0].message.content`
   - Prints a short snippet of the content.

Example HTTP bit:

```text
=== HTTP endpoint check ===
Checking http://localhost:8355/v1/chat/completions...

✅ Endpoint healthy (HTTP 200)
Sample content: "<think>
Okay, the user is asking about a health check from a script called"...
```

On a **single node**, this will still work, but the “secondary” `nvidia-smi` part will only succeed if `SPARK_SECONDARY_HOST` points to something real (could just be `PRIMARY_HOST` too).

---

### 7. `trtllm-bench.sh`

> Send a few requests and compute basic latency / throughput numbers.

Usage:

```bash
# defaults: 3 runs, max_tokens=256
trtllm-bench.sh nvidia/Qwen3-14B-FP4 8355

# override run count and max_tokens
trtllm-bench.sh nvidia/Qwen3-14B-FP4 8355 5 256
```

Parameters:

- `MODEL`
- `PORT`
- `RUNS` (default: 3)
- `MAX_TOKENS` (default: 256)

For each run:

- Uses `curl` to send a `chat/completions` request with `max_tokens = MAX_TOKENS`.
- Measures latency (wall-clock).
- Parses the JSON and extracts `usage.total_tokens`.
- Computes tokens/sec for that run.

Then it prints a summary:

```text
=== Summary ===
Effective runs: 5
Total time:     73.547 s
Total tokens:   1254
Avg latency:    14.709 s
Avg tokens/s:   17.05
```

This is just a “sanity perf” check, not a full benchmark harness.

Works fine in both multi-node and single-node cases as long as the server is up and listening on the given port.

---

### 8. `trtllm-stop.sh`

> Stop the serving processes, optionally remove the stack, optionally tear down the Swarm.

Usage:

```bash
# default: just kill the TRT-LLM processes inside containers on both nodes
trtllm-stop.sh

# kill processes + remove Docker stack
trtllm-stop.sh --stack

# kill processes + remove stack + have both nodes leave the Swarm
trtllm-stop.sh --swarm
```

Modes:

- `--kill-only` (default):
  - On primary and secondary:
    - Find the TRT-LLM container.
    - `pkill` any `trtllm-llmapi-launch`, `trtllm-serve`, `uvicorn` processes inside.
- `--stack`:
  - Does `--kill-only`, then on the primary:
    - `docker stack rm trtllm-multinode`
- `--swarm`:
  - Does `--stack`, then:
    - SSH to secondary: `docker swarm leave --force`
    - On primary: `docker swarm leave --force`

If you’re just bouncing models, `--kill-only` is usually enough.

---

### 9. `trtllm-mn-entrypoint.sh`

This is basically the entrypoint script NVIDIA provides for the multi-node TRT-LLM container. You normally **don’t** call this by hand; it’s used by the `docker-compose.yml` stack.

I keep it here so everything needed to recreate the stack/container behavior is in one place.

---

## Quick workflows

### A. Multi-node Swarm workflow (two Spark nodes)

On **both nodes** (once per shell / in your `.bashrc`):

```bash
export HF_TOKEN=hf_...
export PATH="$HOME/spark-tools/trtllm:$PATH"
export SPARK_PRIMARY_HOST=monad        # or your actual manager hostname
export SPARK_SECONDARY_HOST=dyad      # or your actual worker hostname
```

Then on the **primary** host:

```bash
cd ~/spark-tools/trtllm

# 1. New model helper: kicks off downloads on both nodes
trtllm-new-model.sh nvidia/Qwen3-14B-FP4 8355 2

# (Behind the scenes this runs cluster-download.sh for you.)

# 2. Poll until both nodes are done
trtllm-model-status.sh nvidia/Qwen3-14B-FP4

# 3. Start serving
trtllm-serve.sh nvidia/Qwen3-14B-FP4 8355 2

# 4. Validate
trtllm-validate.sh nvidia/Qwen3-14B-FP4 8355

# 5. Sanity benchmark
trtllm-bench.sh nvidia/Qwen3-14B-FP4 8355

# 6. When done for the day
trtllm-stop.sh --stack         # or --kill-only, or --swarm if you want to nuke the Swarm
```

### B. Single-node workflow (no Swarm, just one Spark box)

On your single node:

```bash
export HF_TOKEN=hf_...
export PATH="$HOME/spark-tools/trtllm:$PATH"
export SPARK_PRIMARY_HOST=$(hostname)
# Optional: quiet down remote checks by pointing SECONDARY at the same host
export SPARK_SECONDARY_HOST=$SPARK_PRIMARY_HOST
```

Then:

```bash
cd ~/spark-tools/trtllm

# 1. Download the model inside the local TRT-LLM container
trtllm-download.sh start nvidia/Qwen3-14B-FP4

# 2. Check progress
trtllm-download.sh status nvidia/Qwen3-14B-FP4
trtllm-model-status.sh nvidia/Qwen3-14B-FP4   # mostly useful for the "local" side

# 3. Start serving (tp_size=1 if you want)
trtllm-serve.sh nvidia/Qwen3-14B-FP4 8355 1

# 4. Validate and benchmark
trtllm-validate.sh nvidia/Qwen3-14B-FP4 8355
trtllm-bench.sh    nvidia/Qwen3-14B-FP4 8355

# 5. Stop serving
trtllm-stop.sh --kill-only
```

Once this all feels solid, the plan is to throw it into a repo (`spark-tools`) so other DGX Spark folks can just clone, tweak host names / stack names, and get to "serving big models" without quite as much pain.
