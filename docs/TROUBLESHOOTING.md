# spark-tools — Troubleshooting Guide

This assumes the standard two-node setup: **monad** (head) and **dyad** (worker).

Quick orientation commands:
```bash
spark-status                        # overall cluster health
journalctl -u spark-ray-vllm -f    # tail vLLM logs (Ray mode)
journalctl -u spark-swarm-stack -f # tail Swarm logs
journalctl -u spark-proxy -f       # tail proxy logs
```

---

## 1. Service Won't Start

### 1a. Ray mode (vLLM)

```bash
journalctl -u spark-ray-head -n 50
journalctl -u spark-ray-vllm -n 50
```

Common causes:

| Symptom in logs                                      | Fix                                                    |
|------------------------------------------------------|--------------------------------------------------------|
| `invalid int value: ''`                              | `TP_SIZE` or `MAX_MODEL_LEN` empty in `model.env`      |
| `HarmonyError: invalid tiktoken vocab file`          | Missing `/etc/encodings/*.tiktoken` — see §5           |
| `No available node types can fulfill resource request`| Ray cluster not fully assembled — check dyad (§3)     |
| `Free memory ... less than desired GPU memory utilization` | Reduce `GPU_MEM_UTIL` in `model.env` or `node.env` |
| `Address already in use`                             | Port 6379 (Ray) or 8000 (vLLM) in use — free it      |

Basic checks:
```bash
cat ~/.config/spark-tools/cluster.env
cat ~/.config/spark-tools/model.env
docker exec node ray status          # are both nodes visible?
docker exec node ls /etc/encodings   # are tiktoken files present?
```

### 1b. Swarm mode (TRT-LLM or vLLM)

```bash
journalctl -u spark-swarm-stack -n 50
docker service ls
docker service logs trtllm-multinode_trtllm --tail 50
```

Common causes:

| Symptom                                              | Fix                                                        |
|------------------------------------------------------|------------------------------------------------------------|
| `service convergence failed`                         | dyad not in Swarm — run `docker swarm join` on dyad        |
| `image not found`                                    | Pull image on both nodes — see §8                          |
| `Conflict. container name already in use`            | `docker rm -f node && sudo systemctl restart spark-swarm-stack` |
| `No such service: trtllm-multinode_trtllm`           | Stack not deployed — `spark-serve`                         |

### 1c. Standalone / split mode

```bash
journalctl -u spark-vllm-standalone -n 50
```

Split mode runs a plain `spark-vllm-standalone.service` on each node independently.
If it fails on dyad, SSH there and check:
```bash
ssh dyad journalctl -u spark-vllm-standalone -n 50
```

### 1d. Proxy service

```bash
journalctl -u spark-proxy -n 50
```

| Symptom                                              | Fix                                                     |
|------------------------------------------------------|---------------------------------------------------------|
| `No secret configured`                               | Create `~/.config/vllm-proxy/secret.env` — see §6      |
| `ModuleNotFoundError: No module named 'fastapi'`     | `cd ~/spark-tools/proxy && python3 -m venv venv && ./venv/bin/pip install -r requirements.txt` |
| `Address already in use` on port 9000                | Another process on 9000 — `sudo lsof -i :9000`         |

---

## 2. CUDA / GPU Issues

### 2a. GPU Memory Out of Memory (OOM)

**Symptom:**
```
ValueError: Free memory on device (77.0/119.7 GiB) is less than desired GPU memory
utilization (0.90, 107.73 GiB)
```

GB10 GPUs use unified memory shared with the system. Available GPU memory depends on
what other processes are running.

**Fix:**

- **Cluster / Ray mode:** Lower `GPU_MEM_UTIL` in `~/.config/spark-tools/model.env`, also add `--enforce-eager`:
  ```bash
  GPU_MEM_UTIL=0.80
  VLLM_EXTRA_ARGS="... --enforce-eager ..."
  sudo systemctl restart spark-ray-vllm
  ```

- **Split mode:** Lower `GPU_MEM_UTIL` in the per-node override file:
  ```bash
  # monad: ~/.config/spark-tools/node.env.monad
  GPU_MEM_UTIL=0.75

  # dyad: ~/.config/spark-tools/node.env.dyad
  GPU_MEM_UTIL=0.85

  sudo systemctl restart spark-vllm-standalone
  ssh dyad sudo systemctl restart spark-vllm-standalone
  ```

Check current GPU memory usage:
```bash
nvidia-smi
```

### 2b. Triton Allocator Crash (TP=2 / cluster mode)

**Symptom:**
```
RuntimeError: Kernel requires a runtime memory allocation, but no allocator was set.
Use triton.set_allocator to specify an allocator.
```

This occurs during CUDA graph capture on GB10 (compute capability 12.1) when running
multi-node tensor parallelism (TP=2).

**Short-term fix:** Add `--enforce-eager` to `VLLM_EXTRA_ARGS`:
```bash
# In ~/.config/spark-tools/model.env
VLLM_EXTRA_ARGS="--enforce-eager --enable-prefix-caching ..."
sudo systemctl restart spark-ray-vllm
```

**Better fix:** Switch to split mode. TP=1 per node means CUDA graphs work correctly,
`--enforce-eager` is unnecessary, and throughput improves significantly:
```bash
spark-mode split
```

### 2c. GPUs Idle During Inference

If `nvidia-smi` shows 0% utilization while requests are in flight:

```bash
# Check NCCL/UCX are bound to the right interface (Ray mode)
docker exec node env | grep -E 'UCX_NET_DEVICES|NCCL_SOCKET_IFNAME|OMPI_MCA'

# Verify QSFP link is up
ip addr show enp1s0f0np0
ping -c3 169.254.x.x    # replace with dyad's link-local address

# Confirm both Ray nodes are present
docker exec node ray status
```

Expected: all network env vars should reference your QSFP interface (e.g., `enp1s0f0np0`).
If not, check `SPARK_QSFP_IFACE` in `cluster.env` and re-run `install.sh`.

---

## 3. Ray Worker Not Showing Up

On monad:
```bash
docker exec node ray status
```

If you see only one node (or zero):

1. Check worker service is running on dyad:
   ```bash
   ssh dyad systemctl status spark-ray-worker
   ssh dyad journalctl -u spark-ray-worker -n 50
   ```

2. Look for these errors in the worker logs:
   - `Connection refused to <HEAD_IP>:6379` → Ray Head is not running on monad, or `HEAD_IP` is wrong
   - `Container name "node" already in use` → stale container; `ssh dyad docker rm -f node`
   - Wrong `HEAD_IP` in `cluster.env` on dyad

3. Verify monad's Ray Head is listening:
   ```bash
   ss -tlnp | grep 6379
   ```

4. Verify dyad can reach monad:
   ```bash
   ssh dyad ping -c3 monad
   ssh dyad nc -zv monad 6379
   ```

---

## 4. Proxy Issues

### 4a. 401 Unauthorized

Clients receive `{"detail": "Unauthorized"}`:
- Confirm the `Authorization: Bearer <token>` header is present and correct
- Check current secret:
  ```bash
  cat ~/.config/vllm-proxy/secret.env
  ```
- Rotate secret:
  ```bash
  ~/spark-tools/scripts/rotate-proxy-secret.sh
  ```
- Restart proxy after rotation:
  ```bash
  sudo systemctl restart spark-proxy
  ```

### 4b. 503 All Backends Down

```bash
curl -s http://localhost:9000/health | python3 -m json.tool
```

Each backend shows `"healthy": true/false`. If both are `false`:
- Is vLLM running on each node?
  ```bash
  curl -s http://monad:8000/health
  curl -s http://dyad:8000/health
  ```
- Does `VLLM_BACKENDS` in the proxy service match the actual hostnames/ports?
  ```bash
  systemctl cat spark-proxy | grep VLLM_BACKENDS
  ```
- Check `/etc/hosts` on monad resolves `dyad`:
  ```bash
  getent hosts dyad
  ```

### 4c. 404 Model Not Found

```json
{"error": "Model 'mymodel' not found on any backend", "available_models": [...]}
```

The requested model isn't loaded on any healthy backend:
- Check what models are actually loaded: `GET /v1/models` via the proxy
- The proxy health endpoint also shows per-backend model lists:
  ```bash
  curl -s http://localhost:9000/health | python3 -m json.tool
  ```
- If backends are healthy but serve different models, use the exact model name shown by `/v1/models`
- `SERVED_MODEL_NAME` in `model.env` can create an alias — use the alias, not the full HF path

---

## 5. Tiktoken / Tokenizer Errors (GPT-OSS Models)

**Symptom:**
```
openai_harmony.HarmonyError: invalid tiktoken vocab file
```

**Fix:** ensure encoding files are in `/etc/encodings/` on **both** nodes and mounted into containers.

```bash
# Verify files exist on monad
ls -lh /etc/encodings/

# Verify files exist on dyad
ssh dyad ls -lh /etc/encodings/

# If missing, copy from repo (on both nodes):
sudo mkdir -p /etc/encodings
sudo cp ~/spark-tools/encodings/*.tiktoken /etc/encodings/

# Or re-download fresh:
sudo curl -fsSL https://openaipublic.blob.core.windows.net/encodings/o200k_base.tiktoken \
  -o /etc/encodings/o200k_base.tiktoken
sudo curl -fsSL https://openaipublic.blob.core.windows.net/encodings/cl100k_base.tiktoken \
  -o /etc/encodings/cl100k_base.tiktoken

# Verify container sees them:
docker exec node ls /etc/encodings
```

---

## 6. SSH Issues Between monad and dyad

spark-tools scripts SSH from monad → dyad without a password. If SSH fails:

```bash
# Test connectivity
ssh dyad hostname

# Common errors and fixes:
# "Connection refused" → sshd not running on dyad: ssh dyad service ssh status
# "Host key verification failed" → add to known_hosts:
ssh-keyscan dyad >> ~/.ssh/known_hosts

# "Permission denied (publickey)" → copy SSH key to dyad:
ssh-copy-id dyad

# Wrong hostname → check /etc/hosts on monad:
getent hosts dyad
```

After any `/etc/hosts` change, re-run `install.sh` to propagate to services.

---

## 7. Model Switching Issues

Switch the active model with:
```bash
spark-set-model nvidia/Qwen3-30B-A3B-FP4
```

If the new model doesn't load:

1. Check `model.env` was updated:
   ```bash
   cat ~/.config/spark-tools/model.env | grep MODEL_NAME
   ```

2. `TP_SIZE` in `model.env` must match what the model requires. For models that fit a
   single GB10, use `TP_SIZE=1` (split mode) or `TP_SIZE=2` (cluster mode across both nodes).

3. `MAX_MODEL_LEN` set too high → OOM. Start conservative and raise:
   ```bash
   # In model.env or node.env.*
   MAX_MODEL_LEN=32768   # safe starting point
   ```

4. Restart services after any model change:
   ```bash
   sudo systemctl restart spark-ray-vllm   # Ray mode
   sudo systemctl restart spark-vllm-standalone  # split mode (monad)
   ssh dyad sudo systemctl restart spark-vllm-standalone  # split mode (dyad)
   ```

---

## 8. Docker Issues

### Image Not Found

```bash
# Check what's available locally
docker images | grep -E 'vllm|trtllm'

# Pull required images (on BOTH nodes)
docker pull nvcr.io/nvidia/vllm:25.09-py3
docker pull nvcr.io/nvidia/tensorrt-llm/release:spark-single-gpu-dev
docker pull scitrera/dgx-spark-vllm:0.15.1-t5   # Ray+vLLM image

# Pull WebUI (monad only)
docker pull ghcr.io/open-webui/open-webui:main
```

### Stale Container Conflicts

```bash
# Error: "Container name '/node' already in use"
docker rm -f node
sudo systemctl restart spark-ray-head   # or spark-swarm-stack
```

Service unit files include `ExecStartPre=-/usr/bin/docker rm -f node` to handle this
automatically, but manual cleanup may be needed after a crash.

### Swarm Not Initialized

```bash
# On monad:
docker info | grep -i swarm

# If "Swarm: inactive", initialize:
docker swarm init --advertise-addr <MONAD_IP>

# Then on dyad, join using the token printed above:
docker swarm join --token <TOKEN> <MONAD_IP>:2377

# Verify both nodes visible:
docker node ls
```

---

## 9. Network / QSFP Issues

The QSFP cable provides 200 Gb/s for GPU-to-GPU traffic. Problems here cause slow
inference or timeout errors in Ray/MPI.

```bash
# Check interface is up (both nodes)
ip link show enp1s0f0np0

# Check link-local address is assigned
ip addr show enp1s0f0np0 | grep '169.254'

# Ping across QSFP link
ping -I enp1s0f0np0 169.254.<dyad-last-octet>.1

# If interface is DOWN, bring it up
sudo ip link set enp1s0f0np0 up

# Check interface name (may differ from default)
ip link | grep -E '^[0-9]+:' | awk '{print $2}' | tr -d ':'
```

If the QSFP interface name differs from `enp1s0f0np0`, update `cluster.env`:
```bash
SPARK_QSFP_IFACE=<actual-interface-name>
```
Then re-run `install.sh` to regenerate service files.

**After a router change / LAN IP change:**

Option 1 — re-run installer:
```bash
cd ~/spark-tools
./install.sh --head-ip <NEW_HEAD_IP> --worker-ip <NEW_WORKER_IP>
```

Option 2 — manual update:
1. Update `/etc/hosts` on both nodes
2. Update `cluster.env` with new IPs
3. `sudo systemctl daemon-reload && sudo systemctl restart spark-ray-head spark-ray-vllm`

QSFP link-local addresses (`169.254.x.x`) don't change when the router changes.

---

## 10. Performance Issues

### Symptoms: High latency, low throughput

Diagnostic steps:
```bash
# Are CUDA graphs enabled? (should be in split mode)
journalctl -u spark-vllm-standalone | grep -i 'cuda graph\|eager'

# Is FlashInfer active?
journalctl -u spark-vllm-standalone | grep -i 'flashinfer\|attention backend'

# Is FP8 KV cache active?
journalctl -u spark-vllm-standalone | grep -i 'kv_cache\|fp8'
```

### Key performance knobs

| Flag                           | Where                  | Impact                                      |
|--------------------------------|------------------------|---------------------------------------------|
| `--enforce-eager`              | `VLLM_EXTRA_ARGS`      | **Remove in split mode** — disables CUDA graphs, hurts throughput |
| `--attention-backend flashinfer` | `VLLM_EXTRA_ARGS`    | Use FlashInfer optimized attention (GB10)   |
| `--kv-cache-dtype fp8`         | `VLLM_EXTRA_ARGS`      | Cuts KV memory ~40%, enables longer context |
| `--enable-prefix-caching`      | `VLLM_EXTRA_ARGS`      | KV cache reuse for repeated prefixes        |
| `GPU_MEM_UTIL`                 | `node.env.*`           | More = more KV cache headroom               |
| `vm.swappiness=1`              | `/etc/sysctl.conf`     | Prevent OS from swapping GPU workers        |
| `FLASHINFER_MOE_BACKEND=latency` | container env        | Latency-optimized MoE dispatch (MoE models)|

`install.sh` sets `vm.swappiness=1` automatically. To apply without rebooting:
```bash
sudo sysctl vm.swappiness=1
```

### Periodic drop_caches

For sustained load, OS page cache can crowd out GPU unified memory. Schedule:
```bash
# Add to crontab on both nodes:
*/15 * * * * sync && echo 3 | sudo tee /proc/sys/vm/drop_caches > /dev/null
```

---

## 11. Full Cluster Reset (without losing models)

```bash
# Quick reset via helper (if installed)
spark-reset

# Manual reset:
sudo systemctl stop spark-ray-vllm spark-ray-head spark-ray-worker spark-swarm-stack
docker rm -f node || true
ssh dyad docker rm -f node || true

# Restart in order: worker first, then head, then vLLM
ssh dyad sudo systemctl start spark-ray-worker    # Ray mode
sudo systemctl start spark-ray-head
sudo systemctl start spark-ray-vllm              # takes several minutes to load model
```

Model cache is in `~/.cache/huggingface/` — it won't be re-downloaded.

---

## 12. Service Files Out of Sync

If you've changed IPs, usernames, or interface names and need to regenerate services:

```bash
cd ~/spark-tools
./install.sh --dry-run   # preview changes
./install.sh --force     # regenerate and reinstall

sudo systemctl daemon-reload
sudo systemctl restart spark-ray-head spark-ray-vllm spark-proxy
```

---

## 13. vLLM Health Endpoint

Quick API sanity check:
```bash
# Direct vLLM (no proxy)
curl -s -o /dev/null -w "%{http_code}\n" http://localhost:8000/health

# Via proxy (with auth)
curl -s -H "Authorization: Bearer $(cat ~/.config/vllm-proxy/secret.env | cut -d= -f2)" \
     http://localhost:9000/health | python3 -m json.tool

# List models
curl -s http://localhost:8000/v1/models | python3 -m json.tool
```

- `200` → healthy
- `000` → vLLM not running (connection refused)
- `503` → starting up or all proxy backends down
