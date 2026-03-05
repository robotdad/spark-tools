# spark-tools — Installation Guide

Sets up a two-node DGX Spark cluster for distributed LLM inference using either
TensorRT-LLM (Docker Swarm + MPI) or vLLM (Ray cluster or split independent instances).

## Cluster Topology

| Node  | Hostname | Role                                              |
|-------|----------|---------------------------------------------------|
| monad | `monad`  | Head — Swarm manager / Ray head, API, Proxy, WebUI |
| dyad  | `dyad`   | Worker — Swarm worker / Ray worker                |

---

## Prerequisites

On **both** nodes:

- Ubuntu 22.04+ (or compatible) with NVIDIA drivers installed
- Docker Engine (not Docker Desktop)
- NVIDIA Container Toolkit (`nvidia-docker2` or equivalent)
- Passwordless SSH from **monad → dyad** (for scripts that push config to dyad)
- QSFP cable connecting both nodes (for high-speed inter-node GPU traffic)
- Python 3.10+ with `venv` support (for the auth proxy — optional)
- Hugging Face account with access to your desired model(s)

**Verify prerequisites:**
```bash
# GPU visible to Docker
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi

# SSH from monad to dyad works without password
ssh dyad hostname

# Python available
python3 --version
```

---

## Quick Start (Automated)

`install.sh` auto-detects your network topology, writes config files, generates and
installs systemd service units, and sets kernel tuning parameters.

### On monad (head node)

```bash
cd ~/spark-tools
./install.sh
```

The script auto-detects that you're on the head node from the hostname `monad`. To be
explicit or override:

```bash
./install.sh --node-type head
```

### On dyad (worker node)

```bash
cd ~/spark-tools
./install.sh --node-type worker
```

### install.sh Options

```
./install.sh [OPTIONS]

  --node-type TYPE      Node role: head or worker  (default: auto-detect from hostname)
  --mode MODE           Backend: swarm or ray       (default: swarm)
  --head-ip IP          monad IP on QSFP network   (default: auto-detect)
  --worker-ip IP        dyad IP on QSFP network    (default: auto-detect)
  --qsfp-iface IFACE    QSFP interface name        (default: auto-detect)
  --user USERNAME       Username for services      (default: current user)
  --force               Overwrite existing configs  (default: skip if present)
  --dry-run             Preview changes, no writes
  --help                Show this message
```

### Examples

```bash
# Auto-detect everything (recommended first run)
./install.sh

# Preview what would be installed
./install.sh --dry-run

# Install worker node explicitly
./install.sh --node-type worker

# Head with Ray backend instead of Swarm
./install.sh --node-type head --mode ray

# Override IPs (useful if auto-detect picks wrong interface)
./install.sh --head-ip 192.168.10.1 --worker-ip 192.168.10.2

# Re-install, overwriting existing configs (after IP or hostname change)
./install.sh --force
```

### What install.sh Does

| Step | Action |
|------|--------|
| Auto-detect | Node type, QSFP interface, head/worker IPs |
| `/etc/hosts` | Adds `monad` and `dyad` hostname entries (if absent) |
| `~/.config/spark-tools/cluster.env` | Writes cluster config (mode, engine, IPs, interface) |
| `~/.config/spark-tools/model.env` | Writes model config from `config/model.env.example` (if absent) |
| `/usr/local/bin/spark-*` | Installs symlinks for all `bin/` commands |
| `/etc/systemd/system/spark-*.service` | Generates and installs service units from templates |
| `/etc/sysctl.conf` | Sets `vm.swappiness=1` for GB10 unified memory tuning |
| `/etc/encodings/` | Copies tiktoken vocab files (both nodes) |

---

## Manual Installation

If you prefer full control, follow these steps.

### Step 1: Clone the Repo (Both Nodes)

```bash
git clone https://github.com/robotdad/spark-tools ~/spark-tools
```

### Step 2: Network Configuration

Find your LAN IP and QSFP interface:
```bash
ip route get 8.8.8.8 | grep -oP 'src \K[\d.]+'   # primary LAN IP
ip link | grep -E '^[0-9]+:' | awk '{print $2}'   # list interfaces
ip addr show enp1s0f0np0                           # QSFP interface (common name)
```

Add hostname entries to `/etc/hosts` on **both** nodes:
```bash
sudo tee -a /etc/hosts <<EOF
<MONAD_IP>  monad
<DYAD_IP>   dyad
EOF
```

### Step 3: Install Tokenizer Files (Both Nodes)

```bash
sudo mkdir -p /etc/encodings
sudo cp ~/spark-tools/encodings/*.tiktoken /etc/encodings/
```

Or download fresh:
```bash
sudo curl -fsSL https://openaipublic.blob.core.windows.net/encodings/o200k_base.tiktoken \
  -o /etc/encodings/o200k_base.tiktoken
sudo curl -fsSL https://openaipublic.blob.core.windows.net/encodings/cl100k_base.tiktoken \
  -o /etc/encodings/cl100k_base.tiktoken
```

### Step 4: Create Config Directory

```bash
mkdir -p ~/.config/spark-tools
```

**Cluster config** (`~/.config/spark-tools/cluster.env`):
```bash
cp ~/spark-tools/config/cluster.env.example ~/.config/spark-tools/cluster.env
# Edit to set your actual IPs, mode, engine, QSFP interface
nano ~/.config/spark-tools/cluster.env
```

Key settings to verify:
```bash
SPARK_MODE=swarm            # or ray
SPARK_ENGINE=trtllm         # or vllm
SPARK_PRIMARY_HOST=monad
SPARK_SECONDARY_HOST=dyad
SPARK_QSFP_IFACE=enp1s0f0np0
```

**Model config** (`~/.config/spark-tools/model.env`):
```bash
cp ~/spark-tools/config/model.env.example ~/.config/spark-tools/model.env
nano ~/.config/spark-tools/model.env
```

> **GB10 memory note:** GB10 GPUs use unified memory shared with the system. Start with
> `GPU_MEM_UTIL=0.85` and reduce to `0.70–0.77` if other services are running alongside.
> In Ray cluster mode (TP=2), add `--enforce-eager` to avoid a Triton allocator crash.
> In split mode (TP=1 per node), `--enforce-eager` is **not** needed.

**Per-node config** (split mode only):
```bash
# On monad:
cp ~/spark-tools/config/node.env.monad.example ~/.config/spark-tools/node.env.monad

# On dyad:
cp ~/spark-tools/config/node.env.dyad.example ~/.config/spark-tools/node.env.dyad
```

### Step 5: Pull Docker Images (Both Nodes)

```bash
# TRT-LLM (swarm mode, trtllm engine)
docker pull nvcr.io/nvidia/tensorrt-llm/release:spark-single-gpu-dev

# vLLM standalone / swarm mode
docker pull nvcr.io/nvidia/vllm:25.09-py3

# Ray + vLLM cluster mode
docker pull scitrera/dgx-spark-vllm:0.15.1-t5
```

On **monad only** (Open WebUI):
```bash
docker pull ghcr.io/open-webui/open-webui:main
```

### Step 6: Install CLI Commands

```bash
# Install symlinks into /usr/local/bin
for cmd in ~/spark-tools/bin/spark-*; do
    sudo ln -sf "$cmd" /usr/local/bin/"$(basename "$cmd")"
done
```

Verify:
```bash
spark-status --help
```

### Step 7: Generate and Install Systemd Services

Replace template placeholders and install service units:
```bash
cd ~/spark-tools

# Set your values
USERNAME="$(whoami)"
HOME_DIR="$HOME"
HEAD_IP="$(ip route get 8.8.8.8 | grep -oP 'src \K[\d.]+')"
WORKER_IP="$(getent hosts dyad | awk '{print $1}')"
QSFP_IFACE="enp1s0f0np0"

# Instantiate templates
for tmpl in systemd/*.service.template; do
    svc="/etc/systemd/system/$(basename "${tmpl%.template}")"
    sed \
        -e "s|{{USERNAME}}|${USERNAME}|g" \
        -e "s|{{HOME_DIR}}|${HOME_DIR}|g" \
        -e "s|{{HEAD_IP}}|${HEAD_IP}|g" \
        -e "s|{{WORKER_IP}}|${WORKER_IP}|g" \
        -e "s|{{QSFP_IFACE}}|${QSFP_IFACE}|g" \
        "$tmpl" | sudo tee "$svc" > /dev/null
done

sudo systemctl daemon-reload
```

Or just use the installer (it does all of the above):
```bash
./install.sh --dry-run  # preview
./install.sh            # generate and install
```

### Step 8: Create Open WebUI Container (monad only)

```bash
mkdir -p ~/open-webui-data

docker create --name open-webui \
  -v ~/open-webui-data:/app/backend/data \
  -p 8080:8080 \
  --restart unless-stopped \
  ghcr.io/open-webui/open-webui:main
```

### Step 9: Set Up Auth Proxy (Optional)

```bash
cd ~/spark-tools/proxy
python3 -m venv venv
./venv/bin/pip install -r requirements.txt

# Create secret
mkdir -p ~/.config/vllm-proxy
printf 'VLLM_PROXY_SECRET=%s\n' "$(openssl rand -hex 32)" \
  > ~/.config/vllm-proxy/secret.env

# Install and enable
sudo cp ~/spark-tools/systemd/spark-proxy.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now spark-proxy
```

### Step 10: Enable and Start Services

**On dyad first** (Ray worker must be up before the head starts):
```bash
# Ray mode
sudo systemctl enable spark-ray-worker
sudo systemctl start spark-ray-worker

# Swarm mode (dyad joins as worker automatically via Swarm)
# Nothing to start on dyad for Swarm; Swarm manager schedules workloads from monad
```

**On monad** (Ray mode):
```bash
sudo systemctl enable spark-ray-head spark-ray-vllm spark-open-webui
sudo systemctl start spark-ray-head
sudo systemctl start spark-open-webui
sudo systemctl start spark-ray-vllm      # waits for model to load (several minutes)
```

**On monad** (Swarm mode):
```bash
# First, initialize Swarm if not done:
docker swarm init --advertise-addr <MONAD_LAN_IP>
# Run the join command printed above on dyad

sudo systemctl enable spark-swarm-stack
sudo systemctl start spark-swarm-stack
```

---

## Multi-Node Setup Order

Always configure monad before dyad, and always start the worker before the head in
Ray mode (the head waits for workers to join before vLLM can initialize).

```
1. monad: ./install.sh --node-type head
2. dyad:  ./install.sh --node-type worker
3. dyad:  sudo systemctl start spark-ray-worker   (Ray mode)
4. monad: sudo systemctl start spark-ray-head
5. monad: sudo systemctl start spark-ray-vllm
```

For Swarm mode, just run `spark-serve` on monad after the Swarm is initialized on both
nodes — the Swarm manager schedules everything.

---

## Post-Install Verification

```bash
# Overall status (all services + connectivity)
spark-status

# Confirm vLLM is up
curl -s http://localhost:8000/health

# List loaded models
curl -s http://localhost:8000/v1/models | python3 -m json.tool

# Quick inference test
curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"'$(curl -s http://localhost:8000/v1/models | python3 -c "import sys,json; print(json.load(sys.stdin)['data'][0]['id'])")'" ,
       "messages":[{"role":"user","content":"Say hello in one sentence."}]}' \
  | python3 -m json.tool

# Ray cluster health (Ray mode only)
docker exec node ray status

# Proxy health (if installed)
curl -s http://localhost:9000/health | python3 -m json.tool
```

Expected `spark-status` output:
```
✓ vLLM running        (monad:8000 → HTTP 200)
✓ Ray Head running
✓ Ray Worker running  (dyad)
✓ Proxy running       (monad:9000)
Ray Cluster: 2 nodes active
```

---

## Switching Modes

After installation, switch between cluster topologies with `spark-mode`:

```bash
spark-mode swarm   # Docker Swarm + TRT-LLM (or vLLM)
spark-mode ray     # Ray cluster + vLLM (TP=2)
spark-mode split   # Independent TP=1 per node (best latency)
```

---

## Optional: Tailscale Remote Access

```bash
# Expose WebUI externally (port 443 → 8080)
tailscale funnel 443

# Expose proxy externally (port 8443 → 9000)
tailscale funnel 8443
```

---

## Troubleshooting

See [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md) for common issues.

Quick reference:
```bash
# View logs
journalctl -u spark-ray-vllm -f
journalctl -u spark-swarm-stack -f
journalctl -u spark-proxy -f

# Full reset without losing model cache
spark-reset

# Check Ray cluster
docker exec node ray status

# Regenerate service files after config change
./install.sh --force
sudo systemctl daemon-reload
```
