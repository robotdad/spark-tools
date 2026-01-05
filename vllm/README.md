# vLLM for DGX Spark

Scripts for running vLLM inference on DGX Spark with focus on **multimodal (vision) models** for image analysis.

**What these scripts solve:**
- Simple Docker-based vLLM deployment (no complex builds)
- Vision-language model serving via OpenAI-compatible API
- Easy health checks and validation
- Quick image analysis testing

---

## Quick Start

**Single-Node Workflow (DGX Spark with vision model)**

```bash
# 1. Setup (once)
export PATH="$HOME/spark-tools/vllm:$PATH"

# 2. Pull vLLM container (one time, ~24GB)
docker pull nvcr.io/nvidia/vllm:25.09-py3

# 3. Start server (downloads model on first run)
vllm-serve.sh

# 4. Wait for startup, then validate
docker logs -f vllm-server  # Wait for "Application startup complete"
vllm-validate.sh

# 5. Test with an image
vllm-test.py /path/to/image.jpg

# 6. Stop when done
vllm-stop.sh
```

---

## Prerequisites

### Hardware Requirements

- **DGX Spark** (ARM64 + Blackwell GB10 GPU)
- 128 GB unified memory
- Supports models up to ~200B parameters (text) or ~30B (vision)

### Software Prerequisites

1. **Docker** with NVIDIA Container Toolkit
   ```bash
   # Verify
   docker ps
   docker run --rm --gpus all nvidia/cuda:12.0.0-base-ubuntu22.04 nvidia-smi
   ```

2. **NVIDIA drivers**
   ```bash
   nvidia-smi  # Should show GB10 GPU
   ```

3. **vLLM Docker image**
   ```bash
   docker pull nvcr.io/nvidia/vllm:25.09-py3
   ```

### Python Prerequisites (for vllm-test.py)

```bash
pip install openai
```

### Directory Structure

These scripts expect to be installed at:
```
~/spark-tools/vllm/
```

Add to PATH (recommended):
```bash
export PATH="$HOME/spark-tools/vllm:$PATH"
```

---

## Why vLLM vs TensorRT-LLM?

**vLLM is better for inference-focused workloads:**

| Feature | vLLM | TensorRT-LLM |
|---------|------|--------------|
| Setup | Load HF model directly | Build engine first |
| Iteration | Instant model switching | Recompile per change |
| Throughput | Superior batching | Better single-request latency |
| API | Built-in OpenAI compatibility | Custom serving layer |
| Multimodal | Native vision support | Limited multimodal |

**Use vLLM when:**
- Running inference at scale
- Need OpenAI-compatible API
- Want to experiment with different models quickly
- Working with multimodal (vision) models

**Use TensorRT-LLM when:**
- Absolute minimum latency is critical
- Production deployment with fixed model
- Need advanced TensorRT optimizations

---

## Script Reference

### `vllm-serve.sh`

Start vLLM server with a model.

```bash
vllm-serve.sh [MODEL] [PORT] [GPU_MEM_UTIL] [MAX_MODEL_LEN]
```

**Parameters:**
- `MODEL` - HuggingFace model ID (default: `Qwen/Qwen2-VL-7B-Instruct`)
- `PORT` - HTTP port (default: `8000`)
- `GPU_MEM_UTIL` - GPU memory utilization (default: `0.9`)
- `MAX_MODEL_LEN` - Max context length (default: `4096`)

**Examples:**
```bash
# Default vision model (Qwen2-VL-7B)
vllm-serve.sh

# Custom model and port
vllm-serve.sh "microsoft/Phi-4-multimodal-instruct" 8001

# Adjust GPU memory for larger models
vllm-serve.sh "Qwen/Qwen2-VL-7B-Instruct" 8000 0.95
```

**Features:**
- Runs in Docker with GPU access
- Auto-downloads model on first run (~14GB for Qwen2-VL-7B)
- Persistent HuggingFace cache across restarts
- OpenAI-compatible API endpoint

**Note:** First startup takes 5-10 minutes while model downloads. Monitor with:
```bash
docker logs -f vllm-server
```

---

### `vllm-stop.sh`

Stop the vLLM server.

```bash
vllm-stop.sh
```

Stops and removes the Docker container.

---

### `vllm-validate.sh`

Health check and validation.

```bash
vllm-validate.sh [MODEL] [PORT]
```

**Checks:**
1. Docker container running
2. GPU status
3. Health endpoint responding
4. Models endpoint working
5. Basic inference test

**Example:**
```bash
vllm-validate.sh
vllm-validate.sh "Qwen/Qwen2-VL-7B-Instruct" 8000
```

---

### `vllm-test.py`

Test image analysis via command line.

```bash
vllm-test.py <image_path> [prompt] [server_url]
```

**Examples:**
```bash
# Basic analysis
vllm-test.py photo.jpg

# Custom prompt
vllm-test.py document.png "Extract all text from this image"

# Remote server
vllm-test.py chart.jpg "Analyze this chart" http://192.168.1.100:8000/v1
```

**Output:**
- Sends image (base64 encoded) to vLLM server
- Prints analysis result
- Shows token usage stats

---

## Supported Models

### Vision-Language Models (Recommended for Image Analysis)

**Qwen2-VL Series** (Best supported in vLLM 0.10.2):
- `Qwen/Qwen2-VL-2B-Instruct` (2B params, ~4GB memory)
- `Qwen/Qwen2-VL-7B-Instruct` (7B params, ~14GB memory) ← **Default**
- Strong OCR, document analysis, multi-image support

**LLaVA Series**:
- `llava-hf/llava-1.5-7b-hf` (7B params)
- `llava-hf/llava-v1.6-mistral-7b-hf` (7B params)

**Phi-4 Multimodal**:
- `microsoft/Phi-4-multimodal-instruct` (14B params)
- Note: Known performance issues with vLLM 0.10.2

### Text-Only Models

vLLM also supports any HuggingFace text model:
- Llama-3.x series
- Mistral series
- Qwen3 (text-only)

---

## Model Capabilities

### Qwen2-VL-7B-Instruct (Default Model)

**Excels at:**
- General image understanding and description
- **OCR** (text extraction from images/documents)
- Document analysis
- Chart and diagram interpretation
- Multi-image comparison
- Visual question answering

**Example use cases:**
```bash
# OCR a document
vllm-test.py scan.png "Extract all text from this document"

# Analyze a chart
vllm-test.py chart.jpg "Analyze this chart and explain the trends"

# Describe an image
vllm-test.py photo.jpg "What's in this image? Be detailed."

# Compare images (multi-turn conversation)
# (Requires API calls with conversation context)
```

---

## Usage Examples

### Basic Server Startup

```bash
# Start with defaults
vllm-serve.sh

# Monitor logs
docker logs -f vllm-server

# In another terminal, validate
vllm-validate.sh
```

### Python API Usage

```python
import base64
from openai import OpenAI

# Connect to vLLM
client = OpenAI(
    api_key="EMPTY",
    base_url="http://localhost:8000/v1"
)

# Analyze an image
with open("image.jpg", "rb") as f:
    image_data = base64.b64encode(f.read()).decode("utf-8")

response = client.chat.completions.create(
    model="Qwen/Qwen2-VL-7B-Instruct",
    messages=[{
        "role": "user",
        "content": [
            {"type": "text", "text": "What's in this image?"},
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}
            }
        ]
    }]
)

print(response.choices[0].message.content)
```

### Remote Access

Get your DGX Spark IP:
```bash
hostname -I
```

From any machine on the network:
```python
client = OpenAI(
    api_key="EMPTY",
    base_url="http://192.168.1.100:8000/v1"  # Your DGX IP
)
```

Or with curl:
```bash
curl -X POST http://192.168.1.100:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2-VL-7B-Instruct",
    "messages": [{
      "role": "user",
      "content": [
        {"type": "text", "text": "Describe this image"},
        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,BASE64_HERE"}}
      ]
    }]
  }'
```

---

## Configuration

### Changing Models

Edit `vllm-serve.sh` or pass as argument:

```bash
# Use a different vision model
vllm-serve.sh "llava-hf/llava-1.5-7b-hf"

# Use a text-only model
vllm-serve.sh "meta-llama/Llama-3.1-8B-Instruct"
```

### Adjusting Memory

If you hit OOM errors or want to reserve memory for other processes:

```bash
# Use 80% of GPU memory instead of 90%
vllm-serve.sh "Qwen/Qwen2-VL-7B-Instruct" 8000 0.8
```

### Changing Port

```bash
# Run on port 8001 instead of 8000
vllm-serve.sh "Qwen/Qwen2-VL-7B-Instruct" 8001
```

### HuggingFace Cache Location

Models are cached in `~/.cache/huggingface/` by default. This is mounted into the container, so downloads persist across container restarts.

To use a different cache location, edit the `-v` mount in `vllm-serve.sh`:
```bash
-v /your/custom/path:/root/.cache/huggingface \
```

---

## Troubleshooting

### "Docker image not found"

**Fix:**
```bash
docker pull nvcr.io/nvidia/vllm:25.09-py3
```

### "Application startup complete" taking too long

**First run:** Model is downloading (~14GB for Qwen2-VL-7B). Check progress:
```bash
docker logs -f vllm-server
```

Look for download progress lines.

### "Connection refused" when testing

Server may still be starting. Wait for:
```
INFO:     Application startup complete.
```

in the logs before testing.

### "Out of memory" error

**Solutions:**
1. Reduce GPU memory utilization:
   ```bash
   vllm-serve.sh "Qwen/Qwen2-VL-7B-Instruct" 8000 0.8
   ```

2. Reduce max context length:
   ```bash
   vllm-serve.sh "Qwen/Qwen2-VL-7B-Instruct" 8000 0.9 2048
   ```

3. Use a smaller model:
   ```bash
   vllm-serve.sh "Qwen/Qwen2-VL-2B-Instruct"
   ```

### Image analysis returns gibberish

Ensure you're using a **vision-language model**, not a text-only model:
- ✅ `Qwen/Qwen2-VL-7B-Instruct` (vision)
- ✅ `llava-hf/llava-1.5-7b-hf` (vision)
- ❌ `Qwen/Qwen3-14B-FP4` (text-only, won't understand images)

### Port already in use

Stop existing server or use different port:
```bash
vllm-stop.sh
# Or
vllm-serve.sh "Qwen/Qwen2-VL-7B-Instruct" 8001
```

---

## Performance Notes

**First request latency:**
- ~30 seconds (model loading into GPU memory)

**Subsequent requests:**
- <2 seconds for typical images (512x512 to 1024x1024)

**Memory usage:**
- Qwen2-VL-7B: ~14GB for model + additional for KV cache
- Larger images = more tokens = more memory

**Throughput:**
- vLLM's continuous batching handles multiple requests efficiently
- Can serve multiple clients simultaneously

---

## Integration with Amplifier

Configure your Amplifier app to use the vLLM server:

```yaml
# In your Amplifier app's settings.yaml
providers:
  default:
    type: vllm
    config:
      base_url: "http://localhost:8000/v1"  # or remote IP
      model: "Qwen/Qwen2-VL-7B-Instruct"
```

The Amplifier vLLM module uses the OpenAI-compatible API under the hood.

---

## Model Size Guidelines

**DGX Spark (128GB unified memory):**

| Model Type | Max Size | Examples |
|------------|----------|----------|
| Text-only | ~200B params | Qwen3-235B, Llama-3.3-70B |
| Vision-language | ~30B params | Qwen2-VL-7B, LLaVA-13B, Phi-4-multimodal |

Vision models require more memory due to image processing overhead.

---

## Additional Resources

- [vLLM Documentation](https://docs.vllm.ai/)
- [DGX Spark vLLM Guide](https://build.nvidia.com/spark/vllm)
- [Qwen2-VL Model Card](https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct)
- [OpenAI Vision API Format](https://platform.openai.com/docs/guides/vision)

---

## Differences from TensorRT-LLM Scripts

| Feature | vLLM | TensorRT-LLM |
|---------|------|--------------|
| Model loading | Direct from HF | Build engine first |
| Multi-node | Single node only | Supports tensor parallelism |
| Setup complexity | Simple Docker run | Requires Swarm setup |
| Model switching | Restart container | Rebuild engine |
| Multimodal | Native support | Limited |

For multi-node large model inference, use the `trtllm/` scripts instead.
