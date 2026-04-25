# vLLM Provider Model Metadata

**Status:** Plan — not started
**Owner:** TBD
**Last updated:** 2026-04-25

## The problem

The `amplifier-module-provider-vllm` provider exposes `list_models()`, which queries
`GET /v1/models` from a vLLM endpoint and returns a `ModelInfo` per discovered model.
Trouble is, vLLM's `/v1/models` only returns the model ID and creation timestamp —
none of the metadata Amplifier's routing matrix actually needs. So today the provider
fills the gaps with hardcoded constants:

```python
# amplifier_module_provider_vllm/__init__.py, list_models()
ModelInfo(
    id=model.id,
    display_name=model.id,
    context_window=128000,                                  # hardcoded
    max_output_tokens=32768,                                # hardcoded
    capabilities=["tools", "streaming", "reasoning", "local"],  # same for all
    defaults={"temperature": None, "max_tokens": 16384},
)
```

Every model gets identical metadata. That's wrong, and it gets more wrong as you
serve heterogeneous models behind a single proxy:

| Model | Real ctx (deployed) | Vision | Reasoning | Coding |
|---|---|---|---|---|
| `google/gemma-4-26B-A4B-it` | 131K (monad) | yes | limited | good |
| `Qwen/Qwen3.6-35B-A3B-FP8` | 262K (dyad) | no | strong | best-in-class |

When Amplifier's routing matrix asks "which model can handle a 200K-token vision
task?" the provider currently says "all of them, identically" and the answer is
wrong.

## The fix, in one sentence

Let users supply a `model_overrides` map in the provider config, merge it into the
ModelInfo objects returned by `list_models()`, and call it done. Everything else
in this document is downstream convenience.

---

## Phase 1 — Provider config: `model_overrides`

**Scope:** A single PR against `amplifier-module-provider-vllm`. Implementable in
one session. This is the only phase that has to land for this to be useful.

### Config shape

```yaml
providers:
  - module: provider-vllm
    source: git+https://github.com/microsoft/amplifier-module-provider-vllm@main
    config:
      base_url: "http://monad:9000/v1"
      api_key: "..."
      default_model: "google/gemma-4-26B-A4B-it"
      model_overrides:
        "google/gemma-4-26B-A4B-it":
          display_name: "Gemma 4 26B (vision)"
          context_window: 131072
          max_output_tokens: 32768
          capabilities: ["tools", "streaming", "vision", "local"]
        "Qwen/Qwen3.6-35B-A3B-FP8":
          display_name: "Qwen 3.6 35B MoE (FP8)"
          context_window: 262144
          max_output_tokens: 32768
          capabilities: ["tools", "streaming", "reasoning", "coding", "local"]
```

### Contract

**Input:** Provider config with optional `model_overrides: dict[str, ModelOverride]`.

**ModelOverride fields (all optional):**
- `display_name: str`
- `context_window: int`
- `max_output_tokens: int`
- `capabilities: list[str]` — replaces the default list, doesn't merge
- `defaults: dict[str, Any]` — merged into existing defaults (override-wins)

**Behavior in `list_models()`:**
1. Fetch model list from `/v1/models` as today.
2. For each `model.id`, look up `model_overrides[model.id]`.
3. If found, use the override values; for fields not in the override, fall back to
   today's hardcoded defaults.
4. If not found, return today's hardcoded ModelInfo unchanged. (No surprise
   regressions for existing configs.)

**Unknown models in `model_overrides` (override exists but model not served):**
silently ignored. Don't fail config validation on this — proxy contents change.

### Implementation sketch

```python
# config schema (pydantic)
class ModelOverride(BaseModel):
    display_name: str | None = None
    context_window: int | None = None
    max_output_tokens: int | None = None
    capabilities: list[str] | None = None
    defaults: dict[str, Any] | None = None

class ProviderConfig(BaseModel):
    base_url: str
    api_key: str | None = None
    default_model: str
    model_overrides: dict[str, ModelOverride] = {}

# in list_models():
async def list_models(self) -> list[ModelInfo]:
    models_response = await self.client.models.list()
    out = []
    for model in models_response.data:
        override = self.config.model_overrides.get(model.id)
        out.append(self._build_model_info(model.id, override))
    return out

def _build_model_info(self, model_id: str, override: ModelOverride | None) -> ModelInfo:
    o = override or ModelOverride()
    base_defaults = {"temperature": None, "max_tokens": 16384}
    return ModelInfo(
        id=model_id,
        display_name=o.display_name or model_id,
        context_window=o.context_window or 128000,
        max_output_tokens=o.max_output_tokens or 32768,
        capabilities=o.capabilities or ["tools", "streaming", "reasoning", "local"],
        defaults={**base_defaults, **(o.defaults or {})},
    )
```

### Tests

1. No `model_overrides` in config → behavior identical to today.
2. Partial override (only `context_window`) → other fields fall through to defaults.
3. Override for model not in `/v1/models` response → silently ignored (no error).
4. `/v1/models` returns model not in `model_overrides` → uses defaults (no error).
5. `capabilities` override fully replaces — does not merge with defaults.

### Why this is the whole job for phase 1

- Smallest possible diff to the provider.
- No new deps.
- No schema changes to `ModelInfo`.
- Backwards compatible — default behavior unchanged.
- Downstream (Amplifier routing matrix) immediately sees richer metadata.

---

## Phase 2 — Profile metadata in spark-tools

**Scope:** Extend `spark-tools/profiles/*.env` files with metadata fields. No code
changes outside profiles unless phase 3 follows.

### Fields to add

Following the existing `PROFILE_*` naming convention (already used for
`PROFILE_DEFAULT_MAX_MODEL_LEN` etc.):

```bash
# In Qwen--Qwen3.6-35B-A3B-FP8.env
PROFILE_DISPLAY_NAME="Qwen 3.6 35B MoE (FP8)"
PROFILE_CONTEXT_WINDOW=262144
PROFILE_MAX_OUTPUT_TOKENS=32768
PROFILE_CAPABILITIES="tools,streaming,reasoning,coding,local"
PROFILE_MODEL_FAMILY="qwen3"
PROFILE_QUANTIZATION="fp8"
PROFILE_TOOL_CALL_PARSER="qwen3_coder"
```

### Field semantics

**Required for phase 3 generation:**
- `PROFILE_DISPLAY_NAME` — human-readable
- `PROFILE_CONTEXT_WINDOW` — integer; the *deployed* max, may be lower than the
  model's theoretical max (e.g. 131K on monad even though Gemma 4 supports 256K)
- `PROFILE_MAX_OUTPUT_TOKENS` — integer
- `PROFILE_CAPABILITIES` — comma-separated list, no spaces. Standard tags:
  `tools`, `streaming`, `reasoning`, `vision`, `coding`, `local`. Routing matrix
  will only match on tags it knows about; novel tags are harmless.

**Informational (not consumed by provider):**
- `PROFILE_MODEL_FAMILY` — for grouping in UIs and routing fallbacks
- `PROFILE_QUANTIZATION` — `fp4` | `fp8` | `bf16` | `int8` etc.; affects quality
  expectations
- `PROFILE_TOOL_CALL_PARSER` — already implicit in `VLLM_EXTRA_ARGS`, but pulling
  it into a named field makes it discoverable

### Where the deployed-vs-theoretical context window comes from

`PROFILE_CONTEXT_WINDOW` should reflect what the model is actually serving with,
which is `PROFILE_DEFAULT_MAX_MODEL_LEN` overridden by the node's `--max-model-len`
flag if present. Phase 3 generation needs to read both and pick the effective value.

For phase 2 alone, just hardcode `PROFILE_CONTEXT_WINDOW` to match the typical
deployment. Phase 3 makes it dynamic.

### Migration

Update all four existing profiles in one PR:
- `google--gemma-4-26B-A4B-it.env`
- `Qwen--Qwen3.6-35B-A3B-FP8.env`
- `nvidia--Qwen3-14B-FP4.env`
- `nvidia--Qwen3-235B-A22B-FP4.env`

No version bump needed — `PROFILE_*` keys are additive and shell sourcing ignores
unknown vars.

---

## Phase 3 — Auto-generate provider config from spark-tools

**Scope:** New spark-tools subcommand (or new top-level command) that emits ready-
to-paste YAML for the `amplifier-module-provider-vllm` config block.

### Command

```
spark-mode provider-config            # writes to stdout
spark-mode provider-config --proxy    # use proxy endpoint (default if active)
spark-mode provider-config --direct   # emit one provider per node
```

### Output (proxy mode, default when proxy is active)

```yaml
providers:
  - module: provider-vllm
    source: git+https://github.com/microsoft/amplifier-module-provider-vllm@main
    config:
      base_url: "http://monad:9000/v1"
      api_key: "<from spark-proxy.env>"
      default_model: "google/gemma-4-26B-A4B-it"
      model_overrides:
        "google/gemma-4-26B-A4B-it":
          display_name: "Gemma 4 26B"
          context_window: 131072
          max_output_tokens: 32768
          capabilities: ["tools", "streaming", "vision", "local"]
        "Qwen/Qwen3.6-35B-A3B-FP8":
          display_name: "Qwen 3.6 35B MoE (FP8)"
          context_window: 262144
          max_output_tokens: 32768
          capabilities: ["tools", "streaming", "reasoning", "coding", "local"]
```

### Implementation outline

1. Determine which nodes are active and which profiles they're running. Reuse the
   logic that `_do_status` already has after the recent enhancement —
   `_query_models()` knows what's actually being served.
2. For each `(node, model_id)`, locate the corresponding profile file and source
   it to extract `PROFILE_*` fields.
3. Compute effective context window:
   `effective_ctx = min(node.max_model_len or PROFILE_DEFAULT_MAX_MODEL_LEN, PROFILE_CONTEXT_WINDOW)`
4. Emit YAML. Use `python -c 'import yaml; ...'` if a Python is on the box;
   otherwise hand-format (the structure is small and stable).
5. `--direct` mode: emit one provider entry per node, each with `default_model`
   set and `model_overrides` containing only that one model.

### Token handling

Don't print the bearer token to stdout by default. Two options:
- `--reveal-token` flag — explicit opt-in.
- Default emits `api_key: "${SPARK_PROXY_TOKEN}"` and prints a hint about exporting
  the env var. Pick this default; it's safer.

---

## Phase 4 — Routing matrix integration

**Scope:** Documentation and worked examples. No code in this phase — just a
recipe for users.

### Example routing matrix

Once phase 1 ships and `ModelInfo.capabilities` is meaningful per model, users can
write Amplifier routing matrices like:

```yaml
routing:
  vision:
    require_capabilities: ["vision"]
    # only Gemma 4 has vision → routes there automatically

  coding:
    require_capabilities: ["coding", "tools"]
    # Qwen 3.6 has both, Gemma 4 lacks "coding" tag → routes to Qwen

  reasoning:
    require_capabilities: ["reasoning"]
    prefer_context_window: ">100000"
    # Qwen 3.6 wins on both axes

  fast:
    prefer_model: "google/gemma-4-26B-A4B-it"
    # explicit pick when capabilities don't disambiguate

  general:
    fallback: "google/gemma-4-26B-A4B-it"
```

### Things to document

- Standard capability tag vocabulary (above).
- How `require_capabilities` interacts with multi-provider configs (proxy vs
  direct).
- The fact that `default_model` in the provider config is the floor — routing
  matrix overrides on a per-call basis via the `model` field, which the proxy
  honors.

---

## Out of scope (explicitly)

These came up while scoping but are deliberately deferred:

- **Custom metadata endpoint on the proxy.** Would let the proxy serve metadata
  alongside `/v1/models`. Cleaner architecturally, but requires changes to
  spark-proxy and a non-OpenAI-standard endpoint. Defer until phase 3 proves
  the static-config approach is annoying enough.
- **vLLM upstream PR for richer `/v1/models`.** Right answer long-term, wrong
  timeline. Track separately.
- **ModelInfo schema changes.** Adding `quantization`, `model_family`, etc. to
  `ModelInfo` itself would be cleaner, but requires coordinated changes across
  Amplifier core. Phase 1 deliberately avoids this — overrides only fill
  fields that already exist in `ModelInfo`.
- **Per-deployment quality benchmarks.** "Qwen 3.6 is best at coding" should
  eventually be backed by benchmark numbers, not vibes. Not this plan.

---

## Order of operations

If you only have time for one phase, do **Phase 1**. It's self-contained, lives
entirely in the provider repo, and immediately unblocks the routing matrix.

If you have time for two, do **Phase 1 + Phase 2**. Phase 2 by itself doesn't do
anything — but it captures the metadata in version control next to the operational
config, which is the right home for it. Even if phase 3 never ships, a human can
copy the values into a provider config by hand.

Phase 3 is convenience automation. Phase 4 is documentation. Both are nice but
neither is on the critical path.

## Success criteria for phase 1

- `model_overrides` config key is documented in the provider README.
- `list_models()` returns distinct `ModelInfo` per model when overrides are set.
- Existing configs without `model_overrides` produce identical output to today.
- Tests cover the five cases listed in Phase 1 § Tests.
- Versioned release on the provider repo so spark-tools / Amplifier configs can
  pin to it.
