"""Tests for spark-tools per-node and model configuration files.

Adapted from Brian's dgx-spark-cluster test_config_files.py.

Brian's setup                          spark-tools equivalent
─────────────────────────────────────  ───────────────────────────────────────
~/spark-cluster/config/                config/ (repo-relative)
node.env.spark-1 (moderate, 0.80)     node.env.monad.example (moderate, 0.80)
node.env.spark-2 (aggressive, 0.90)   node.env.dyad.example  (aggressive, 0.90)
GPU_MEMORY_UTILIZATION key             GPU_MEM_UTIL key
MAX_NUM_SEQS top-level key             max-num-seqs inside VLLM_EXTRA_ARGS
TP_SIZE=2 in model.env (shared)       TP_SIZE=2 in model.env.example (shared)
No SERVED_MODEL_NAME field             SERVED_MODEL_NAME= (separate field)
No TRT-LLM keys                        TRTLLM_* keys present in model.env

Key invariants tested:
- monad node  : GPU_MEM_UTIL=0.80, MAX_MODEL_LEN=131072, TP_SIZE=1
- dyad node   : GPU_MEM_UTIL=0.90, MAX_MODEL_LEN=262144, TP_SIZE=1
- model shared: FP8 KV cache, tool calling, prefix caching in VLLM_EXTRA_ARGS
- enforce-eager present in model.env (multi-node TP=2), absent in per-node files
"""

import os

import pytest

from helpers import REPO_ROOT, parse_env_file

# ---------------------------------------------------------------------------
# Config paths
# ---------------------------------------------------------------------------

CONFIG_DIR = os.path.join(REPO_ROOT, "config")

MODEL_ENV = os.path.join(CONFIG_DIR, "model.env.example")
MONAD_ENV = os.path.join(CONFIG_DIR, "node.env.monad.example")
DYAD_ENV = os.path.join(CONFIG_DIR, "node.env.dyad.example")

ALL_CONFIG_FILES = [MODEL_ENV, MONAD_ENV, DYAD_ENV]


# ---------------------------------------------------------------------------
# 1. All three config files exist
# ---------------------------------------------------------------------------


class TestAllFilesExist:
    @pytest.mark.parametrize(
        "path,label",
        [
            (MODEL_ENV, "model.env.example"),
            (MONAD_ENV, "node.env.monad.example"),
            (DYAD_ENV, "node.env.dyad.example"),
        ],
    )
    def test_file_exists(self, path, label):
        assert os.path.isfile(path), f"{label} must exist at {path}"


# ---------------------------------------------------------------------------
# 2. node.env.monad.example — moderate profile (primary node)
#    Mirrors Brian's node.env.spark-1: conservative utilization, 128K context
# ---------------------------------------------------------------------------


class TestNodeEnvMonad:
    @pytest.fixture
    def config(self):
        assert os.path.isfile(MONAD_ENV), f"Must exist: {MONAD_ENV}"
        return parse_env_file(MONAD_ENV)

    @pytest.fixture
    def raw(self):
        with open(MONAD_ENV) as f:
            return f.read()

    # Parallelism

    def test_tp_size_is_1(self, config):
        """Split-node mode: each GB10 serves independently at TP=1."""
        assert config.get("TP_SIZE") == "1", "monad must have TP_SIZE=1"

    # Memory

    def test_gpu_mem_util_conservative(self, config):
        """Moderate utilization leaves headroom for OS and side processes."""
        assert config.get("GPU_MEM_UTIL") == "0.80"

    def test_max_model_len(self, config):
        """131072 tokens is safe ceiling for 0.80 util + FP8 KV on GB10."""
        assert config.get("MAX_MODEL_LEN") == "131072"

    # VLLM_EXTRA_ARGS: what should be present

    def test_extra_args_has_prefix_caching(self, config):
        assert "--enable-prefix-caching" in config["VLLM_EXTRA_ARGS"]

    def test_extra_args_has_block_size_128(self, config):
        assert "--block-size 128" in config["VLLM_EXTRA_ARGS"]

    def test_extra_args_has_flashinfer_backend(self, config):
        assert "--attention-backend flashinfer" in config["VLLM_EXTRA_ARGS"]

    def test_extra_args_has_fp8_kv_cache(self, config):
        """FP8 KV cache saves ~40% memory; must be present in split-node mode too."""
        assert "--kv-cache-dtype fp8" in config["VLLM_EXTRA_ARGS"]

    def test_extra_args_has_tool_calling(self, config):
        assert "--enable-auto-tool-choice" in config["VLLM_EXTRA_ARGS"]

    def test_extra_args_has_trust_remote_code(self, config):
        assert "--trust-remote-code" in config["VLLM_EXTRA_ARGS"]

    def test_extra_args_has_max_num_seqs(self, config):
        """monad runs at TP=1 with more per-seq headroom → higher max-num-seqs."""
        assert "--max-num-seqs" in config["VLLM_EXTRA_ARGS"]
        assert "--max-num-seqs 64" in config["VLLM_EXTRA_ARGS"]

    # VLLM_EXTRA_ARGS: what should NOT be present

    def test_extra_args_no_enforce_eager(self, config):
        """enforce-eager is only needed for multi-node TP=2; remove it at TP=1
        so CUDA graph capture runs and improves single-node throughput."""
        assert "--enforce-eager" not in config["VLLM_EXTRA_ARGS"], (
            "monad (TP=1) must NOT have --enforce-eager in VLLM_EXTRA_ARGS"
        )

    def test_key_uses_gpu_mem_util_not_gpu_memory_utilization(self, raw):
        """Must use GPU_MEM_UTIL (spark-tools convention), not GPU_MEMORY_UTILIZATION."""
        assert "GPU_MEM_UTIL=" in raw
        assert "GPU_MEMORY_UTILIZATION=" not in raw


# ---------------------------------------------------------------------------
# 3. node.env.dyad.example — aggressive profile (secondary node)
#    Mirrors Brian's node.env.spark-2: higher utilization, 256K context
# ---------------------------------------------------------------------------


class TestNodeEnvDyad:
    @pytest.fixture
    def config(self):
        assert os.path.isfile(DYAD_ENV), f"Must exist: {DYAD_ENV}"
        return parse_env_file(DYAD_ENV)

    @pytest.fixture
    def raw(self):
        with open(DYAD_ENV) as f:
            return f.read()

    # Parallelism

    def test_tp_size_is_1(self, config):
        """Split-node mode: each GB10 serves independently at TP=1."""
        assert config.get("TP_SIZE") == "1", "dyad must have TP_SIZE=1"

    # Memory

    def test_gpu_mem_util_aggressive(self, config):
        """Aggressive utilization when dyad is dedicated to inference."""
        assert config.get("GPU_MEM_UTIL") == "0.90"

    def test_max_model_len(self, config):
        """262144 (256K) tokens achievable at 0.90 util + FP8 KV on GB10."""
        assert config.get("MAX_MODEL_LEN") == "262144"

    # VLLM_EXTRA_ARGS: what should be present

    def test_extra_args_has_prefix_caching(self, config):
        assert "--enable-prefix-caching" in config["VLLM_EXTRA_ARGS"]

    def test_extra_args_has_block_size_128(self, config):
        assert "--block-size 128" in config["VLLM_EXTRA_ARGS"]

    def test_extra_args_has_flashinfer_backend(self, config):
        assert "--attention-backend flashinfer" in config["VLLM_EXTRA_ARGS"]

    def test_extra_args_has_fp8_kv_cache(self, config):
        assert "--kv-cache-dtype fp8" in config["VLLM_EXTRA_ARGS"]

    def test_extra_args_has_tool_calling(self, config):
        assert "--enable-auto-tool-choice" in config["VLLM_EXTRA_ARGS"]

    def test_extra_args_has_trust_remote_code(self, config):
        assert "--trust-remote-code" in config["VLLM_EXTRA_ARGS"]

    def test_extra_args_has_max_num_seqs(self, config):
        """dyad runs longer contexts → lower max-num-seqs to stay in VRAM budget."""
        assert "--max-num-seqs" in config["VLLM_EXTRA_ARGS"]
        assert "--max-num-seqs 32" in config["VLLM_EXTRA_ARGS"]

    # VLLM_EXTRA_ARGS: what should NOT be present

    def test_extra_args_no_enforce_eager(self, config):
        """enforce-eager is only needed for multi-node TP=2; remove it at TP=1."""
        assert "--enforce-eager" not in config["VLLM_EXTRA_ARGS"], (
            "dyad (TP=1) must NOT have --enforce-eager in VLLM_EXTRA_ARGS"
        )

    def test_key_uses_gpu_mem_util_not_gpu_memory_utilization(self, raw):
        assert "GPU_MEM_UTIL=" in raw
        assert "GPU_MEMORY_UTILIZATION=" not in raw

    # Dyad vs monad differences

    def test_dyad_has_higher_gpu_util_than_monad(self):
        monad = parse_env_file(MONAD_ENV)
        dyad = parse_env_file(DYAD_ENV)
        assert float(dyad["GPU_MEM_UTIL"]) > float(monad["GPU_MEM_UTIL"])

    def test_dyad_has_longer_context_than_monad(self):
        monad = parse_env_file(MONAD_ENV)
        dyad = parse_env_file(DYAD_ENV)
        assert int(dyad["MAX_MODEL_LEN"]) > int(monad["MAX_MODEL_LEN"])

    def test_dyad_has_fewer_max_num_seqs_than_monad(self):
        """Longer context per sequence → fewer concurrent sequences on dyad."""

        def _extract_max_num_seqs(cfg: dict) -> int:
            args = cfg.get("VLLM_EXTRA_ARGS", "")
            for token in args.split():
                if token.isdigit():
                    # Find the value after --max-num-seqs
                    pass
            # Parse properly
            parts = args.split()
            for i, part in enumerate(parts):
                if part == "--max-num-seqs" and i + 1 < len(parts):
                    return int(parts[i + 1])
            raise ValueError("--max-num-seqs not found")

        monad_seqs = _extract_max_num_seqs(parse_env_file(MONAD_ENV))
        dyad_seqs = _extract_max_num_seqs(parse_env_file(DYAD_ENV))
        assert dyad_seqs < monad_seqs, (
            f"dyad max-num-seqs ({dyad_seqs}) should be < monad ({monad_seqs})"
        )


# ---------------------------------------------------------------------------
# 4. model.env.example — shared model configuration (multi-node TP=2 defaults)
# ---------------------------------------------------------------------------


class TestModelEnvExample:
    @pytest.fixture
    def config(self):
        assert os.path.isfile(MODEL_ENV), f"Must exist: {MODEL_ENV}"
        return parse_env_file(MODEL_ENV)

    @pytest.fixture
    def raw(self):
        with open(MODEL_ENV) as f:
            return f.read()

    # Core identity fields

    def test_has_model_name(self, config):
        assert "MODEL_NAME" in config
        assert config["MODEL_NAME"], "MODEL_NAME must be non-empty"

    def test_has_served_model_name_field(self, config):
        """User's model.env has SERVED_MODEL_NAME as a separate top-level field."""
        assert "SERVED_MODEL_NAME" in config

    def test_has_vllm_extra_args(self, config):
        assert "VLLM_EXTRA_ARGS" in config
        assert config["VLLM_EXTRA_ARGS"], "VLLM_EXTRA_ARGS must be non-empty"

    # Parallelism & memory (multi-node defaults)

    def test_tp_size_is_2(self, config):
        """Shared model.env defaults to TP=2 (multi-node); per-node files override."""
        assert config.get("TP_SIZE") == "2"

    def test_has_max_model_len(self, config):
        assert "MAX_MODEL_LEN" in config
        assert config["MAX_MODEL_LEN"].isdigit()

    def test_has_gpu_mem_util(self, config):
        assert "GPU_MEM_UTIL" in config

    def test_key_uses_gpu_mem_util_not_gpu_memory_utilization(self, raw):
        assert "GPU_MEM_UTIL=" in raw
        assert "GPU_MEMORY_UTILIZATION=" not in raw

    # TRT-LLM settings (multi-backend architecture)

    def test_has_trtllm_port(self, config):
        """spark-tools supports TRT-LLM; TRTLLM_PORT must be defined."""
        assert "TRTLLM_PORT" in config

    def test_has_trtllm_max_batch_size(self, config):
        assert "TRTLLM_MAX_BATCH_SIZE" in config

    def test_has_trtllm_max_num_tokens(self, config):
        assert "TRTLLM_MAX_NUM_TOKENS" in config

    def test_has_trtllm_gpu_mem_fraction(self, config):
        assert "TRTLLM_GPU_MEM_FRACTION" in config

    # VLLM_EXTRA_ARGS: flags that MUST be present

    def test_extra_args_has_fp8_kv_cache(self, config):
        """--kv-cache-dtype fp8 saves ~40% KV memory; required for large models."""
        assert "--kv-cache-dtype fp8" in config["VLLM_EXTRA_ARGS"]

    def test_extra_args_has_auto_tool_choice(self, config):
        assert "--enable-auto-tool-choice" in config["VLLM_EXTRA_ARGS"]

    def test_extra_args_has_tool_call_parser(self, config):
        assert "--tool-call-parser" in config["VLLM_EXTRA_ARGS"]

    def test_extra_args_has_prefix_caching(self, config):
        assert "--enable-prefix-caching" in config["VLLM_EXTRA_ARGS"]

    def test_extra_args_has_trust_remote_code(self, config):
        assert "--trust-remote-code" in config["VLLM_EXTRA_ARGS"]

    def test_extra_args_has_block_size_128(self, config):
        assert "--block-size 128" in config["VLLM_EXTRA_ARGS"]

    def test_extra_args_has_flashinfer_attention_backend(self, config):
        assert "--attention-backend flashinfer" in config["VLLM_EXTRA_ARGS"]

    def test_extra_args_has_enforce_eager(self, config):
        """--enforce-eager is required for multi-node TP=2 on GB10 (avoids Triton
        allocator crash during CUDA graph capture). Per-node files remove it."""
        assert "--enforce-eager" in config["VLLM_EXTRA_ARGS"], (
            "model.env (multi-node TP=2 default) must include --enforce-eager"
        )

    def test_extra_args_has_max_num_seqs(self, config):
        assert "--max-num-seqs" in config["VLLM_EXTRA_ARGS"]

    # VLLM_EXTRA_ARGS: flags that must NOT be present (moved to proper keys)

    def test_extra_args_no_gpu_memory_utilization_flag(self, config):
        """GPU utilization is a first-class key (GPU_MEM_UTIL) not an extra arg."""
        assert "--gpu-memory-utilization" not in config["VLLM_EXTRA_ARGS"]

    def test_extra_args_no_distributed_executor_backend(self, config):
        """Distributed executor is set per-template; not hardcoded in model.env."""
        assert "--distributed-executor-backend" not in config["VLLM_EXTRA_ARGS"]

    def test_extra_args_no_max_num_batched_tokens(self, config):
        assert "--max-num-batched-tokens" not in config["VLLM_EXTRA_ARGS"]

    def test_extra_args_no_served_model_name_flag(self, config):
        """SERVED_MODEL_NAME is its own top-level key; must not duplicate in args."""
        assert "--served-model-name" not in config["VLLM_EXTRA_ARGS"]


# ---------------------------------------------------------------------------
# 5. Cross-file invariants
# ---------------------------------------------------------------------------


class TestCrossFileInvariants:
    def test_all_three_files_exist(self):
        for path in ALL_CONFIG_FILES:
            assert os.path.isfile(path), f"Config file must exist: {path}"

    def test_both_node_files_have_tp_size_1(self):
        """Split-node invariant: every node file must override TP to 1."""
        for path, label in [(MONAD_ENV, "monad"), (DYAD_ENV, "dyad")]:
            cfg = parse_env_file(path)
            assert cfg.get("TP_SIZE") == "1", f"{label} must have TP_SIZE=1"

    def test_node_files_do_not_set_tp_size_2(self):
        for path, label in [(MONAD_ENV, "monad"), (DYAD_ENV, "dyad")]:
            cfg = parse_env_file(path)
            assert cfg.get("TP_SIZE") != "2", (
                f"{label} must not use TP_SIZE=2 (that's the multi-node shared default)"
            )

    def test_node_files_both_drop_enforce_eager(self):
        """Both per-node files must omit --enforce-eager (single-node TP=1 mode)."""
        for path, label in [(MONAD_ENV, "monad"), (DYAD_ENV, "dyad")]:
            cfg = parse_env_file(path)
            assert "--enforce-eager" not in cfg.get("VLLM_EXTRA_ARGS", ""), (
                f"{label} must not have --enforce-eager in VLLM_EXTRA_ARGS"
            )

    def test_model_env_has_enforce_eager_node_files_do_not(self):
        """Exactly illustrates the TP=2 vs TP=1 distinction.

        model.env  (TP=2, multi-node) → needs --enforce-eager
        node files (TP=1, split-node) → must NOT have --enforce-eager
        """
        model_cfg = parse_env_file(MODEL_ENV)
        assert "--enforce-eager" in model_cfg["VLLM_EXTRA_ARGS"], (
            "model.env must have --enforce-eager for multi-node TP=2"
        )
        for path, label in [(MONAD_ENV, "monad"), (DYAD_ENV, "dyad")]:
            cfg = parse_env_file(path)
            assert "--enforce-eager" not in cfg["VLLM_EXTRA_ARGS"], (
                f"{label} must not carry --enforce-eager (TP=1 runs CUDA graphs fine)"
            )

    def test_node_files_both_have_fp8_kv_cache(self):
        """FP8 KV cache must be active in split-node mode too."""
        for path, label in [(MONAD_ENV, "monad"), (DYAD_ENV, "dyad")]:
            cfg = parse_env_file(path)
            assert "--kv-cache-dtype fp8" in cfg.get("VLLM_EXTRA_ARGS", ""), (
                f"{label} must keep --kv-cache-dtype fp8"
            )

    def test_node_files_both_have_tool_calling(self):
        """Tool-calling must remain enabled in per-node overrides."""
        for path, label in [(MONAD_ENV, "monad"), (DYAD_ENV, "dyad")]:
            cfg = parse_env_file(path)
            assert "--enable-auto-tool-choice" in cfg.get("VLLM_EXTRA_ARGS", ""), (
                f"{label} must keep --enable-auto-tool-choice"
            )

    def test_model_env_tp_size_higher_than_node_envs(self):
        """model.env defaults to TP=2 for multi-node; node files override to 1."""
        model_cfg = parse_env_file(MODEL_ENV)
        monad_cfg = parse_env_file(MONAD_ENV)
        dyad_cfg = parse_env_file(DYAD_ENV)
        assert int(model_cfg["TP_SIZE"]) > int(monad_cfg["TP_SIZE"])
        assert int(model_cfg["TP_SIZE"]) > int(dyad_cfg["TP_SIZE"])

    def test_dyad_has_higher_util_than_monad(self):
        monad_cfg = parse_env_file(MONAD_ENV)
        dyad_cfg = parse_env_file(DYAD_ENV)
        assert float(dyad_cfg["GPU_MEM_UTIL"]) > float(monad_cfg["GPU_MEM_UTIL"])

    def test_dyad_has_larger_context_than_monad(self):
        monad_cfg = parse_env_file(MONAD_ENV)
        dyad_cfg = parse_env_file(DYAD_ENV)
        assert int(dyad_cfg["MAX_MODEL_LEN"]) > int(monad_cfg["MAX_MODEL_LEN"])
