"""Tests for all spark-tools systemd service templates.

Adapted from Brian's dgx-spark-cluster test_standalone_service_template.py
and extended to cover the full template set.

spark-tools ships five templates (in systemd/):
  spark-proxy.service.template       — uvicorn proxy (no Ray / Docker needed)
  spark-ray-head.service.template    — Ray head node (docker run, ray start --head)
  spark-ray-vllm.service.template    — vLLM-via-Ray (docker exec, requires Ray head)
  spark-ray-worker.service.template  — Ray worker node (docker run, ray start --address)
  spark-swarm-stack.service.template — TRT-LLM via Docker Swarm (docker stack deploy)

Each class below focuses on one template.  A shared DryRunSubstitution class at
the end verifies every template is free of un-substituted {{PLACEHOLDER}} tokens
after replacing its own placeholders with dummy values.
"""

import os
import re

import pytest

from helpers import REPO_ROOT

# ---------------------------------------------------------------------------
# Template paths
# ---------------------------------------------------------------------------

SYSTEMD_DIR = os.path.join(REPO_ROOT, "systemd")

PROXY_TPL = os.path.join(SYSTEMD_DIR, "spark-proxy.service.template")
RAY_HEAD_TPL = os.path.join(SYSTEMD_DIR, "spark-ray-head.service.template")
RAY_VLLM_TPL = os.path.join(SYSTEMD_DIR, "spark-ray-vllm.service.template")
RAY_WORKER_TPL = os.path.join(SYSTEMD_DIR, "spark-ray-worker.service.template")
SWARM_STACK_TPL = os.path.join(SYSTEMD_DIR, "spark-swarm-stack.service.template")

ALL_TEMPLATES = [
    PROXY_TPL,
    RAY_HEAD_TPL,
    RAY_VLLM_TPL,
    RAY_WORKER_TPL,
    SWARM_STACK_TPL,
]

# Placeholder sets per template (used in dry-run substitution tests)
_PLACEHOLDERS: dict[str, dict[str, str]] = {
    PROXY_TPL: {
        "{{USERNAME}}": "testuser",
        "{{VLLM_BACKENDS}}": "http://localhost:8000",
        "{{SPARK_TOOLS_DIR}}": "/home/testuser/spark-tools",
        "{{PROXY_PORT}}": "9000",
    },
    RAY_HEAD_TPL: {
        "{{USERNAME}}": "testuser",
        "{{QSFP_IFACE}}": "eth0",
        "{{HEAD_IP}}": "192.168.1.10",
        "{{RAY_VLLM_IMAGE}}": "myregistry/ray-vllm:latest",
    },
    RAY_VLLM_TPL: {
        "{{USERNAME}}": "testuser",
        "{{PORT}}": "8000",
    },
    RAY_WORKER_TPL: {
        "{{USERNAME}}": "testuser",
        "{{QSFP_IFACE}}": "eth0",
        "{{HEAD_IP}}": "192.168.1.10",
        "{{RAY_VLLM_IMAGE}}": "myregistry/ray-vllm:latest",
    },
    SWARM_STACK_TPL: {
        "{{USERNAME}}": "testuser",
        "{{COMPOSE_FILE}}": "/home/testuser/spark-tools/trtllm/compose.yml",
    },
}


def _read(path: str) -> str:
    assert os.path.isfile(path), f"Template must exist: {path}"
    with open(path) as f:
        return f.read()


def _substitute(path: str) -> str:
    """Replace all known placeholders with dummy values."""
    content = _read(path)
    for placeholder, value in _PLACEHOLDERS[path].items():
        content = content.replace(placeholder, value)
    return content


# ---------------------------------------------------------------------------
# 1. All templates exist
# ---------------------------------------------------------------------------


class TestAllTemplatesExist:
    @pytest.mark.parametrize(
        "path",
        ALL_TEMPLATES,
        ids=[os.path.basename(p) for p in ALL_TEMPLATES],
    )
    def test_template_file_exists(self, path):
        assert os.path.isfile(path), f"Expected template at: {path}"


# ---------------------------------------------------------------------------
# 2. spark-proxy.service.template
#    Runs the uvicorn proxy; no Ray or Docker dependency.
# ---------------------------------------------------------------------------


class TestProxyTemplate:
    @pytest.fixture
    def content(self):
        return _read(PROXY_TPL)

    # [Unit] section

    def test_description(self, content):
        assert "Spark" in content and "Proxy" in content

    def test_after_network_not_docker(self, content):
        """Proxy only needs the network — no Docker dependency."""
        assert "After=network.target" in content

    def test_not_after_docker(self, content):
        """Proxy must NOT depend on docker.service (it's a pure Python process)."""
        assert "After=docker.service" not in content

    # [Service] section

    def test_type_simple(self, content):
        assert "Type=simple" in content

    def test_user_placeholder(self, content):
        assert "User={{USERNAME}}" in content

    def test_environment_file_secret(self, content):
        assert "EnvironmentFile=" in content
        assert "secret.env" in content

    def test_working_directory_placeholder(self, content):
        assert "{{SPARK_TOOLS_DIR}}" in content

    def test_execstart_uses_uvicorn(self, content):
        assert "uvicorn" in content
        assert "vllm_proxy:app" in content

    def test_execstart_host_0000(self, content):
        assert "--host 0.0.0.0" in content

    def test_proxy_port_placeholder(self, content):
        assert "{{PROXY_PORT}}" in content

    def test_upstream_port_placeholder(self, content):
        assert "{{VLLM_BACKENDS}}" in content

    def test_restart_always(self, content):
        assert "Restart=always" in content

    # No Ray / no Docker exec

    def test_no_docker_run(self, content):
        """Proxy is a plain Python process — must not use docker run."""
        assert "docker run" not in content

    def test_no_docker_exec(self, content):
        assert "docker exec" not in content

    def test_no_ray_reference(self, content):
        assert "ray start" not in content
        assert "Requires=spark-ray" not in content


# ---------------------------------------------------------------------------
# 3. spark-ray-head.service.template
#    Starts the Ray head node via docker run.
# ---------------------------------------------------------------------------


class TestRayHeadTemplate:
    @pytest.fixture
    def content(self):
        return _read(RAY_HEAD_TPL)

    def test_after_docker_service(self, content):
        assert "After=docker.service" in content

    def test_requires_docker_service(self, content):
        assert "Requires=docker.service" in content

    def test_type_simple(self, content):
        assert "Type=simple" in content

    def test_user_placeholder(self, content):
        assert "User={{USERNAME}}" in content

    def test_exec_start_pre_removes_container(self, content):
        assert "ExecStartPre=-/usr/bin/docker rm -f spark-ray" in content

    def test_execstart_docker_run(self, content):
        assert "/usr/bin/docker run" in content

    def test_no_docker_exec(self, content):
        """Head starts Ray in a fresh container — no docker exec."""
        assert "docker exec" not in content

    def test_container_name_spark_ray(self, content):
        assert "--name spark-ray" in content

    def test_gpus_all(self, content):
        assert "--gpus all" in content

    def test_network_host(self, content):
        assert "--network host" in content

    def test_hf_cache_volume_mount(self, content):
        assert ".cache/huggingface" in content
        assert "-v" in content

    def test_ray_image_placeholder(self, content):
        assert "{{RAY_VLLM_IMAGE}}" in content

    def test_qsfp_iface_placeholder(self, content):
        assert "{{QSFP_IFACE}}" in content

    def test_head_ip_placeholder(self, content):
        assert "{{HEAD_IP}}" in content

    def test_ray_start_head_command(self, content):
        assert "ray start" in content
        assert "--head" in content

    def test_exec_stop_docker_stop(self, content):
        assert "ExecStop=/usr/bin/docker stop spark-ray" in content

    def test_restart_always(self, content):
        assert "Restart=always" in content

    def test_restart_sec_5(self, content):
        assert "RestartSec=5" in content


# ---------------------------------------------------------------------------
# 4. spark-ray-vllm.service.template
#    Runs vLLM inside the already-running Ray container via docker exec.
# ---------------------------------------------------------------------------


class TestRayVllmTemplate:
    @pytest.fixture
    def content(self):
        return _read(RAY_VLLM_TPL)

    def test_after_ray_head(self, content):
        """vLLM-via-Ray must start after the Ray head is up."""
        assert "After=spark-ray-head.service" in content

    def test_requires_ray_head(self, content):
        assert "Requires=spark-ray-head.service" in content

    def test_type_oneshot(self, content):
        assert "Type=oneshot" in content

    def test_remain_after_exit(self, content):
        assert "RemainAfterExit=yes" in content

    def test_user_placeholder(self, content):
        assert "User={{USERNAME}}" in content

    def test_environment_file_model_env(self, content):
        assert "EnvironmentFile=" in content
        assert "model.env" in content

    def test_exec_start_pre_waits_for_ray(self, content):
        """Must poll for Ray readiness before starting vLLM."""
        assert "ray status" in content

    def test_execstart_docker_exec(self, content):
        """vLLM is started inside the Ray container via docker exec."""
        assert "docker exec spark-ray" in content

    def test_vllm_serve_command(self, content):
        assert "vllm serve" in content
        assert "${MODEL_NAME}" in content

    def test_tensor_parallel_size_arg(self, content):
        assert "--tensor-parallel-size ${TP_SIZE}" in content

    def test_max_model_len_arg(self, content):
        assert "--max-model-len ${MAX_MODEL_LEN}" in content

    def test_gpu_memory_util_arg(self, content):
        assert "--gpu-memory-utilization ${GPU_MEM_UTIL}" in content

    def test_distributed_executor_backend_ray(self, content):
        """Multi-node Ray mode must pass --distributed-executor-backend ray."""
        assert "--distributed-executor-backend ray" in content

    def test_vllm_extra_args(self, content):
        assert "${VLLM_EXTRA_ARGS}" in content

    def test_port_placeholder(self, content):
        assert "{{PORT}}" in content

    def test_exec_stop_pkill_vllm(self, content):
        assert "pkill" in content and "vllm serve" in content

    def test_restart_on_failure(self, content):
        assert "Restart=on-failure" in content

    def test_no_docker_run(self, content):
        """vLLM runs inside the existing Ray container — must NOT docker run."""
        assert "/usr/bin/docker run" not in content


# ---------------------------------------------------------------------------
# 5. spark-ray-worker.service.template
#    Joins the Ray cluster as a worker node via docker run.
# ---------------------------------------------------------------------------


class TestRayWorkerTemplate:
    @pytest.fixture
    def content(self):
        return _read(RAY_WORKER_TPL)

    def test_after_docker_service(self, content):
        assert "After=docker.service" in content

    def test_requires_docker_service(self, content):
        assert "Requires=docker.service" in content

    def test_type_simple(self, content):
        assert "Type=simple" in content

    def test_user_placeholder(self, content):
        assert "User={{USERNAME}}" in content

    def test_exec_start_pre_removes_container(self, content):
        assert "ExecStartPre=-/usr/bin/docker rm -f spark-ray" in content

    def test_execstart_docker_run(self, content):
        assert "/usr/bin/docker run" in content

    def test_container_name_spark_ray(self, content):
        assert "--name spark-ray" in content

    def test_gpus_all(self, content):
        assert "--gpus all" in content

    def test_network_host(self, content):
        assert "--network host" in content

    def test_hf_cache_volume_mount(self, content):
        assert ".cache/huggingface" in content

    def test_ray_image_placeholder(self, content):
        assert "{{RAY_VLLM_IMAGE}}" in content

    def test_qsfp_iface_placeholder(self, content):
        assert "{{QSFP_IFACE}}" in content

    def test_head_ip_placeholder(self, content):
        assert "{{HEAD_IP}}" in content

    def test_ray_start_worker_address(self, content):
        """Worker must join an existing cluster — not start a new head."""
        assert "ray start" in content
        assert "--address=" in content
        assert "--head" not in content

    def test_exec_stop_docker_stop(self, content):
        assert "ExecStop=/usr/bin/docker stop spark-ray" in content

    def test_restart_always(self, content):
        assert "Restart=always" in content

    def test_no_docker_exec(self, content):
        assert "docker exec" not in content


# ---------------------------------------------------------------------------
# 6. spark-swarm-stack.service.template
#    Deploys TRT-LLM via Docker Swarm stack — no Ray, no vLLM.
# ---------------------------------------------------------------------------


class TestSwarmStackTemplate:
    @pytest.fixture
    def content(self):
        return _read(SWARM_STACK_TPL)

    def test_after_docker_service(self, content):
        assert "After=docker.service" in content

    def test_requires_docker_service(self, content):
        assert "Requires=docker.service" in content

    def test_type_oneshot(self, content):
        assert "Type=oneshot" in content

    def test_remain_after_exit(self, content):
        assert "RemainAfterExit=yes" in content

    def test_user_placeholder(self, content):
        assert "User={{USERNAME}}" in content

    def test_execstart_docker_stack_deploy(self, content):
        assert "docker stack deploy" in content

    def test_compose_file_placeholder(self, content):
        assert "{{COMPOSE_FILE}}" in content

    def test_stack_name_trtllm(self, content):
        assert "trtllm-multinode" in content

    def test_exec_stop_stack_rm(self, content):
        assert "docker stack rm trtllm-multinode" in content

    def test_timeout_start_sec(self, content):
        assert "TimeoutStartSec=" in content

    def test_no_ray_reference(self, content):
        assert "ray start" not in content
        assert "spark-ray" not in content

    def test_no_vllm_serve(self, content):
        assert "vllm serve" not in content


# ---------------------------------------------------------------------------
# 7. Dry-run placeholder substitution — no {{...}} must remain after fill-in
# ---------------------------------------------------------------------------


class TestDryRunSubstitution:
    @pytest.mark.parametrize(
        "path",
        ALL_TEMPLATES,
        ids=[os.path.basename(p) for p in ALL_TEMPLATES],
    )
    def test_no_remaining_placeholders_after_substitution(self, path):
        """After substituting every declared placeholder, no {{X}} may remain."""
        content = _substitute(path)
        remaining = re.findall(r"\{\{[A-Z_]+\}\}", content)
        assert remaining == [], (
            f"{os.path.basename(path)}: un-substituted placeholders after fill-in: "
            f"{remaining}"
        )

    @pytest.mark.parametrize(
        "path",
        ALL_TEMPLATES,
        ids=[os.path.basename(p) for p in ALL_TEMPLATES],
    )
    def test_username_placeholder_is_substituted(self, path):
        """Every template must honour {{USERNAME}}."""
        content = _substitute(path)
        assert "testuser" in content, (
            f"{os.path.basename(path)}: {{{{USERNAME}}}} must be present and substituted"
        )

    def test_proxy_template_substituted_has_proxy_port(self):
        content = _substitute(PROXY_TPL)
        assert "9000" in content  # the dummy PROXY_PORT value

    def test_proxy_template_substituted_has_spark_tools_dir(self):
        content = _substitute(PROXY_TPL)
        assert "/home/testuser/spark-tools" in content

    def test_ray_head_substituted_has_head_ip(self):
        content = _substitute(RAY_HEAD_TPL)
        assert "192.168.1.10" in content

    def test_ray_vllm_substituted_has_port(self):
        content = _substitute(RAY_VLLM_TPL)
        assert "8000" in content  # the dummy PORT value

    def test_swarm_stack_substituted_has_compose_file(self):
        content = _substitute(SWARM_STACK_TPL)
        assert "/home/testuser/spark-tools/trtllm/compose.yml" in content


# ---------------------------------------------------------------------------
# 8. Cross-template invariants
# ---------------------------------------------------------------------------


class TestCrossTemplateInvariants:
    def test_all_templates_have_install_section(self):
        """Every template must have [Install] with WantedBy=multi-user.target."""
        for path in ALL_TEMPLATES:
            content = _read(path)
            assert "[Install]" in content, (
                f"{os.path.basename(path)}: missing [Install]"
            )
            assert "WantedBy=multi-user.target" in content, (
                f"{os.path.basename(path)}: missing WantedBy=multi-user.target"
            )

    def test_all_templates_have_unit_section(self):
        """Every template must have a [Unit] section with a Description."""
        for path in ALL_TEMPLATES:
            content = _read(path)
            assert "[Unit]" in content, f"{os.path.basename(path)}: missing [Unit]"
            assert "Description=" in content, (
                f"{os.path.basename(path)}: missing Description="
            )

    def test_all_templates_have_service_section(self):
        for path in ALL_TEMPLATES:
            content = _read(path)
            assert "[Service]" in content, (
                f"{os.path.basename(path)}: missing [Service]"
            )

    def test_all_templates_have_username_placeholder(self):
        """Every template requires {{USERNAME}} — no hardcoded user names."""
        for path in ALL_TEMPLATES:
            content = _read(path)
            assert "{{USERNAME}}" in content, (
                f"{os.path.basename(path)}: missing {{{{USERNAME}}}} placeholder"
            )

    def test_ray_templates_share_container_name(self):
        """All Ray templates must use the same container name (spark-ray) for
        consistency so ExecStop on one can affect the other."""
        for path in (RAY_HEAD_TPL, RAY_VLLM_TPL, RAY_WORKER_TPL):
            content = _read(path)
            assert "spark-ray" in content, (
                f"{os.path.basename(path)}: must reference 'spark-ray' container"
            )

    def test_proxy_does_not_require_ray(self):
        """Proxy must be deployable without Ray (standalone LAN proxy)."""
        content = _read(PROXY_TPL)
        assert "Requires=spark-ray" not in content
        assert "After=spark-ray" not in content

    def test_swarm_stack_does_not_require_ray(self):
        """TRT-LLM Swarm mode is independent of the Ray stack."""
        content = _read(SWARM_STACK_TPL)
        assert "Requires=spark-ray" not in content
        assert "ray start" not in content
