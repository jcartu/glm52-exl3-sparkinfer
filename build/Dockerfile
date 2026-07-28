# syntax=docker/dockerfile:1.7

ARG BASE_IMAGE=verdictai/glm52-exl3-sparkinfer@sha256:0433ae94665b769b78dd301f952d907508a3ba80bce47a1630ec20ade8812dff
FROM ${BASE_IMAGE}

ARG BASE_IMAGE
ARG VLLM_COMMIT
ARG SPARKINFER_COMMIT

RUN test -n "${VLLM_COMMIT}" && test -n "${SPARKINFER_COMMIT}"

LABEL org.opencontainers.image.title="GLM-5.2 EXL3 dynamic LoRA runtime" \
      org.opencontainers.image.description="Digest-pinned EXL3/Trellis runtime with dynamic BF16 rank-16 fully-sharded LoRA support" \
      org.opencontainers.image.base.name="${BASE_IMAGE}" \
      ai.verdict.base.digest="sha256:0433ae94665b769b78dd301f952d907508a3ba80bce47a1630ec20ade8812dff" \
      ai.verdict.vllm.commit="${VLLM_COMMIT}" \
      ai.verdict.sparkinfer.commit="${SPARKINFER_COMMIT}" \
      ai.verdict.lora.contract="dynamic,bfloat16,rank-16,fully-sharded,tp4"

# The named BuildKit contexts are exact Git commits supplied by the build command.
# Overlay Python sources while retaining the base image's ABI-matched native modules.
COPY --from=vllm-src /vllm/ /opt/venv/lib/python3.12/site-packages/vllm/
COPY --from=sparkinfer-src /sparkinfer/ /opt/venv/lib/python3.12/site-packages/sparkinfer/

RUN /opt/venv/bin/python -m compileall -q \
        /opt/venv/lib/python3.12/site-packages/vllm \
        /opt/venv/lib/python3.12/site-packages/sparkinfer \
    && /opt/venv/bin/python - <<'PY'
import pathlib
import sparkinfer
import torch
import vllm
from sparkinfer.moe import trellis_moe
from vllm.lora.layers.column_parallel_linear import ColumnParallelLinearWithLoRA

required = (
    pathlib.Path("/opt/glm52/lib/exllamav3_ext.cpython-312-x86_64-linux-gnu.so"),
    pathlib.Path("/opt/glm52/lib/libexl3_torch212_compat.so"),
    pathlib.Path("/opt/glm52/kv-scale/glm52_nvfp4_mla_outer_scales.json"),
    pathlib.Path(vllm.__file__).parent / "_C_stable_libtorch.abi3.so",
)
missing = [str(path) for path in required if not path.is_file()]
if missing:
    raise RuntimeError(f"missing runtime assets: {missing}")
if not all(
    callable(getattr(trellis_moe, name, None))
    for name in ("run_fc1", "activation_output", "run_fc2", "run_sum")
):
    raise RuntimeError("staged Trellis adapter API is unavailable")
if not hasattr(ColumnParallelLinearWithLoRA, "project_mla_kv_b"):
    raise RuntimeError("MLA kv_b LoRA projection API is unavailable")
print(vllm.__file__)
print(sparkinfer.__file__)
print(torch.__version__)
PY

ENV VLLM_EXL3_EXT_PATH=/opt/glm52/lib \
    VLLM_EXL3_ABI_SHIM=/opt/glm52/lib/libexl3_torch212_compat.so \
    VLLM_NVFP4_MLA_SCALES_FILE=/opt/glm52/kv-scale/glm52_nvfp4_mla_outer_scales.json
