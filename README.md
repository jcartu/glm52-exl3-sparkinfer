# GLM-5.2 EXL3-TR3 — production serving stack for SM120

Production-grade serving of **GLM-5.2 (753B MoE)** quantized to **EXL3 Trellis 3.0 bpw** with a
rank-sliced **EXL3 MTP-78 draft head** and dynamically loaded, fully sharded **BF16 rank-16
LoRA adapters**, on 4× NVIDIA RTX PRO 6000 Blackwell (SM120, 96 GB, PCIe — no NVLink).

The qualified default is TP4 / DCP4, `nvfp4_ds_mla` KV, MTP-3 greedy speculative decoding,
FULL_AND_PIECEWISE CUDA graphs, a 32,768-token request cap, and two active sequences. These
limits are deliberate: the adapter-resident model uses about 93.2 GiB per GPU.

**Qualified context geometry**:
- The underlying checkpoint advertises a 1,048,576-token native window.
- The dynamic-LoRA preset ships at **32,768 tokens** and `--max-num-seqs 2`.
- The measured KV pool is **252,928 tokens** (2.45 GiB per rank; vLLM reports 7.71875×
  theoretical 32k concurrency before the explicit two-sequence scheduler cap).
- A **30,553-token adapted prompt** plus decode completed successfully in the release
  qualification, covering 93.24% of the shipped request cap.

The adapter contract is intentionally narrow and explicit: Hugging Face PEFT safetensors,
`torch.bfloat16`, rank 16, fully sharded across TP4, one resident adapter, loaded and unloaded
through vLLM's runtime LoRA endpoints. The image, base, and both source trees are reproducibly
pinned.

| Artifact | Pin |
|---|---|
| Runtime image | `ghcr.io/jcartu/glm52-exl3-lora@sha256:7af67ad8dd7406f0a4de8ac68be872d24697a4191ba9b23c44db1d265cc9c338` |
| Common base | `verdictai/glm52-exl3-sparkinfer@sha256:0433ae94665b769b78dd301f952d907508a3ba80bce47a1630ec20ade8812dff` |
| vLLM source | [`95d7914de19c56a21a1668f3b7273b5798424e47`](https://github.com/jcartu/vllm/commit/95d7914de19c56a21a1668f3b7273b5798424e47), tag `exl3-lora-experts-r1` |
| Sparkinfer source | [`fc8051efee755563e2c7a4ce87ce8b683db58381`](https://github.com/jcartu/sparkinfer/commit/fc8051efee755563e2c7a4ce87ce8b683db58381), tag `exl3-lora-trellis-r1` |
| Model weights | [brandonmusic/GLM-5.2-EXL3-TR3-3.0bpw](https://huggingface.co/brandonmusic/GLM-5.2-EXL3-TR3-3.0bpw) |
| Qualification adapter | BF16 rank 16, alpha 32; safetensors SHA-256 `0c7c99940c7459a568441f2cd774c4c2ec0fe06be725e634497980f6fa2f6a5b` |

## Quickstart

```bash
# 1. Download the model and obtain a compatible BF16 rank-16 PEFT adapter.
huggingface-cli download brandonmusic/GLM-5.2-EXL3-TR3-3.0bpw \
  --local-dir ./GLM-5.2-EXL3-TR3-3.0bpw

# 2. Start the digest-pinned runtime.
cd deploy
export MODEL_DIR="$(realpath ../GLM-5.2-EXL3-TR3-3.0bpw)"
export ADAPTER_DIR="$(realpath /path/to/rank16-adapter)"
export CACHE_DIR="${HOME}/.cache/glm52-exl3-lora-v31"
./server.sh start

# 3. After /health is ready, register the mounted adapter dynamically.
LORA_NAME=my-adapter ./server.sh load
curl -fsS http://127.0.0.1:8000/v1/models | jq .
```

The OpenAI-compatible completion and chat endpoints come up on `:8000`. The model load and
graph capture take several minutes on a cold cache. The first adapter registration measured
27.5 seconds; unloading measured 2 ms and a warm reload measured 8.0 seconds.

The preset encodes two runtime requirements found during GPU qualification:
`VLLM_PCIE_ALLREDUCE_BACKEND=b12x` and `VLLM_DISABLE_SHARED_EXPERTS_STREAM=1`. B12X PCIe
oneshot channels are stream-affine; overriding the latter reintroduces a CUDA-graph failure.

## What's validated

Full evidence and exact measurements: [`docs/RELEASE_TEST_SUITE.md`](docs/RELEASE_TEST_SUITE.md).
All rows below were exercised on 4× RTX PRO 6000 with the published image filesystem.

| Check | Result |
|---|---|
| Remote reproducible build | GitHub tags resolved to vLLM `95d7914…` and Sparkinfer `fc8051e…`; image published and anonymously pulled by digest |
| Source validation | vLLM focused LoRA `109 passed`; CPU MLA `22 passed`; EXL3 bridge/device `14 passed`; Sparkinfer GPU suite `29 passed` |
| Shipped API harness | health, greedy chat, four tool-call scenarios, and streaming deltas: **ALL PASS** for base and adapter |
| Dynamic lifecycle | load `200` in 27.516 s; unload `200` in 0.002 s; warm reload `200` in 7.974 s |
| Adapter effect | 32/32 shared token log-probabilities changed; exact release-image smoke max delta `0.5778587` |
| Base isolation | base text and token log-probabilities remained bit-for-bit identical across adapter registration and unload |
| CUDA graphs + mixed routing | base and adapter requests passed concurrently under FULL_AND_PIECEWISE graphs |
| DCP4 + MTP-3 | B12X world-size-4 collectives captured and served; 1,599 / 1,839 draft tokens accepted (**86.95%**) |
| Prefix cache | 3,265-token base request `7.503 → 0.649 s` (11.57×); adapter `3.906 → 0.868 s` (4.50×) |
| Near-capacity context | adapted 30,553-token prompt completed in 18.795 s |
| Warm decode | base **84.36 tok/s**; adapter **62.76 tok/s**; mixed concurrency-2 **97.83 aggregate tok/s** |
| Deterministic quality gates | factual, arithmetic, Python expression, and exact-format instruction: **4/4 base and 4/4 adapter** |
| Retrieval at shipped capacity | **9/9 base + 9/9 adapter** at 8k, 16k, and 30k prompt targets × depths 0.1/0.5/0.9 |

The older long-context and independent quality studies remain available as historical baseline
evidence in [`docs/benchmarks/`](docs/benchmarks/) and
[`docs/independent-eval/`](docs/independent-eval/); they used the pre-LoRA image lineage and
different capacity presets.

## Repository layout

```text
deploy/                  ready-to-run serving
  docker-compose.yml       qualified TP4/DCP4/MTP-3 dynamic-LoRA preset
  docker-compose-dcp1.yml  qualified TP4/DCP1 graph fallback; MTP disabled
  server.sh                lifecycle plus dynamic load/unload commands
build/
  Dockerfile               digest-pinned base plus immutable Git source contexts
  overlay/                 historical pre-merge source snapshot; not copied by the current build
tests/                   baseline health, generation, tools, and retrieval harnesses
docs/
  RELEASE_TEST_SUITE.md    current LoRA qualification plus historical full-suite evidence
  independent-eval/        third-party pre-LoRA evaluation report
  benchmarks/              dated benchmark sessions
```

## Qualified presets

| | DCP4 default | DCP1 fallback |
|---|---|---|
| Decode context parallelism | 4 | 1 |
| MTP | 3 greedy | disabled |
| GPU memory utilization | 0.93 | 0.90 |
| CUDA graph sizes | 4, 8 | 1, 2, 4, 8 |
| Max model length / sequences | 32,768 / 2 | 32,768 / 2 |
| Dynamic BF16 rank-16 LoRA | qualified | qualified |

Use DCP4 for the released MTP/prefix-cache path. The DCP1 file is a conservative graph fallback
matching the separately qualified MTP-off topology; enabling MTP in that file is possible but
was not part of this release gate. Historical high-capacity DCP4/DCP1 measurements in
[`docs/benchmarks/`](docs/benchmarks/) used older images without the 14.3 GB adapter and must not
be treated as capacity claims for this preset.

## Building the image yourself

The image overlays the exact Python packages from the two published source tags while retaining
the base image's ABI-matched native modules:

```bash
docker buildx build --load \
  --file build/Dockerfile \
  --build-context vllm-src=https://github.com/jcartu/vllm.git#exl3-lora-experts-r1 \
  --build-context sparkinfer-src=https://github.com/jcartu/sparkinfer.git#exl3-lora-trellis-r1 \
  --build-arg VLLM_COMMIT=95d7914de19c56a21a1668f3b7273b5798424e47 \
  --build-arg SPARKINFER_COMMIT=fc8051efee755563e2c7a4ce87ce8b683db58381 \
  --tag glm52-exl3-lora:v31 .
```

The build fails unless both source pins are supplied and verifies the staged Trellis API, MLA
LoRA projection API, native runtime assets, and Python bytecode compilation before exporting.
The release build log resolved both annotated tags to the full commits shown above.

## Running validation

```bash
bash -n deploy/server.sh
MODEL_DIR=/path/to/model ADAPTER_DIR=/path/to/adapter CACHE_DIR=/tmp/cache \
  docker compose -f deploy/docker-compose.yml config --quiet

cd deploy
MODEL_DIR=/path/to/model ADAPTER_DIR=/path/to/adapter ./server.sh start
LORA_NAME=my-adapter ./server.sh load
LORA_NAME=my-adapter ./server.sh status
LORA_NAME=my-adapter ./server.sh unload
```

The baseline harnesses under `tests/` remain useful for health, generation, tool calling, and
retrieval regression checks. Dynamic-LoRA release evidence is recorded in the test-suite
addendum because it requires the qualified adapter artifact and four SM120 GPUs.

## Known limitations

- The shipped scheduler cap is two active sequences and the request cap is 32,768 tokens.
  Raising either changes the measured memory and graph contract and requires requalification.
- One BF16 rank-16 adapter is supported at a time. Unload it before replacing files or loading a
  different adapter; 3D LoRA weights were not qualified.
- Adapter decode was **62.76 tok/s** versus **84.36 tok/s** base in the measured 128-token
  sequence workload. This release makes no universal performance-improvement claim.
- The base image repeatedly logs an optional FlashAttention-2 ABI probe error
  (`_vllm_fa2_C.abi3.so` undefined symbol). The selected backend is `B12X_MLA_SPARSE`; graph
  capture and serving succeed, so this is known startup noise rather than the active path.
- First use of previously unseen prefill/LoRA shapes can emit JIT-monitor warnings and incur a
  latency spike. Keep the compilation cache mounted and warm representative shapes before
  latency-sensitive traffic.
- B12X PCIe oneshot all-reduce is stream-affine. Do not override
  `VLLM_DISABLE_SHARED_EXPERTS_STREAM=1` in the qualified graph preset.
- SM120 lacks the TMEM/TCGEN05/WGMMA features required by some sparse-MLA families; the released
  `B12X_MLA_SPARSE` plus `nvfp4_ds_mla` path is the qualified implementation.

## Rollback

Stop the release compose project, then restart a previously retained image or container:

```bash
cd deploy
./server.sh stop

# Qualification-host rollback retained during release:
docker start glm52-exl3-v26-5001
```

The retained host container is `e08c3601feed…`, backed by local image
`sha256:d55205e3ae3d81f00a2770dee91c2bf1662a5efe29c6c897be5ac3010ca75895`.
For a portable rollback, set `IMAGE` to any prior digest-pinned runtime and run `server.sh
start` with the same `MODEL_DIR`, `ADAPTER_DIR`, and `CACHE_DIR`. Do not delete the retained
container until the new release completes its burn-in window.

## Contributing

PRs welcome — see [CONTRIBUTING.md](CONTRIBUTING.md). Rough split: **runtime/kernel changes**
belong upstream (vllm #139 / sparkinfer #49 lineage); **deployment, docs, validation harnesses,
and results** belong here. Every performance or correctness claim in a PR should come with the
command that produced it and, where possible, a `tests/` harness addition — that's the standard
the existing results were held to, including the ones that refuted our own assumptions.

## Acknowledgments

- [local-inference-lab](https://github.com/local-inference-lab) — the Gilded Gnosis / SparkInfer
  v20 common base and review of the EXL3 PRs
- malaiwah — the LDLQ-calibrated EXL3-TR3 rank-sliced MTP-78 head
- turboderp's [exllamav3](https://github.com/turboderp-org/exllamav3) — the EXL3 format and kernels
- Zhipu AI — GLM-5.2 (model weights under their license; this repo covers serving code only)

## License

Repository contents (scripts, configs, docs): [MIT](LICENSE). Model weights, the base image, and
upstream projects carry their own licenses.
