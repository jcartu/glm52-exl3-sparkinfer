# Release Test Suite — GLM-5.2-EXL3-TR3-3.0bpw

The current dynamic-LoRA release addendum was measured 2026-07-28 on 4× RTX PRO 6000
Blackwell 96 GB. Sections after the addendum preserve the 2026-07-25 pre-LoRA image-lineage
suite and its embedded raw benchmark logs; those older capacity/performance results are not
claims about the adapter-resident preset.

Section 8 of the historical suite reports an **independent third-party evaluation** run by
[malaiwah/glm52-exl3-vast](https://github.com/malaiwah/glm52-exl3-vast/tree/main).

---

## 2026-07-28 addendum — dynamic BF16 rank-16 LoRA release

### Immutable inputs

| Artifact | Pin |
|---|---|
| Published runtime | `ghcr.io/jcartu/glm52-exl3-lora@sha256:7af67ad8dd7406f0a4de8ac68be872d24697a4191ba9b23c44db1d265cc9c338` |
| Base runtime | `verdictai/glm52-exl3-sparkinfer@sha256:0433ae94665b769b78dd301f952d907508a3ba80bce47a1630ec20ade8812dff` |
| vLLM | `95d7914de19c56a21a1668f3b7273b5798424e47`, tag `exl3-lora-experts-r1` |
| Sparkinfer | `fc8051efee755563e2c7a4ce87ce8b683db58381`, tag `exl3-lora-trellis-r1` |
| Adapter contract | PEFT safetensors, BF16, rank 16, alpha 32, fully sharded TP4 |
| Qualification adapter SHA-256 | `0c7c99940c7459a568441f2cd774c4c2ec0fe06be725e634497980f6fa2f6a5b` |

The final image was rebuilt with named GitHub BuildKit contexts. The build log resolved the
annotated source tags to the full commits above, compiled both installed Python trees, checked
the staged Trellis and MLA projection APIs, exported OCI index
`sha256:7af67ad8dd7406f0a4de8ac68be872d24697a4191ba9b23c44db1d265cc9c338`, and then passed an
anonymous digest pull. The exact registry-tagged filesystem was started separately for the
final smoke; adapter load returned 200 and 32/32 token log-probabilities changed.

### Source gates

| Gate | Result |
|---|---|
| vLLM focused LoRA suite | `109 passed` |
| CPU MLA regression selection | `22 passed, 2314 deselected` |
| EXL3 device and modular bridge | `14 passed` |
| Sparkinfer pinned GPU suite | `29 passed, 1 warning` |
| Ruff / bytecode compilation | PASS in both source trees |
| No-adapter monolithic parity | SHA-256 `5560efedbb5abedc6c44c9d37e2e439536b621c75d4f1c96a127bf049714065d` |

The real layer-10/expert-0 factor oracle observed finite, nonzero BF16 rank-16 factors:

| Projection | A shape | B shape | B norm | SHA prefix |
|---|---:|---:|---:|---|
| gate | 16×6144 | 2048×16 | 0.070327 | `1e28…` |
| up | 16×6144 | 2048×16 | 0.0837356 | `a243…` |
| down | 16×2048 | 6144×16 | 0.158349 | `7589…` |

### Qualified serving contract

| Setting | Value |
|---|---|
| GPUs / topology | 4× RTX PRO 6000, TP4, DCP4 A2A |
| KV / attention / MoE | `nvfp4_ds_mla` / `B12X_MLA_SPARSE` / `b12x` |
| Speculation | MTP-3, greedy draft sampling |
| Graphs | FULL_AND_PIECEWISE, capture sizes 4 and 8 |
| Scheduler | max model length 32,768; max sequences 2; max batched tokens 3,072 |
| Memory | utilization 0.93; model load 81.34 GiB/rank; actual graph pool 0.14 GiB |
| KV capacity | 2.45 GiB/rank, 252,928 tokens |
| LoRA | dynamic updates, BF16, max rank 16, fully sharded, max adapters 1 |
| PCIe all-reduce | B12X; shared-expert auxiliary stream disabled |

At utilization 0.90, DCP4+MTP-3 correctly failed before serving because available KV memory was
`-0.67 GiB`; this was capacity, not a source error. At 0.93 the same topology captured all
graphs and exposed 252,928 KV tokens. B12X all-reduce with the shared-expert auxiliary stream
enabled failed with `PCIe oneshot allreduce channels are stream-affine`; the released preset
therefore fixes `VLLM_DISABLE_SHARED_EXPERTS_STREAM=1`. The CPP alternative was also rejected
after an invalid-argument failure during graph capture.

### Dynamic lifecycle and routing

| Check | Observed result |
|---|---|
| First dynamic load | HTTP 200, 27.478 s in the full qualification; 27.516 s on the exact release filesystem |
| Base request | HTTP 200 under DCP4/MTP-3/graphs |
| Adapted request | HTTP 200; 32/32 shared token log-probabilities changed |
| Exact release-image max log-probability delta | 0.5778586865 |
| Mixed concurrency-2 base + adapter | both HTTP 200, 2.155 s for 64-token requests |
| Unload | HTTP 200, 0.002 s |
| Base after unload | text and all token log-probabilities bit-for-bit equal to pre-unload base |
| Warm reload | HTTP 200, 7.974 s; adapted request passed |

This isolation check is important: adapter-aware prefix/graph metadata must not let base requests
reuse adapted KV or routed-expert state. Exact base equality before and after unload is the
observable gate.

### Prefix cache, MTP, capacity, and performance

| Workload | Result |
|---|---|
| 3,265-token base prefix, cold → repeat | 7.503 → 0.649 s (**11.57×**) |
| 3,265-token adapted prefix, cold → repeat | 3.906 → 0.868 s (**4.50×**) |
| MTP totals after qualification | 613 drafts, 1,839 draft tokens, 1,599 accepted (**86.95%**) |
| Warm 128-token base decode, 3 runs | 1.473 / 1.540 / 1.538 s; **84.36 tok/s** aggregate |
| Warm 128-token adapter decode, 3 runs | 2.037 / 2.050 / 2.031 s; **62.76 tok/s** aggregate |
| Mixed base+adapter concurrency 2 | 256 tokens / 2.617 s; **97.83 aggregate tok/s** |
| Adapted long context | 5,332 prompt + 128 completion tokens in 4.716 s |
| Adapted near-cap request | 30,553 prompt tokens (93.24% of cap), completed in 18.795 s |
| GPU footprint with adapter resident | 93,215–93,255 MiB used; 4,034–4,074 MiB free per GPU |

The prefix-cache total at the end of all cold stress cases was 10,752 hits / 62,115 queried
tokens; dedicated identical-prefix pairs above are the meaningful cache-speed gates. The
near-cap response content was not used as a quality score—only successful memory-safe execution
was asserted.

### Deterministic quality gates

Chat requests used `chat_template_kwargs={"enable_thinking": false}`, temperature 0, and the
same seed. Both base and adapter passed:

| Case | Required result | Base | Adapter |
|---|---|---|---|
| Factual | `Tokyo` | PASS | PASS |
| Arithmetic | `323` | PASS | PASS |
| Python expression | valid even-number sum expression | PASS | PASS |
| Exact-format instruction | `red green blue` | PASS | PASS |

The final shipped compose preset was then recreated from the published digest and its existing
API harness passed for **both** the base model and the dynamically registered adapter: health,
exact greedy chat, four non-streaming tool-call scenarios, and streaming tool-call deltas all
reported `ALL PASS`.

The updated retrieval harness also covered the shipped context range at three insertion depths:

| Prompt target | Depths | Base | Adapter | Observed per-request range |
|---:|---|---:|---:|---|
| 8,000 | 0.1 / 0.5 / 0.9 | 3/3 | 3/3 | base 2–10 s; adapter 3–6 s |
| 16,000 | 0.1 / 0.5 / 0.9 | 3/3 | 3/3 | base 4–7 s; adapter 5–9 s |
| 30,000 | 0.1 / 0.5 / 0.9 | 3/3 | 3/3 | base 8–13 s; adapter 10–17 s |
| **Total** | | **9/9** | **9/9** | all replies were the exact planted code |

This is a regression gate, not evidence that the adapter universally improves quality. The
adapter had measurable decode overhead in this workload, and the release makes no universal
throughput claim.

### Known log noise and rollback

The base image repeatedly probes an optional FA2 extension whose ABI does not match its Torch
build and logs an undefined-symbol error. The active backend is B12X sparse MLA; model load,
graph capture, DCP4, prefix cache, MTP, and dynamic LoRA all served after those messages. First
use of unseen long-prefill/LoRA shapes also emitted JIT-monitor latency warnings; the persistent
compile cache prevents repeat compilation.

The qualification host retained the stopped prior-production container
`glm52-exl3-v26-5001` (`e08c3601feed…`) and local image
`sha256:d55205e3ae3d81f00a2770dee91c2bf1662a5efe29c6c897be5ac3010ca75895`.
Rollback is:

```bash
cd deploy
./server.sh stop
docker start glm52-exl3-v26-5001
```

The retained container must remain in place until the release burn-in is complete.

---

## 2026-07-25 update (3) — per-model runtime scoping + per-ordinal arch cache key (v26)

Follow-up to the v20 rebase, from CodeRabbit review on
[local-inference-lab/vllm#139](https://github.com/local-inference-lab/vllm/pull/139).

**Bug:** the EXL3 rank-sliced runtime cache was keyed only on device, dtype, shape,
topk and planner settings. The cached entry owns mutable Trellis/prefill scratch and
parity staging buffers. A target MoE layer and the rank-sliced MTP-78 draft layer
match on *every* one of those components -- same hidden/intermediate size, same local
expert count, same topk, same planner env, and both resolve `max_num_batched_tokens`
from the same scheduler config -- so the draft was reusing the target's scratch. That
defeats the target/draft isolation their independently captured CUDA graphs rely on.

**Fix:** the cache key is now scoped to the owning quant config, so each model gets
exactly one runtime. This is deliberately coarser than per-layer: the prefill arena is
~1054 MiB, so per-layer runtimes would need tens of GiB per rank across 75+ layers.

**Runtime:** `verdictai/glm52-exl3-sparkinfer:v31-gg-v20-sic3828fd-vllm0c79e41-cu132-sm120a` (`sha256:8753406f…`).

**Measured effect (the fix is observable in memory, not in the log -- the planner line
uses `info_once` and is deduplicated):**

| | Rank-sliced runtimes | GPU KV cache | Concurrency @524K |
| --- | --- | --- | --- |
| pre-fix (shared scratch) | 1 (target+draft collide) | 1,115,904 tokens | 2.13x |
| **v26 (scoped)** | **2 (isolated)** | 998,656 tokens | 1.90x |

The 119,552-token KV reduction corresponds to ~1056 MiB, matching the 1054.2 MiB arena
-- direct evidence that a second runtime is allocated. Lower KV capacity is the
intended cost of the isolation.

**Second fix in this runtime (v26).** CodeRabbit correctly rejected a first attempt at
the compile-cache device key: `torch.cuda.get_device_capability()` /
`get_device_name()` already resolve against `torch.cuda.current_device()`, so passing
that ordinal explicitly was a no-op and left the process-wide key free to freeze
whichever GPU was current on the first call. The real fix memoizes the architecture key
**per device ordinal** and threads the ordinal through `_static_compile_cache_context`,
which is `lru_cache`d on the compile callable and would otherwise have re-frozen the
identity at that layer. The returned key still omits the ordinal, so GPUs of the same
architecture keep sharing compiled artifacts. This matters on this rig specifically:
its four boards report two different device names (Max-Q and non-Max-Q) while sharing
compute capability 12.0.

**Quality on v26:**

| Suite | Config | Result |
| --- | --- | --- |
| Estonia | c2, 5 runs | 5/5 pass, 0 fail, correct rate 1.00 |
| LAVD | c5, 5 runs | 3 EXACT / 2 NEAR / 0 FAIL, correct rate 1.00 |

For continuity: the pre-fix runtime measured LAVD 2E/3N/0F. An intermediate scoped
build measured 2E/2N/1F on one 5-run sample and 2E/3N/0F on a second; that single
failure was a wrong ledger total at 7,114 completion tokens against a 24,576 cap, i.e.
an answer-quality miss rather than truncation, and within this profile's run-to-run
spread. This runtime shows no failures on either suite.

---

## 2026-07-25 update (2) — rebased onto the FINAL Gilded Gnosis v20 base

The runtime is rebased onto the **finalized v20 common base**
`voipmonitor/vllm:gilded-gnosis-v20-vllm0c79e41-sic3828fd-fi801d57a-cu132-20260727`
(vLLM `5517197`, Sparkinfer `be0edca`, FlashInfer `801d57a`, CUTLASS 4.6.0).

**New runtime:** `verdictai/glm52-exl3-sparkinfer:v31-gg-v20-sic3828fd-vllm0c79e41-cu132-sm120a`
(`sha256:da185fe8…`).

**Why:** the finalized base consolidates the DCP prefill **auto-policy** and the
corrected **workspace accounting**, resolving the >8k DCP prefill collapse present
in earlier v20 candidates, plus long-context MTP alignment and deterministic
dynamic-MoE output. EXL3 is enabled by rebasing the EXL3 *source layer* onto that
pinned stack (the base ships no EXL3/Trellis loader), keeping EXL3 quantization
separate while sharing the corrected runtime. Of the 12 runtime overlay files only
`models/deepseek_v2.py` and `v1/attention/backends/mla/indexer.py` differ in the new
base; both were re-derived from the v20-final versions with the EXL3 edits replayed
on top, so v20's DCP/indexer work is preserved rather than overwritten by the overlay.

**Config change:** `server.sh` / `docker-compose.yml` now set the DCP policy that the
base launcher resolves from `DCP_*=auto` for TP4/DCP4 — query split 1, full-CKV gather 1,
top-k owner merge 1, indexer shards 0, CKV prefetch depth 1, prefetch workspace 1024 MiB,
and `DCP_PREFILL_WORKSPACE=1` (`VLLM_DCP_PROJECT_BEFORE_MERGE=1` +
`VLLM_B12X_MLA_DCP_GATHER_IN_WORKSPACE=1`). These are set explicitly because the preset
calls `vllm serve` directly and bypasses `/usr/local/bin/serve-gilded-gnosis.sh`.
Note `VLLM_DCP_QUERY_SPLIT` moves from `0` to `1` versus the previous preset.

**Boot assertions observed:** engine
`v0.11.2.dev280+gilded.gnosis.v20.vllm5517197.sibe0edca.fi801d57a.cu132.20260725`,
`vLLM is using nccl==2.30.4`, `EXL3 rank-sliced runtime planned: Trellis m=1..32
block_m=8, prefill trellis block_m=64 arena=1054.2MiB capacity=3072 chunk=128 topk=8`,
`Preallocated 30.8 MiB for 2 persistent CKV execution lane(s)`, `Using native CKV layer
prefetch with depth=1 and 2 workspace slots`, GPU KV cache **1,115,904 tokens** (2.13x at
524,288). The base's `InstantTensor loader` line does not appear on this path because the
EXL3 checkpoint loads through the EXL3 rank-sliced loader.

**Regression (no degradation):**

| Suite | Config | Result |
| --- | --- | --- |
| Estonia | c2, 5 runs | **5/5 pass**, 0 fail, correct rate 1.00, 70.0 tok/s aggregate |
| LAVD | c5, 5 runs | **2 EXACT / 3 NEAR / 0 FAIL**, correct rate 1.00, 55.6 tok/s aggregate |

All sections below were measured on the previous (`v21-mtp78tr3`) image and remain
valid for the checkpoint itself; only the runtime base changed.

---

## 2026-07-25 update — MTP layer-78 is now EXL3 tr3

This checkpoint now ships the **MTP (layer 78) routed experts in EXL3 Trellis tr3
(3.0 bpw)**, matching layers 3-77; the previous BF16 MTP head is retired. The
layer-78 file drops from 19.9 GB to 4.24 GB (−15.66 GB), freeing ~3.9 GiB/rank.

**Requires the updated runtime:**
`verdictai/glm52-exl3-sparkinfer:v31-gg-v20-sic3828fd-vllm0c79e41-cu132-sm120a`
(`sha256:9b1befc1…`) **plus `VLLM_EXL3_TRELLIS_MIN_M=1`** (the compose / server.sh
default in this repo). The prior v20 image cannot load a tr3 MTP layer. The two
loader fixes are in vLLM PR #139 (local-inference-lab/vllm#139); no Sparkinfer
change was needed (validated against #49 / `si1a88b38`).

Re-measured on the same 4× RTX PRO 6000 (TP4/DCP4, MTP-3, util 0.96,
`VLLM_EXL3_TRELLIS_MIN_M=1`, auto-profiled KV):

| Metric | BF16-MTP build (Sections below) | tr3-MTP build (this) |
| --- | --- | --- |
| GPU KV cache @ 0.96 util | ~680K tok (~1.3× @ 524K) | **1,132,544 tok (2.16× @ 524K)** |
| Prefill 8k / 64k / 128k (tok/s) | 2,551 / — / 1,833 | 2,521 / 1,916 / 1,765 |
| Decode C1 / C4 / C8 (tok/s) | 87.5 / 219.3 / 308.1 | 89.7 / 225.3 / 293.5 |
| Estonia (long-ctx retrieval) | PASS 30/30 | **PASS 10/10** |
| LAVD (ledger consistency) | 18E / 11N / 1F | **EXACT 5 / NEAR 5 / FAIL 0** |

Decode is ~neutral (MTP is lossless); the real gain is **~+66% KV-cache /
concurrency headroom** from the freed VRAM. Accuracy is unchanged — the detailed
Sections 1-8 below were measured on the prior BF16-MTP build and remain
representative for quality; only the image, the layer-78 format, and the
KV/serving preset changed.

---

## What this quantization costs

| | |
| --- | --- |
| BF16 full precision | 1,506 GB |
| EXL3 TR3 3.0bpw | **316.5 GB** (tr3-MTP build; was 332.2 GB with the BF16 MTP head) |
| Size vs BF16 | **21.0%** (a 79.0% reduction) |
| Effective whole-model rate | ~3.45 bpw (routed experts, **now including MTP layer-78**, are a flat 3.0; attention, shared experts, embeddings, LM head, and the non-expert parts of layer 78 stay BF16) |

BF16 weights alone would need 16x 96 GB cards. This fits on 4 with room for the KV cache.

## Image

```text
verdictai/glm52-exl3-sparkinfer:v31-gg-v20-sic3828fd-vllm0c79e41-cu132-sm120a@sha256:0433ae94665b769b78dd301f952d907508a3ba80bce47a1630ec20ade8812dff
```

This is the current runtime. The per-benchmark sections further down were
measured on the earlier `v21-mtp78tr3` image and are retained as the
checkpoint-level record; see the dated update sections above for what changed
in the runtime since, and for the re-verification run on this image.

## Serving preset (as shipped)

| Setting | Value |
| --- | --- |
| `GPU_MEMORY_UTILIZATION` | 0.96 |
| `MAX_MODEL_LEN` | 524288 (tr3-MTP build; was 262144) |
| `NUM_GPU_BLOCKS_OVERRIDE` | empty → auto-profile (~1,132,544 KV tokens @ 0.96; was pinned 1024) |
| `VLLM_EXL3_TRELLIS_MIN_M` | **1** (required for the tr3 MTP draft's small-m GEMMs; was 4) |
| `MAX_NUM_BATCHED_TOKENS` | 3072 |
| `MAX_NUM_SEQS` | 8 |
| MTP | enabled, 3 tokens, greedy draft |
| `ENABLE_ASYNC_SCHEDULING` | 0 (correctness guard) |
| `VLLM_B12X_MLA_SPEC_EXTEND_AS_DECODE` | 0 (lossless setting) |
| Attention / MoE | `B12X_MLA_SPARSE` / `b12x` |
| Quantization | `exl3` |
| KV cache | `nvfp4_ds_mla` (shipped default) and `fp8` (comparison arm) |

Only `--kv-cache-dtype` and (for section 3) `--default-chat-template-kwargs` were parameterized;
every other flag is byte-identical to the published compose file.

---

# Summary

| Test | Runs | nvfp4_ds_mla | fp8 |
| --- | ---: | --- | --- |
| Estonia (needle retrieval, 133K ctx) | 30 @ c2 | **PASS 30 / FAIL 0** | **PASS 30 / FAIL 0** |
| LAVD (ledger consistency) | 30 @ c5 | 18 EXACT / 11 NEAR / 1 FAIL (97%) | 15 EXACT / 13 NEAR / 2 FAIL (93%) |
| Hotel-lights, low tier | 30 @ c5 | 15 EXACT / 15 FAIL (50%) | 18 EXACT / 12 FAIL (60%) |
| Hotel-lights, Max tier | 30 @ c5 | **20 EXACT / 10 FAIL (67%)** | **20 EXACT / 10 FAIL (67%)** |
| KLD vs BF16 | 5 | 0.116138 | **0.101198** |
| Decode C1 / C8 (tok/s) | — | 87.5 / 308.1 | 86.6 / 312.2 |
| Prefill 8k / 128k (tok/s) | — | 2,551 / 1,833 | 2,660 / 1,771 |

---

# 1. Estonia — long-context needle retrieval

133,186-token prompt. **30 runs**, concurrency 2, temperature 0, repetition penalty 1.25,
`max_tokens` 40000, regex-scored on the final answer line.

| KV cache | score | completed | hit max_tokens | tok p50 | avg latency | TTFT | gen tok/s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| nvfp4_ds_mla | **PASS 30 / FAIL 0** | 30/30 | 0 | 2,250 | 38.0 s | 0.61 s | 67.9 |
| fp8 | **PASS 30 / FAIL 0** | 30/30 | 0 | 2,370 | 51.8 s | 0.60 s | 71.2 |

100% on both KV formats, no run near the token cap. The repetition penalty matters here: without
it the model loops on retrieval phrasing and exhausts the output budget without answering.

<details><summary>Estonia — nvfp4_ds_mla log</summary>

```text
╭─────────────────────────────── Configuration ────────────────────────────────╮
│ Completion Token Statistics Benchmark                                        │
│ Model: GLM-5.2-EXL3-TR3-3.0bpw @ localhost:8000                              │
│ Prompt: profile:estonia                                                      │
│ Concurrency: 2                                                               │
│ Measured runs: 30 | Max tokens: 40000                                        │
│ Scoring: \bestonia\b                                                         │
╰──────────────────────────────────────────────────────────────────────────────╯

╭────────────────────────────── Completion Stats ──────────────────────────────╮
│ Completion Token Statistics                                                  │
│ One optional prefix-cache scout request is used to populate prefill first.   │
│ Built-in profile run at fixed concurrency C=2.                               │
╰──────────────────────────────────────────────────────────────────────────────╯
Profile                                                       
╭────────────────┬───────────────────────────────────────────╮
│ field          │ value                                     │
├────────────────┼───────────────────────────────────────────┤
│ profile        │ estonia                                   │
│ prompt         │ profile:estonia                           │
│ prompt chars   │ 707,372                                   │
│ requested runs │ 30                                        │
│ concurrency    │ 2                                         │
│ max tokens     │ 40000                                     │
│ scoring        │ regex                                     │
│ prefill scout  │ 133,186 prompt tok / 79.72s = 1,671 tok/s │
│ correct regex  │ \bestonia\b                               │
╰────────────────┴───────────────────────────────────────────╯
╭───────────────────────── Whole-run GPU Power ──────────────────────────╮
│ avg 1,105 W | max 1,124 W | limit 1,200 W | over 10m 56s | 273 samples │
╰────────────────────────────────────────────────────────────────────────╯
Concurrency Results                                                             
╭────┬────────┬───────────┬──────┬─────────┬──────────┬──────────┬─────────┬───╮
│ p… │ done/… │     score │ sta… │ output… │ output … │ aggrega… │ avg re… │ … │
├────┼────────┼───────────┼──────┼─────────┼──────────┼──────────┼─────────┼───┤
│  2 │  30/30 │ PASS 30 … │ ★★★… │   2,250 │    3,687 │     67.9 │    38.0 │ … │
╰────┴────────┴───────────┴──────┴─────────┴──────────┴──────────┴─────────┴───╯
Selected C=2                                     
╭────────────────────────────┬──────────────────╮
│ metric                     │            value │
├────────────────────────────┼──────────────────┤
│ completed                  │            30/30 │
│ score                      │ PASS 30 / FAIL 0 │
│ stars                      │    ★★★★★★★★★★ 👍 │
│ hit max_tokens             │                0 │
│ completion tokens avg      │            2,536 │
│ completion tokens p50      │            2,250 │
│ completion tokens p90      │            3,687 │
│ completion tokens p99      │            4,725 │
│ elapsed avg                │            38.0s │
│ TTFT avg                   │            0.61s │
│ aggregate gen tok/s        │             67.9 │
│ mean per-request gen tok/s │             67.9 │
╰────────────────────────────┴──────────────────╯
Interpretation: completion-token p50/p90/p99 tells how many decode tokens the 
model needed to reach its final answer under this engine/config. Correctness is 
scored from the final non-empty answer line by default, matching the GLM 
dense-MLA vs NSA benchmark methodology. The prefill scout is not a scored 
answer; it is the max_tokens=1 prefix-cache warmup and its prompt_tokens/TTFT 
value is reported as scout prefill speed. Concurrency Results groups completed 
requests by parallelism; Completed Requests shows the latest individual finished
answers.

Results saved to 
/tmp/claude-1000/-home-brandonmusic-KLC-SANDBOXES/50980f6d-56ae-4115-a6bf-0d1737
7be8eb/scratchpad/testsuite/results2/estonia-nvfp4_ds_mla-c2-r30-rp125.json
```
</details>

<details><summary>Estonia — fp8 log</summary>

```text
╭─────────────────────────────── Configuration ────────────────────────────────╮
│ Completion Token Statistics Benchmark                                        │
│ Model: GLM-5.2-EXL3-TR3-3.0bpw @ localhost:8000                              │
│ Prompt: profile:estonia                                                      │
│ Concurrency: 2                                                               │
│ Measured runs: 30 | Max tokens: 40000                                        │
│ Scoring: \bestonia\b                                                         │
╰──────────────────────────────────────────────────────────────────────────────╯

╭────────────────────────────── Completion Stats ──────────────────────────────╮
│ Completion Token Statistics                                                  │
│ One optional prefix-cache scout request is used to populate prefill first.   │
│ Built-in profile run at fixed concurrency C=2.                               │
╰──────────────────────────────────────────────────────────────────────────────╯
Profile                                                       
╭────────────────┬───────────────────────────────────────────╮
│ field          │ value                                     │
├────────────────┼───────────────────────────────────────────┤
│ profile        │ estonia                                   │
│ prompt         │ profile:estonia                           │
│ prompt chars   │ 707,372                                   │
│ requested runs │ 30                                        │
│ concurrency    │ 2                                         │
│ max tokens     │ 40000                                     │
│ scoring        │ regex                                     │
│ prefill scout  │ 133,186 prompt tok / 78.91s = 1,688 tok/s │
│ correct regex  │ \bestonia\b                               │
╰────────────────┴───────────────────────────────────────────╯
╭───────────────────────── Whole-run GPU Power ──────────────────────────╮
│ avg 1,101 W | max 1,127 W | limit 1,200 W | over 14m 28s | 361 samples │
╰────────────────────────────────────────────────────────────────────────╯
Concurrency Results                                                             
╭────┬────────┬───────────┬──────┬─────────┬──────────┬──────────┬─────────┬───╮
│ p… │ done/… │     score │ sta… │ output… │ output … │ aggrega… │ avg re… │ … │
├────┼────────┼───────────┼──────┼─────────┼──────────┼──────────┼─────────┼───┤
│  2 │  30/30 │ PASS 30 … │ ★★★… │   2,370 │    4,513 │     71.2 │    51.8 │ … │
╰────┴────────┴───────────┴──────┴─────────┴──────────┴──────────┴─────────┴───╯
Selected C=2                                     
╭────────────────────────────┬──────────────────╮
│ metric                     │            value │
├────────────────────────────┼──────────────────┤
│ completed                  │            30/30 │
│ score                      │ PASS 30 / FAIL 0 │
│ stars                      │    ★★★★★★★★★★ 👍 │
│ hit max_tokens             │                0 │
│ completion tokens avg      │            3,644 │
│ completion tokens p50      │            2,370 │
│ completion tokens p90      │            4,513 │
│ completion tokens p99      │           26,158 │
│ elapsed avg                │            51.8s │
│ TTFT avg                   │            0.60s │
│ aggregate gen tok/s        │             71.2 │
│ mean per-request gen tok/s │             67.3 │
╰────────────────────────────┴──────────────────╯
Interpretation: completion-token p50/p90/p99 tells how many decode tokens the 
model needed to reach its final answer under this engine/config. Correctness is 
scored from the final non-empty answer line by default, matching the GLM 
dense-MLA vs NSA benchmark methodology. The prefill scout is not a scored 
answer; it is the max_tokens=1 prefix-cache warmup and its prompt_tokens/TTFT 
value is reported as scout prefill speed. Concurrency Results groups completed 
requests by parallelism; Completed Requests shows the latest individual finished
answers.

Results saved to 
/tmp/claude-1000/-home-brandonmusic-KLC-SANDBOXES/50980f6d-56ae-4115-a6bf-0d1737
7be8eb/scratchpad/testsuite/results2/estonia-fp8-c2-r30-rp125.json
```
</details>

# 2. LAVD — long-context ledger consistency

48,302-char structured ledger; the model must find human data-entry errors, apply the repair rule,
and return ticket count and hours. Ground truth **72, 46.0**. **30 runs**, concurrency 5,
temperature 0, repetition penalty 1.15, `max_tokens` 24576.

| KV cache | EXACT | NEAR | FAIL | pass (E+N) | hit max_tokens | tok p50 | avg latency | gen tok/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| nvfp4_ds_mla | 18 | 11 | 1 | **29/30 (97%)** | 0 | 9,206 | 180.6 s | 53.3 |
| fp8 | 15 | 13 | 2 | **28/30 (93%)** | 0 | 8,908 | 190.4 s | 52.2 |

All three failures are single-axis near-misses, not parse errors or truncation:
`65, 42.5` (count -7) · `72, 41.75` (count exact, hours -4.25) · `66, 45.25` (count -6).
Each used fewer tokens than the ~9K median, i.e. they stopped searching early rather than looping.

<details><summary>LAVD — nvfp4_ds_mla log</summary>

```text
╭─────────────────────────────── Configuration ────────────────────────────────╮
│ LAVD Context Consistency Test                                                │
│ Model: GLM-5.2-EXL3-TR3-3.0bpw @ localhost:8000                              │
│ Prompt: profile:lavd-test                                                    │
│ Concurrency: 5                                                               │
│ Measured runs: 30 | Max tokens: 24576                                        │
│ Scoring: EXACT / NEAR / FAIL numeric pair                                    │
╰──────────────────────────────────────────────────────────────────────────────╯

╭────────────────────────────── Completion Stats ──────────────────────────────╮
│ LAVD Context Consistency Test                                                │
│ Arithmetic is intentionally simple; the test checks whether the model keeps  │
│ a long structured context consistent, finds human data-entry errors, applies │
│ the repair rule, and returns the final ticket count and hours. Built-in      │
│ profile run at fixed concurrency C=5.                                        │
╰──────────────────────────────────────────────────────────────────────────────╯
Profile                               
╭────────────────┬───────────────────╮
│ field          │ value             │
├────────────────┼───────────────────┤
│ profile        │ lavd-test         │
│ prompt         │ profile:lavd-test │
│ prompt chars   │ 48,302            │
│ requested runs │ 30                │
│ concurrency    │ 5                 │
│ max tokens     │ 24576             │
│ scoring        │ ledger_lavd       │
│ expected       │ 72, 46            │
│ prompt sha256  │ 5c83674d5f0fd2a7  │
│ dataset sha256 │ 612f8041bbca048c  │
╰────────────────┴───────────────────╯
╭───────────────────────── Whole-run GPU Power ──────────────────────────╮
│ avg 1,130 W | max 1,143 W | limit 1,200 W | over 19m 20s | 483 samples │
╰────────────────────────────────────────────────────────────────────────╯
Concurrency Results                                                             
╭──┬──────┬─────────────────────┬────┬─────────┬────────┬──────────┬───────┬───╮
│  │ don… │               score │ s… │ output… │ outpu… │ aggrega… │ avg … │ … │
├──┼──────┼─────────────────────┼────┼─────────┼────────┼──────────┼───────┼───┤
│  │ 30/… │ EXACT 18 / NEAR 11… │ ★… │   9,206 │ 11,799 │     53.3 │ 180.6 │ … │
╰──┴──────┴─────────────────────┴────┴─────────┴────────┴──────────┴───────┴───╯
Selected C=5                                                
╭────────────────────────────┬─────────────────────────────╮
│ metric                     │                       value │
├────────────────────────────┼─────────────────────────────┤
│ completed                  │                       30/30 │
│ score                      │ EXACT 18 / NEAR 11 / FAIL 1 │
│ stars                      │               ★★★★★★☆☆☆☆ 👍 │
│ hit max_tokens             │                           0 │
│ completion tokens avg      │                       9,517 │
│ completion tokens p50      │                       9,206 │
│ completion tokens p90      │                      11,799 │
│ completion tokens p99      │                      14,634 │
│ elapsed avg                │                      180.6s │
│ TTFT avg                   │                       1.99s │
│ aggregate gen tok/s        │                        53.3 │
│ mean per-request gen tok/s │                        53.3 │
╰────────────────────────────┴─────────────────────────────╯
Failed Final Answers                                                     
╭────┬───┬────────┬───────┬─────────────────────────────────────────────╮
│  # │ C │ tokens │ score │ final answer                                │
├────┼───┼────────┼───────┼─────────────────────────────────────────────┤
│ 14 │ 5 │  5,721 │ FAIL  │ 65, 42.5 (count -7, hours -3.50) | 65, 42.5 │
╰────┴───┴────────┴───────┴─────────────────────────────────────────────╯
Interpretation: EXACT means the parsed final numeric pair is exactly 72, 46.0. 
NEAR means both count and hours are within the configured tolerance; FAIL means 
the answer was unparseable or outside tolerance. The 10-slot quality bar is a 
rounded distribution: ★=EXACT, ☆=NEAR, ✕=FAIL.

Results saved to 
/tmp/claude-1000/-home-brandonmusic-KLC-SANDBOXES/50980f6d-56ae-4115-a6bf-0d1737
7be8eb/scratchpad/testsuite/results2/lavd-nvfp4_ds_mla-c5-r30-rp115.json
```
</details>

<details><summary>LAVD — fp8 log</summary>

```text
╭─────────────────────────────── Configuration ────────────────────────────────╮
│ LAVD Context Consistency Test                                                │
│ Model: GLM-5.2-EXL3-TR3-3.0bpw @ localhost:8000                              │
│ Prompt: profile:lavd-test                                                    │
│ Concurrency: 5                                                               │
│ Measured runs: 30 | Max tokens: 24576                                        │
│ Scoring: EXACT / NEAR / FAIL numeric pair                                    │
╰──────────────────────────────────────────────────────────────────────────────╯

╭────────────────────────────── Completion Stats ──────────────────────────────╮
│ LAVD Context Consistency Test                                                │
│ Arithmetic is intentionally simple; the test checks whether the model keeps  │
│ a long structured context consistent, finds human data-entry errors, applies │
│ the repair rule, and returns the final ticket count and hours. Built-in      │
│ profile run at fixed concurrency C=5.                                        │
╰──────────────────────────────────────────────────────────────────────────────╯
Profile                               
╭────────────────┬───────────────────╮
│ field          │ value             │
├────────────────┼───────────────────┤
│ profile        │ lavd-test         │
│ prompt         │ profile:lavd-test │
│ prompt chars   │ 48,302            │
│ requested runs │ 30                │
│ concurrency    │ 5                 │
│ max tokens     │ 24576             │
│ scoring        │ ledger_lavd       │
│ expected       │ 72, 46            │
│ prompt sha256  │ 5c83674d5f0fd2a7  │
│ dataset sha256 │ 612f8041bbca048c  │
╰────────────────┴───────────────────╯
╭───────────────────────── Whole-run GPU Power ──────────────────────────╮
│ avg 1,127 W | max 1,138 W | limit 1,200 W | over 19m 45s | 493 samples │
╰────────────────────────────────────────────────────────────────────────╯
Concurrency Results                                                             
╭──┬──────┬─────────────────────┬────┬─────────┬────────┬──────────┬───────┬───╮
│  │ don… │               score │ s… │ output… │ outpu… │ aggrega… │ avg … │ … │
├──┼──────┼─────────────────────┼────┼─────────┼────────┼──────────┼───────┼───┤
│  │ 30/… │ EXACT 15 / NEAR 13… │ ★… │   8,908 │ 13,173 │     52.2 │ 190.4 │ … │
╰──┴──────┴─────────────────────┴────┴─────────┴────────┴──────────┴───────┴───╯
Selected C=5                                                
╭────────────────────────────┬─────────────────────────────╮
│ metric                     │                       value │
├────────────────────────────┼─────────────────────────────┤
│ completed                  │                       30/30 │
│ score                      │ EXACT 15 / NEAR 13 / FAIL 2 │
│ stars                      │               ★★★★★☆☆☆☆✕ 👍 │
│ hit max_tokens             │                           0 │
│ completion tokens avg      │                       9,831 │
│ completion tokens p50      │                       8,908 │
│ completion tokens p90      │                      13,173 │
│ completion tokens p99      │                      14,658 │
│ elapsed avg                │                      190.4s │
│ TTFT avg                   │                       1.95s │
│ aggregate gen tok/s        │                        52.2 │
│ mean per-request gen tok/s │                        52.3 │
╰────────────────────────────┴─────────────────────────────╯
Failed Final Answers                                                       
╭────┬───┬────────┬───────┬───────────────────────────────────────────────╮
│  # │ C │ tokens │ score │ final answer                                  │
├────┼───┼────────┼───────┼───────────────────────────────────────────────┤
│ 13 │ 5 │  9,982 │ FAIL  │ 72, 41.75 (count +0, hours -4.25) | 72, 41.75 │
│ 23 │ 5 │  7,131 │ FAIL  │ 66, 45.25 (count -6, hours -0.75) | 66, 45.25 │
╰────┴───┴────────┴───────┴───────────────────────────────────────────────╯
Interpretation: EXACT means the parsed final numeric pair is exactly 72, 46.0. 
NEAR means both count and hours are within the configured tolerance; FAIL means 
the answer was unparseable or outside tolerance. The 10-slot quality bar is a 
rounded distribution: ★=EXACT, ☆=NEAR, ✕=FAIL.

Results saved to 
/tmp/claude-1000/-home-brandonmusic-KLC-SANDBOXES/50980f6d-56ae-4115-a6bf-0d1737
7be8eb/scratchpad/testsuite/results2/lavd-fp8-c5-r30-rp115.json
```
</details>

# 3. Hotel-lights — reasoning, low tier vs Max tier

100-room light-cycling puzzle, expected answer 48. **30 runs each**, concurrency 5, temperature 0,
exact numeric scoring. Run at both reasoning tiers.

| Tier | KV cache | EXACT | FAIL | pass | hit max_tokens | avg completion tok | wall |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| low | nvfp4_ds_mla | 15 | 15 | 50% | 2 | 18,903 | 39 min |
| low | fp8 | 18 | 12 | 60% | 3 | 20,806 | 44 min |
| **Max** | nvfp4_ds_mla | **20** | 10 | **67%** | 4 | 37,389 | 77 min |
| **Max** | fp8 | **20** | 10 | **67%** | 6 | 38,346 | 81 min |

Max tier buys +7 to +17 points for **2x the reasoning tokens and 2x the wall time**. Both KV
formats converge to the same 67% at Max. Some runs still exhaust even a 60,000-token budget, so
this task can absorb unbounded reasoning.

**Reasoning-tier gotcha, worth knowing before reproducing.** This model's `chat_template.jinja`
line 2 reads:

```jinja
{%- set effective_reasoning_effort = 'high' if reasoning_effort is defined and reasoning_effort == 'high' else 'max' -%}
```

Only the literal string `high` selects the lower tier. **Anything else — including `max`, or
omitting the field — selects Max.** The shipped compose sends `--default-chat-template-kwargs '{"reasoning_effort":"high"}'`, so the default
preset runs the *lower* tier. The Max rows above were produced by setting that server default to
`max`; each run logged the resolved value from the live container to confirm the tier applied.

<details><summary>Hotel-lights low tier — nvfp4_ds_mla log</summary>

```text
╭─────────────────────────────── Configuration ────────────────────────────────╮
│ Hotel Lights Reasoning Test                                                  │
│ Model: GLM-5.2-EXL3-TR3-3.0bpw @ localhost:8000                              │
│ Prompt: profile:hotel-lights                                                 │
│ Concurrency: 5                                                               │
│ Measured runs: 30 | Max tokens: 40000                                        │
│ Scoring: EXACT / FAIL final number                                           │
╰──────────────────────────────────────────────────────────────────────────────╯

╭────────────────────────────── Completion Stats ──────────────────────────────╮
│ Hotel Lights Reasoning Test                                                  │
│ A compact reasoning profile with a known numeric answer. It checks whether   │
│ the model handles repeated toggles plus the cat reset rule and returns 48.   │
│ Built-in profile run at fixed concurrency C=5.                               │
╰──────────────────────────────────────────────────────────────────────────────╯
Profile                                                
╭────────────────┬────────────────────────────────────╮
│ field          │ value                              │
├────────────────┼────────────────────────────────────┤
│ profile        │ hotel-lights                       │
│ prompt         │ profile:hotel-lights               │
│ prompt chars   │ 385                                │
│ requested runs │ 30                                 │
│ concurrency    │ 5                                  │
│ max tokens     │ 40000                              │
│ scoring        │ numeric_exact                      │
│ expected       │ 48                                 │
│ prefill scout  │ 102 prompt tok / 0.19s = 526 tok/s │
╰────────────────┴────────────────────────────────────╯
╭───────────────────────── Whole-run GPU Power ──────────────────────────╮
│ avg 1,129 W | max 1,145 W | limit 1,200 W | over 38m 36s | 964 samples │
╰────────────────────────────────────────────────────────────────────────╯
Concurrency Results                                                             
╭──┬──────┬─────────────────────┬────┬─────────┬────────┬──────────┬───────┬───╮
│  │ don… │               score │ s… │ output… │ outpu… │ aggrega… │ avg … │ … │
├──┼──────┼─────────────────────┼────┼─────────┼────────┼──────────┼───────┼───┤
│  │ 30/… │ EXACT 15 / NEAR 0 … │ ★… │  18,048 │ 29,997 │     52.4 │ 361.2 │ … │
╰──┴──────┴─────────────────────┴────┴─────────┴────────┴──────────┴───────┴───╯
Selected C=5                                                
╭────────────────────────────┬─────────────────────────────╮
│ metric                     │                       value │
├────────────────────────────┼─────────────────────────────┤
│ completed                  │                       30/30 │
│ score                      │ EXACT 15 / NEAR 0 / FAIL 15 │
│ stars                      │               ★★★★★✕✕✕✕✕ 👍 │
│ hit max_tokens             │                           2 │
│ completion tokens avg      │                      18,903 │
│ completion tokens p50      │                      18,048 │
│ completion tokens p90      │                      29,997 │
│ completion tokens p99      │                      40,000 │
│ elapsed avg                │                      361.2s │
│ TTFT avg                   │                       0.31s │
│ aggregate gen tok/s        │                        52.4 │
│ mean per-request gen tok/s │                        51.8 │
╰────────────────────────────┴─────────────────────────────╯
Failed Final Answers                                                            
╭────┬───┬────────┬───────┬────────────────────────────────────────────────────╮
│  # │ C │ tokens │ score │ final answer                                       │
├────┼───┼────────┼───────┼────────────────────────────────────────────────────┤
│  4 │ 5 │ 11,559 │ FAIL  │ 49 (expected 48, got 49) | 49                      │
│  5 │ 5 │ 40,000 │ FAIL  │ 31 (expected 48, got 31) | Let's check $k=2 \times │
│    │   │        │       │ 3 \times 5 \times 7 \times 11 \times 13 \times 17  │
│    │   │        │       │ \times 19 \times 23 \times 29 \times 31 \          │
│  7 │ 5 │ 18,848 │ FAIL  │ 52 (expected 48, got 52) | 52                      │
│  8 │ 5 │  9,706 │ FAIL  │ 45 (expected 48, got 45) | 45                      │
│ 13 │ 5 │ 15,266 │ FAIL  │ 47 (expected 48, got 47) | 47                      │
│ 14 │ 5 │ 17,935 │ FAIL  │ 50 (expected 48, got 50) | 50                      │
│ 17 │ 5 │ 25,984 │ FAIL  │ 47 (expected 48, got 47) | 47                      │
│ 18 │ 5 │  6,827 │ FAIL  │ 49 (expected 48, got 49) | 49                      │
│ 20 │ 5 │ 20,794 │ FAIL  │ 49 (expected 48, got 49) | 49                      │
│ 22 │ 5 │ 18,403 │ FAIL  │ 49 (expected 48, got 49) | 49                      │
│ 24 │ 5 │ 11,375 │ FAIL  │ 42 (expected 48, got 42) | 42                      │
│ 26 │ 5 │ 40,000 │ FAIL  │ 46 (expected 48, got 46) | Let me review k=46      │
╰────┴───┴────────┴───────┴────────────────────────────────────────────────────╯
Interpretation: EXACT means the final parsed number matches the expected answer.
FAIL means the answer was unparseable or a different number. The 10-slot quality
bar is a rounded distribution: ★=EXACT, ✕=FAIL.

Results saved to 
/tmp/claude-1000/-home-brandonmusic-KLC-SANDBOXES/50980f6d-56ae-4115-a6bf-0d1737
7be8eb/scratchpad/testsuite/results2/hotel-nvfp4_ds_mla-c5-r30.json
```
</details>

<details><summary>Hotel-lights low tier — fp8 log</summary>

```text
╭─────────────────────────────── Configuration ────────────────────────────────╮
│ Hotel Lights Reasoning Test                                                  │
│ Model: GLM-5.2-EXL3-TR3-3.0bpw @ localhost:8000                              │
│ Prompt: profile:hotel-lights                                                 │
│ Concurrency: 5                                                               │
│ Measured runs: 30 | Max tokens: 40000                                        │
│ Scoring: EXACT / FAIL final number                                           │
╰──────────────────────────────────────────────────────────────────────────────╯

╭────────────────────────────── Completion Stats ──────────────────────────────╮
│ Hotel Lights Reasoning Test                                                  │
│ A compact reasoning profile with a known numeric answer. It checks whether   │
│ the model handles repeated toggles plus the cat reset rule and returns 48.   │
│ Built-in profile run at fixed concurrency C=5.                               │
╰──────────────────────────────────────────────────────────────────────────────╯
Profile                                                
╭────────────────┬────────────────────────────────────╮
│ field          │ value                              │
├────────────────┼────────────────────────────────────┤
│ profile        │ hotel-lights                       │
│ prompt         │ profile:hotel-lights               │
│ prompt chars   │ 385                                │
│ requested runs │ 30                                 │
│ concurrency    │ 5                                  │
│ max tokens     │ 40000                              │
│ scoring        │ numeric_exact                      │
│ expected       │ 48                                 │
│ prefill scout  │ 102 prompt tok / 0.19s = 524 tok/s │
╰────────────────┴────────────────────────────────────╯
╭────────────────────────── Whole-run GPU Power ───────────────────────────╮
│ avg 1,121 W | max 1,136 W | limit 1,200 W | over 43m 45s | 1,092 samples │
╰──────────────────────────────────────────────────────────────────────────╯
Concurrency Results                                                             
╭──┬──────┬─────────────────────┬────┬─────────┬────────┬──────────┬───────┬───╮
│  │ don… │               score │ s… │ output… │ outpu… │ aggrega… │ avg … │ … │
├──┼──────┼─────────────────────┼────┼─────────┼────────┼──────────┼───────┼───┤
│  │ 30/… │ EXACT 18 / NEAR 0 … │ ★… │  20,108 │ 30,229 │     52.0 │ 400.1 │ … │
╰──┴──────┴─────────────────────┴────┴─────────┴────────┴──────────┴───────┴───╯
Selected C=5                                                
╭────────────────────────────┬─────────────────────────────╮
│ metric                     │                       value │
├────────────────────────────┼─────────────────────────────┤
│ completed                  │                       30/30 │
│ score                      │ EXACT 18 / NEAR 0 / FAIL 12 │
│ stars                      │               ★★★★★★✕✕✕✕ 👍 │
│ hit max_tokens             │                           3 │
│ completion tokens avg      │                      20,806 │
│ completion tokens p50      │                      20,108 │
│ completion tokens p90      │                      30,229 │
│ completion tokens p99      │                      40,000 │
│ elapsed avg                │                      400.1s │
│ TTFT avg                   │                       0.31s │
│ aggregate gen tok/s        │                        52.0 │
│ mean per-request gen tok/s │                        51.6 │
╰────────────────────────────┴─────────────────────────────╯
Failed Final Answers                                                            
╭────┬───┬────────┬───────┬────────────────────────────────────────────────────╮
│  # │ C │ tokens │ score │ final answer                                       │
├────┼───┼────────┼───────┼────────────────────────────────────────────────────┤
│  5 │ 5 │  9,772 │ FAIL  │ 49 (expected 48, got 49) | 49                      │
│  6 │ 5 │  8,645 │ FAIL  │ 49 (expected 48, got 49) | 49                      │
│  9 │ 5 │  9,616 │ FAIL  │ 49 (expected 48, got 49) | 49                      │
│ 11 │ 5 │ 40,000 │ FAIL  │ 17801 (expected 48, got 17801) | What about $m=37  │
│    │   │        │       │ \times 37 \times 13 = 17801 >                      │
│ 12 │ 5 │ 23,780 │ FAIL  │ 49 (expected 48, got 49) | 49                      │
│ 14 │ 5 │ 17,984 │ FAIL  │ 46 (expected 48, got 46) | 46                      │
│ 15 │ 5 │ 40,000 │ FAIL  │ 0 (expected 48, got 0) | So $M(70) = 70$. Count =  │
│    │   │        │       │ 0.                                                 │
│ 16 │ 5 │ 14,794 │ FAIL  │ 47 (expected 48, got 47) | 47                      │
│ 19 │ 5 │ 16,012 │ FAIL  │ 47 (expected 48, got 47) | 47                      │
│ 23 │ 5 │ 11,105 │ FAIL  │ 49 (expected 48, got 49) | 49                      │
│ 26 │ 5 │  8,702 │ FAIL  │ 49 (expected 48, got 49) | 49                      │
│ 30 │ 5 │ 40,000 │ FAIL  │ 2 (expected 48, got 2) | So the elements in $D_2$  │
│    │   │        │       │ that are $\ge                                      │
╰────┴───┴────────┴───────┴────────────────────────────────────────────────────╯
Interpretation: EXACT means the final parsed number matches the expected answer.
FAIL means the answer was unparseable or a different number. The 10-slot quality
bar is a rounded distribution: ★=EXACT, ✕=FAIL.

Results saved to 
/tmp/claude-1000/-home-brandonmusic-KLC-SANDBOXES/50980f6d-56ae-4115-a6bf-0d1737
7be8eb/scratchpad/testsuite/results2/hotel-fp8-c5-r30.json
```
</details>

<details><summary>Hotel-lights Max tier — nvfp4_ds_mla log</summary>

```text
╭─────────────────────────────── Configuration ────────────────────────────────╮
│ Hotel Lights Reasoning Test                                                  │
│ Model: GLM-5.2-EXL3-TR3-3.0bpw @ localhost:8000                              │
│ Prompt: profile:hotel-lights                                                 │
│ Concurrency: 5                                                               │
│ Measured runs: 30 | Max tokens: 60000                                        │
│ Scoring: EXACT / FAIL final number                                           │
╰──────────────────────────────────────────────────────────────────────────────╯

╭────────────────────────────── Completion Stats ──────────────────────────────╮
│ Hotel Lights Reasoning Test                                                  │
│ A compact reasoning profile with a known numeric answer. It checks whether   │
│ the model handles repeated toggles plus the cat reset rule and returns 48.   │
│ Built-in profile run at fixed concurrency C=5.                               │
╰──────────────────────────────────────────────────────────────────────────────╯
Profile                                                
╭────────────────┬────────────────────────────────────╮
│ field          │ value                              │
├────────────────┼────────────────────────────────────┤
│ profile        │ hotel-lights                       │
│ prompt         │ profile:hotel-lights               │
│ prompt chars   │ 385                                │
│ requested runs │ 30                                 │
│ concurrency    │ 5                                  │
│ max tokens     │ 60000                              │
│ scoring        │ numeric_exact                      │
│ expected       │ 48                                 │
│ prefill scout  │ 102 prompt tok / 0.20s = 507 tok/s │
╰────────────────┴────────────────────────────────────╯
╭────────────────────────── Whole-run GPU Power ───────────────────────────╮
│ avg 1,130 W | max 1,144 W | limit 1,200 W | over 77m 13s | 1,927 samples │
╰──────────────────────────────────────────────────────────────────────────╯
Concurrency Results                                                             
╭──┬──────┬─────────────────────┬────┬─────────┬────────┬──────────┬───────┬───╮
│  │ don… │               score │ s… │ output… │ outpu… │ aggrega… │ avg … │ … │
├──┼──────┼─────────────────────┼────┼─────────┼────────┼──────────┼───────┼───┤
│  │ 30/… │ EXACT 20 / NEAR 0 … │ ★… │  34,282 │ 60,000 │     51.6 │ 725.4 │ … │
╰──┴──────┴─────────────────────┴────┴─────────┴────────┴──────────┴───────┴───╯
Selected C=5                                                
╭────────────────────────────┬─────────────────────────────╮
│ metric                     │                       value │
├────────────────────────────┼─────────────────────────────┤
│ completed                  │                       30/30 │
│ score                      │ EXACT 20 / NEAR 0 / FAIL 10 │
│ stars                      │               ★★★★★★★✕✕✕ 👍 │
│ hit max_tokens             │                           4 │
│ completion tokens avg      │                      37,389 │
│ completion tokens p50      │                      34,282 │
│ completion tokens p90      │                      60,000 │
│ completion tokens p99      │                      60,000 │
│ elapsed avg                │                      725.4s │
│ TTFT avg                   │                       0.31s │
│ aggregate gen tok/s        │                        51.6 │
│ mean per-request gen tok/s │                        51.2 │
╰────────────────────────────┴─────────────────────────────╯
Failed Final Answers                                                            
╭────┬───┬────────┬───────┬────────────────────────────────────────────────────╮
│  # │ C │ tokens │ score │ final answer                                       │
├────┼───┼────────┼───────┼────────────────────────────────────────────────────┤
│  1 │ 5 │ 24,913 │ FAIL  │ 50 (expected 48, got 50) | 50                      │
│  7 │ 5 │ 60,000 │ FAIL  │ 283 (expected 48, got 283) | Let's test $k' = 2    │
│    │   │        │       │ \cdot 191 \cdot 283                                │
│ 11 │ 5 │ 32,608 │ FAIL  │ 40 (expected 48, got 40) | 40                      │
│ 14 │ 5 │ 60,000 │ FAIL  │ 8 (expected 48, got 8) | Divisors of 32: 1, 2, 4,  │
│    │   │        │       │ 8,                                                 │
│ 16 │ 5 │ 30,373 │ FAIL  │ 52 (expected 48, got 52) | 52                      │
│ 21 │ 5 │ 31,931 │ FAIL  │ 49 (expected 48, got 49) | 49                      │
│ 23 │ 5 │ 36,285 │ FAIL  │ 47 (expected 48, got 47) | 47                      │
│ 25 │ 5 │ 25,730 │ FAIL  │ 32 (expected 48, got 32) | 32                      │
│ 28 │ 5 │ 60,000 │ FAIL  │ 62 (expected 48, got 62) | m=62: 62,               │
│ 29 │ 5 │ 60,000 │ FAIL  │ 83 (expected 48, got 83) | Let's re-verify Room 83 │
╰────┴───┴────────┴───────┴────────────────────────────────────────────────────╯
Interpretation: EXACT means the final parsed number matches the expected answer.
FAIL means the answer was unparseable or a different number. The 10-slot quality
bar is a rounded distribution: ★=EXACT, ✕=FAIL.

Results saved to 
/tmp/claude-1000/-home-brandonmusic-KLC-SANDBOXES/50980f6d-56ae-4115-a6bf-0d1737
7be8eb/scratchpad/testsuite/results2/hotelmax-nvfp4_ds_mla-c5-r30.json
```
</details>

<details><summary>Hotel-lights Max tier — fp8 log</summary>

```text
╭─────────────────────────────── Configuration ────────────────────────────────╮
│ Hotel Lights Reasoning Test                                                  │
│ Model: GLM-5.2-EXL3-TR3-3.0bpw @ localhost:8000                              │
│ Prompt: profile:hotel-lights                                                 │
│ Concurrency: 5                                                               │
│ Measured runs: 30 | Max tokens: 60000                                        │
│ Scoring: EXACT / FAIL final number                                           │
╰──────────────────────────────────────────────────────────────────────────────╯

╭────────────────────────────── Completion Stats ──────────────────────────────╮
│ Hotel Lights Reasoning Test                                                  │
│ A compact reasoning profile with a known numeric answer. It checks whether   │
│ the model handles repeated toggles plus the cat reset rule and returns 48.   │
│ Built-in profile run at fixed concurrency C=5.                               │
╰──────────────────────────────────────────────────────────────────────────────╯
Profile                                                
╭────────────────┬────────────────────────────────────╮
│ field          │ value                              │
├────────────────┼────────────────────────────────────┤
│ profile        │ hotel-lights                       │
│ prompt         │ profile:hotel-lights               │
│ prompt chars   │ 385                                │
│ requested runs │ 30                                 │
│ concurrency    │ 5                                  │
│ max tokens     │ 60000                              │
│ scoring        │ numeric_exact                      │
│ expected       │ 48                                 │
│ prefill scout  │ 102 prompt tok / 0.19s = 525 tok/s │
╰────────────────┴────────────────────────────────────╯
╭────────────────────────── Whole-run GPU Power ───────────────────────────╮
│ avg 1,124 W | max 1,138 W | limit 1,200 W | over 80m 42s | 2,012 samples │
╰──────────────────────────────────────────────────────────────────────────╯
Concurrency Results                                                             
╭──┬──────┬─────────────────────┬────┬─────────┬────────┬──────────┬───────┬───╮
│  │ don… │               score │ s… │ output… │ outpu… │ aggrega… │ avg … │ … │
├──┼──────┼─────────────────────┼────┼─────────┼────────┼──────────┼───────┼───┤
│  │ 30/… │ EXACT 20 / NEAR 0 … │ ★… │  34,446 │ 60,000 │     51.7 │ 742.4 │ … │
╰──┴──────┴─────────────────────┴────┴─────────┴────────┴──────────┴───────┴───╯
Selected C=5                                                
╭────────────────────────────┬─────────────────────────────╮
│ metric                     │                       value │
├────────────────────────────┼─────────────────────────────┤
│ completed                  │                       30/30 │
│ score                      │ EXACT 20 / NEAR 0 / FAIL 10 │
│ stars                      │               ★★★★★★★✕✕✕ 👍 │
│ hit max_tokens             │                           6 │
│ completion tokens avg      │                      38,346 │
│ completion tokens p50      │                      34,446 │
│ completion tokens p90      │                      60,000 │
│ completion tokens p99      │                      60,000 │
│ elapsed avg                │                      742.4s │
│ TTFT avg                   │                       0.31s │
│ aggregate gen tok/s        │                        51.7 │
│ mean per-request gen tok/s │                        51.5 │
╰────────────────────────────┴─────────────────────────────╯
Failed Final Answers                                                            
╭────┬───┬────────┬───────┬────────────────────────────────────────────────────╮
│  # │ C │ tokens │ score │ final answer                                       │
├────┼───┼────────┼───────┼────────────────────────────────────────────────────┤
│  3 │ 5 │ 31,254 │ FAIL  │ 45 (expected 48, got 45) | 45                      │
│  5 │ 5 │ 60,000 │ FAIL  │ 31 (expected 48, got 31) | What about $n=31 \      │
│ 13 │ 5 │ 60,000 │ FAIL  │ 33 (expected 48, got 33) | D=11: 11, 33,           │
│ 16 │ 5 │ 60,000 │ FAIL  │ 4 (expected 48, got 4) | What about $m=25 \cdot 4  │
│    │   │        │       │ =                                                  │
│ 17 │ 5 │ 60,000 │ FAIL  │ 38903 (expected 48, got 38903) | $38903 /          │
│ 18 │ 5 │ 17,288 │ FAIL  │ 49 (expected 48, got 49) | 49                      │
│ 19 │ 5 │ 19,525 │ FAIL  │ 49 (expected 48, got 49) | 49                      │
│ 24 │ 5 │ 60,000 │ FAIL  │ 3 (expected 48, got 3) | Guest 7: 7 times. $7      │
│    │   │        │       │ \equiv 1 \pmod 3$. R -> G -> R.                    │
│ 27 │ 5 │ 26,455 │ FAIL  │ 49 (expected 48, got 49) | 49                      │
│ 30 │ 5 │ 60,000 │ FAIL  │ 5 (expected 48, got 5) | m=35: divs=[1,5,7,35].    │
│    │   │        │       │ n=1: 1->0. n=5:                                    │
╰────┴───┴────────┴───────┴────────────────────────────────────────────────────╯
Interpretation: EXACT means the final parsed number matches the expected answer.
FAIL means the answer was unparseable or a different number. The 10-slot quality
bar is a rounded distribution: ★=EXACT, ✕=FAIL.

Results saved to 
/tmp/claude-1000/-home-brandonmusic-KLC-SANDBOXES/50980f6d-56ae-4115-a6bf-0d1737
7be8eb/scratchpad/testsuite/results2/hotelmax-fp8-c5-r30.json
```
</details>

# 4. KLD vs BF16 reference

**5 cold-reload runs per dtype** (fresh container each run), 2,047 scored positions,
WikiText-2 @ ctx 2048, `speculative_config=None`. Lower is better.

| run | nvfp4_ds_mla | fp8 |
| --- | ---: | ---: |
| 1 | 0.115928 | 0.100616 |
| 2 | 0.115569 | 0.102350 |
| 3 | 0.117059 | 0.100503 |
| 4 | 0.115955 | 0.100163 |
| 5 | 0.116181 | 0.102360 |
| **mean** | **0.116138** | **0.101198** |
| sd | 0.000559 | 0.001069 |

fp8 shows 0.0149 (-12.9%) lower divergence from the BF16 reference. nvfp4_ds_mla remains the
shipped default because it costs roughly half the KV bytes per token — that is what provides the
context headroom at this preset — and it matched or beat fp8 on the retrieval and ledger tasks.

This is the honest measure of what quantization costs: divergence is small but **not zero**.

# 5. Prefill throughput

`--standalone-prefill --prefill-only --prefill-contexts 8k,64k,128k`.

| Context | nvfp4 TTFT | nvfp4 tok/s | fp8 TTFT | fp8 tok/s |
| --- | ---: | ---: | ---: | ---: |
| 8k | 3.21 s | 2,551 | 3.08 s | 2,660 |
| 64k | 32.96 s | 1,957 | 33.80 s | 1,909 |
| 128k | 70.31 s | 1,833 | 72.78 s | 1,771 |

<details><summary>Prefill — nvfp4_ds_mla log</summary>

```text
╭─────────────────────────────── Configuration ────────────────────────────────╮
│ LLM Inference Benchmark                                                      │
│ Model: GLM-5.2-EXL3-TR3-3.0bpw @ localhost:8000                              │
│ Decode concurrency: [1, 2, 4, 8, 16, 32, 64, 128]                            │
│ Decode contexts: ['0', '16k', '32k', '64k', '128k']                          │
│ Decode: skipped (--prefill-only) | Max tokens: 8192                          │
│ Pre-decode warmup: C=1 max-runnable context for 3s                           │
│ Prefill-only: standalone cold profile (auto) | Sustained decode: 0 cells     │
╰──────────────────────────────────────────────────────────────────────────────╯
Engine: vLLM 
0.11.2.dev280+gilded.gnosis.v20.vllm3e731bc.si1a88b38.fi801d57a.cu132.20260722  
Models: ['GLM-5.2-EXL3-TR3-3.0bpw']
KV cache budget (vLLM metrics): 262,144 tokens (1024 blocks × 64; local 65,536 ×
CP 4; CP source: local process)
Model context length: 262,144 tokens
Prefill tests: standalone cold profile ['8k', '64k', '128k']
Calibrating padding text (run=zamzztpcfkqj, up to 128k)...
  Token targeting: single-point estimate from 8k (use --token-targeting exact 
for /tokenize binary search)
  Calibrated: 6.18 chars/token (cached, source=8k)
  8k: 50,601 chars (~8,191 tokens)
  16k: 101,202 chars (~16,383 tokens)
  32k: 202,404 chars (~32,767 tokens)
  64k: 404,809 chars (~65,535 tokens)
  128k: 809,618 chars (~131,071 tokens)
Done.



llm-decode-bench v0.4.29
Prefill Speed (C=1, client ISL / TTFT)                                          
                                                                                
                                                                PCIe rx/tx      
  Context    Tokens   TTFT (s)   Client tok/s   Server tok/s           avg   N  
 ────────────────────────────────────────────────────────────────────────────── 
  8k          8,202       3.21          2,551      2,565 (2)   21578/21298   2  
  64k        64,516      32.96          1,957      1,963 (1)   77950/79445   1  
  128k      128,890      70.31          1,833      1,838 (1)   88481/90763   1  
                                                                                
Client tok/s = prompt_tokens / TTFT. Integrated scout rows come from the 
prefix-cache scout request that decode needs anyway. Server tok/s is optional 
Prometheus validation when the engine exports prefill counters and the exact 
counter delta is uncontaminated; for vLLM this uses newly computed KV tokens, 
not request prompt tokens.


Results saved to 
/tmp/claude-1000/-home-brandonmusic-KLC-SANDBOXES/50980f6d-56ae-4115-a6bf-0d1737
7be8eb/scratchpad/testsuite/results2/prefill-nvfp4_ds_mla.json
```
</details>

<details><summary>Prefill — fp8 log</summary>

```text
╭─────────────────────────────── Configuration ────────────────────────────────╮
│ LLM Inference Benchmark                                                      │
│ Model: GLM-5.2-EXL3-TR3-3.0bpw @ localhost:8000                              │
│ Decode concurrency: [1, 2, 4, 8, 16, 32, 64, 128]                            │
│ Decode contexts: ['0', '16k', '32k', '64k', '128k']                          │
│ Decode: skipped (--prefill-only) | Max tokens: 8192                          │
│ Pre-decode warmup: C=1 max-runnable context for 3s                           │
│ Prefill-only: standalone cold profile (auto) | Sustained decode: 0 cells     │
╰──────────────────────────────────────────────────────────────────────────────╯
Engine: vLLM 
0.11.2.dev280+gilded.gnosis.v20.vllm3e731bc.si1a88b38.fi801d57a.cu132.20260722  
Models: ['GLM-5.2-EXL3-TR3-3.0bpw']
KV cache budget (vLLM metrics): 262,144 tokens (1024 blocks × 64; local 65,536 ×
CP 4; CP source: local process)
Model context length: 262,144 tokens
Prefill tests: standalone cold profile ['8k', '64k', '128k']
Calibrating padding text (run=vzteflzjnjfk, up to 128k)...
  Token targeting: single-point estimate from 8k (use --token-targeting exact 
for /tokenize binary search)
  Calibrated: 6.18 chars/token (cached, source=8k)
  8k: 50,601 chars (~8,191 tokens)
  16k: 101,202 chars (~16,383 tokens)
  32k: 202,404 chars (~32,767 tokens)
  64k: 404,809 chars (~65,535 tokens)
  128k: 809,618 chars (~131,071 tokens)
Done.



llm-decode-bench v0.4.29
Prefill Speed (C=1, client ISL / TTFT)                                          
                                                                                
                                                                PCIe rx/tx      
  Context    Tokens   TTFT (s)   Client tok/s   Server tok/s           avg   N  
 ────────────────────────────────────────────────────────────────────────────── 
  8k          8,201       3.08          2,660      2,674 (2)   25261/22621   2  
  64k        64,515      33.80          1,909      1,915 (1)   80688/78187   1  
  128k      128,889      72.78          1,771      1,776 (1)   85527/85895   1  
                                                                                
Client tok/s = prompt_tokens / TTFT. Integrated scout rows come from the 
prefix-cache scout request that decode needs anyway. Server tok/s is optional 
Prometheus validation when the engine exports prefill counters and the exact 
counter delta is uncontaminated; for vLLM this uses newly computed KV tokens, 
not request prompt tokens.


Results saved to 
/tmp/claude-1000/-home-brandonmusic-KLC-SANDBOXES/50980f6d-56ae-4115-a6bf-0d1737
7be8eb/scratchpad/testsuite/results2/prefill-fp8.json
```
</details>

# 6. Decode throughput, C1-C8

`--concurrency 1,2,3,4,5,6,7,8 --contexts 0 --duration 20 --max-tokens 8192 --skip-prefill --temperature 0`.
Aggregate tokens/sec:

| concurrency | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| nvfp4_ds_mla | 87.5 | 147.4 | 183.7 | 219.3 | 242.4 | 274.9 | 291.1 | **308.1** |
| fp8 | 86.6 | 143.3 | 184.7 | 217.2 | 241.2 | 270.3 | 289.6 | **312.2** |

Per-user decode (nvfp4): 87.5 / 73.7 / 61.2 / 54.8 / 48.5 / 45.8 / 41.6 / 38.5 tok/s — 3.5x
aggregate scaling C1 to C8. Single-stream figures are for the shipped lossless configuration
(`VLLM_B12X_MLA_SPEC_EXTEND_AS_DECODE=0`); the lossy alternative trades correctness for about
+10 tok/s at C1 and is deliberately disabled.

<details><summary>Decode C1-C8 — nvfp4_ds_mla log</summary>

```text
╭─────────────────────────────── Configuration ────────────────────────────────╮
│ LLM Inference Benchmark                                                      │
│ Model: GLM-5.2-EXL3-TR3-3.0bpw @ localhost:8000                              │
│ Decode concurrency: [1, 2, 3, 4, 5, 6, 7, 8]                                 │
│ Decode contexts: ['0']                                                       │
│ Duration: 20.0s per decode test | Max tokens: 8192                           │
│ Pre-decode warmup: C=1 max-runnable context for 3s                           │
│ Prefill: skipped | Sustained decode: 8 cells                                 │
╰──────────────────────────────────────────────────────────────────────────────╯
Engine: vLLM 
0.11.2.dev280+gilded.gnosis.v20.vllm3e731bc.si1a88b38.fi801d57a.cu132.20260722  
Models: ['GLM-5.2-EXL3-TR3-3.0bpw']
KV cache budget (vLLM metrics): 262,144 tokens (1024 blocks × 64; local 65,536 ×
CP 4; CP source: local process)
Model context length: 262,144 tokens
Prefill tests: skipped
Done.



llm-decode-bench v0.4.29
╭────────────────────────────────── Phase 2 ───────────────────────────────────╮
│ Sustained Decode                                                             │
│ Steady-state decode throughput after the engine has admitted the requested   │
│ concurrency and passed warmup. Use this as the main tuning/regression signal │
│ for kernels, NCCL, DCP, MTP, and scheduler changes.                          │
╰──────────────────────────────────────────────────────────────────────────────╯
Aggregate tok/s  + TTFT/ITL                                                     
╭────────┬────────┬────────┬────────┬────────┬────────┬───────┬────────┬───────╮
│ ctx \  │        │        │        │        │        │       │        │       │
│ conc   │      1 │      2 │      3 │      4 │      5 │     6 │      7 │     8 │
├────────┼────────┼────────┼────────┼────────┼────────┼───────┼────────┼───────┤
│ 0      │   87.5 │  147.4 │  183.7 │  219.3 │  242.4 │ 274.9 │  291.1 │ 308.1 │
│        │ 161/11 │ 253/13 │ 363/16 │ 398/18 │ 442/20 │ 472/… │ 499/23 │ 542/… │
╰────────┴────────┴────────┴────────┴────────┴────────┴───────┴────────┴───────╯
Sustained Decode: aggregate tok/s uses OpenAI stream usage by default 
(continuous completion_tokens when the server supports it). Prometheus is kept 
as validation/scheduler data.
Aggregate source(s): openai_continuous_usage
Per-Request tok/s                                                     
╭────────────┬──────┬──────┬──────┬──────┬──────┬──────┬──────┬──────╮
│ ctx \ conc │    1 │    2 │    3 │    4 │    5 │    6 │    7 │    8 │
├────────────┼──────┼──────┼──────┼──────┼──────┼──────┼──────┼──────┤
│ 0          │ 87.5 │ 73.7 │ 61.2 │ 54.8 │ 48.5 │ 45.8 │ 41.6 │ 38.5 │
╰────────────┴──────┴──────┴──────┴──────┴──────┴──────┴──────┴──────╯
Client request latency: p50 / p90 ms                          
╭────────────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────╮
│ ctx \ conc │   1 │   2 │   3 │   4 │   5 │   6 │   7 │   8 │
├────────────┼─────┼─────┼─────┼─────┼─────┼─────┼─────┼─────┤
│ 0          │ —/— │ —/— │ —/— │ —/— │ —/— │ —/— │ —/— │ —/— │
╰────────────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────╯
Aggregate cells show dim detail as TTFT ms / ITL ms for the same ctx/conc 
coordinate. ITL is computed from observed generated tokens, including streams 
stopped at the measurement boundary; a missing ITL means no stream produced at 
least two measured output tokens. Per-request tok/s and request latency are 
shown in separate per-cell matrices. Completion/sample counts and full 
request-level distributions remain in JSON under request_samples.
Sustained mode: client latency metrics explain request UX variance; aggregate 
tok/s remains the primary throughput signal. 
ITL=(last_token_time-first_token_time)/(output_tokens-1), user tok/s=1/ITL.
Hardware Summary                                                                
╭───┬─┬───────┬───────────┬───────┬─────────┬─────┬──────┬─────┬───────────────╮
│ … │ │ mode  │ GPU avg/… │ Mem … │ W avg/… │ T … │ CPU… │ VR… │ PCIe rx/tx a… │
├───┼─┼───────┼───────────┼───────┼─────────┼─────┼──────┼─────┼───────────────┤
│ 0 │ │ sust… │  100/100% │   48% │ 1067/1… │ 86C │  73C │ 94… │   10945/10913 │
│ 0 │ │ sust… │  100/100% │   49% │ 1104/1… │ 89C │  73C │ 94… │   17508/17924 │
│ 0 │ │ sust… │  100/100% │   48% │ 1126/1… │ 90C │  73C │ 94… │   22034/21826 │
│ 0 │ │ sust… │  100/100% │   49% │ 1136/1… │ 89C │  73C │ 94… │   24354/24152 │
│ 0 │ │ sust… │    99/99% │   47% │ 1137/1… │ 90C │  73C │ 94… │   26736/26290 │
│ 0 │ │ sust… │    99/99% │   47% │ 1141/1… │ 90C │  73C │ 94… │   29873/29618 │
│ 0 │ │ sust… │    99/99% │   47% │ 1144/1… │ 90C │  79C │ 94… │   31554/31721 │
│ 0 │ │ sust… │   99/100% │   46% │ 1144/1… │ 90C │  73C │ 94… │   33035/33073 │
╰───┴─┴───────┴───────────┴───────┴─────────┴─────┴──────┴─────┴───────────────╯
╭──────────────────────── Whole-run GPU Power ─────────────────────────╮
│ avg 1,062 W | max 1,147 W | limit 1,200 W | over 3m 51s | 97 samples │
╰──────────────────────────────────────────────────────────────────────╯
Hardware summary is sampled from nvidia-smi during the measured part of each 
cell. Whole-run GPU power is the sampled sum of GPU power draw across the 
complete benchmark run, not wall-outlet system power. PCIe rx/tx is MB/s and is 
a coarse live diagnostic, not a per-kernel NCCL profiler.

╭────────────────────────────────── Phase 3 ───────────────────────────────────╮
│ Burst / E2E Decode                                                           │
│ Not run. Re-run with --run-burst to append a finite client-facing request    │
│ burst after Sustained Decode. This is intentionally disabled by default      │
│ because it adds another full decode matrix.                                  │
╰──────────────────────────────────────────────────────────────────────────────╯

╭────────────────────────────── Primary Summary ───────────────────────────────╮
│ Primary matrices repeated last so the important numbers are visible without  │
│ scrolling back through diagnostics.                                          │
╰──────────────────────────────────────────────────────────────────────────────╯
Aggregate decode tok/s                                                       
╭────────────┬──────┬───────┬───────┬───────┬───────┬───────┬───────┬───────╮
│ ctx \ conc │    1 │     2 │     3 │     4 │     5 │     6 │     7 │     8 │
├────────────┼──────┼───────┼───────┼───────┼───────┼───────┼───────┼───────┤
│ 0          │ 87.5 │ 147.4 │ 183.7 │ 219.3 │ 242.4 │ 274.9 │ 291.1 │ 308.1 │
╰────────────┴──────┴───────┴───────┴───────┴───────┴───────┴───────┴───────╯

Results saved to 
/tmp/claude-1000/-home-brandonmusic-KLC-SANDBOXES/50980f6d-56ae-4115-a6bf-0d1737
7be8eb/scratchpad/testsuite/results2/decode-c1c8-nvfp4_ds_mla.json
```
</details>

<details><summary>Decode C1-C8 — fp8 log</summary>

```text
╭─────────────────────────────── Configuration ────────────────────────────────╮
│ LLM Inference Benchmark                                                      │
│ Model: GLM-5.2-EXL3-TR3-3.0bpw @ localhost:8000                              │
│ Decode concurrency: [1, 2, 3, 4, 5, 6, 7, 8]                                 │
│ Decode contexts: ['0']                                                       │
│ Duration: 20.0s per decode test | Max tokens: 8192                           │
│ Pre-decode warmup: C=1 max-runnable context for 3s                           │
│ Prefill: skipped | Sustained decode: 8 cells                                 │
╰──────────────────────────────────────────────────────────────────────────────╯
Engine: vLLM 
0.11.2.dev280+gilded.gnosis.v20.vllm3e731bc.si1a88b38.fi801d57a.cu132.20260722  
Models: ['GLM-5.2-EXL3-TR3-3.0bpw']
KV cache budget (vLLM metrics): 262,144 tokens (1024 blocks × 64; local 65,536 ×
CP 4; CP source: local process)
Model context length: 262,144 tokens
Prefill tests: skipped
Done.



llm-decode-bench v0.4.29
╭────────────────────────────────── Phase 2 ───────────────────────────────────╮
│ Sustained Decode                                                             │
│ Steady-state decode throughput after the engine has admitted the requested   │
│ concurrency and passed warmup. Use this as the main tuning/regression signal │
│ for kernels, NCCL, DCP, MTP, and scheduler changes.                          │
╰──────────────────────────────────────────────────────────────────────────────╯
Aggregate tok/s  + TTFT/ITL                                                     
╭────────┬────────┬────────┬────────┬────────┬────────┬───────┬────────┬───────╮
│ ctx \  │        │        │        │        │        │       │        │       │
│ conc   │      1 │      2 │      3 │      4 │      5 │     6 │      7 │     8 │
├────────┼────────┼────────┼────────┼────────┼────────┼───────┼────────┼───────┤
│ 0      │   86.6 │  143.3 │  184.7 │  217.2 │  241.2 │ 270.3 │  289.6 │ 312.2 │
│        │ 153/11 │ 244/14 │ 363/16 │ 393/18 │ 443/20 │ 474/… │ 510/23 │ 538/… │
╰────────┴────────┴────────┴────────┴────────┴────────┴───────┴────────┴───────╯
Sustained Decode: aggregate tok/s uses OpenAI stream usage by default 
(continuous completion_tokens when the server supports it). Prometheus is kept 
as validation/scheduler data.
Aggregate source(s): openai_continuous_usage
Per-Request tok/s                                                     
╭────────────┬──────┬──────┬──────┬──────┬──────┬──────┬──────┬──────╮
│ ctx \ conc │    1 │    2 │    3 │    4 │    5 │    6 │    7 │    8 │
├────────────┼──────┼──────┼──────┼──────┼──────┼──────┼──────┼──────┤
│ 0          │ 86.6 │ 71.7 │ 61.6 │ 54.3 │ 48.2 │ 45.0 │ 41.4 │ 39.0 │
╰────────────┴──────┴──────┴──────┴──────┴──────┴──────┴──────┴──────╯
Client request latency: p50 / p90 ms                          
╭────────────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────╮
│ ctx \ conc │   1 │   2 │   3 │   4 │   5 │   6 │   7 │   8 │
├────────────┼─────┼─────┼─────┼─────┼─────┼─────┼─────┼─────┤
│ 0          │ —/— │ —/— │ —/— │ —/— │ —/— │ —/— │ —/— │ —/— │
╰────────────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────╯
Aggregate cells show dim detail as TTFT ms / ITL ms for the same ctx/conc 
coordinate. ITL is computed from observed generated tokens, including streams 
stopped at the measurement boundary; a missing ITL means no stream produced at 
least two measured output tokens. Per-request tok/s and request latency are 
shown in separate per-cell matrices. Completion/sample counts and full 
request-level distributions remain in JSON under request_samples.
Sustained mode: client latency metrics explain request UX variance; aggregate 
tok/s remains the primary throughput signal. 
ITL=(last_token_time-first_token_time)/(output_tokens-1), user tok/s=1/ITL.
Hardware Summary                                                                
╭───┬─┬───────┬───────────┬───────┬─────────┬─────┬──────┬─────┬───────────────╮
│ … │ │ mode  │ GPU avg/… │ Mem … │ W avg/… │ T … │ CPU… │ VR… │ PCIe rx/tx a… │
├───┼─┼───────┼───────────┼───────┼─────────┼─────┼──────┼─────┼───────────────┤
│ 0 │ │ sust… │  100/100% │   47% │ 1067/1… │ 86C │  74C │ 95… │   10754/10642 │
│ 0 │ │ sust… │  100/100% │   48% │ 1100/1… │ 88C │  73C │ 95… │   17246/17439 │
│ 0 │ │ sust… │  100/100% │   47% │ 1117/1… │ 89C │  73C │ 95… │   21808/21260 │
│ 0 │ │ sust… │  100/100% │   47% │ 1126/1… │ 90C │  73C │ 95… │   24263/24104 │
│ 0 │ │ sust… │    99/99% │   46% │ 1131/1… │ 90C │  74C │ 95… │   25999/25642 │
│ 0 │ │ sust… │    99/99% │   46% │ 1134/1… │ 90C │  74C │ 95… │   28964/28378 │
│ 0 │ │ sust… │   99/100% │   46% │ 1138/1… │ 90C │  74C │ 95… │   30996/31210 │
│ 0 │ │ sust… │   99/100% │   45% │ 1137/1… │ 90C │  73C │ 95… │   33155/32814 │
╰───┴─┴───────┴───────────┴───────┴─────────┴─────┴──────┴─────┴───────────────╯
╭──────────────────────── Whole-run GPU Power ─────────────────────────╮
│ avg 1,060 W | max 1,139 W | limit 1,200 W | over 3m 50s | 96 samples │
╰──────────────────────────────────────────────────────────────────────╯
Hardware summary is sampled from nvidia-smi during the measured part of each 
cell. Whole-run GPU power is the sampled sum of GPU power draw across the 
complete benchmark run, not wall-outlet system power. PCIe rx/tx is MB/s and is 
a coarse live diagnostic, not a per-kernel NCCL profiler.

╭────────────────────────────────── Phase 3 ───────────────────────────────────╮
│ Burst / E2E Decode                                                           │
│ Not run. Re-run with --run-burst to append a finite client-facing request    │
│ burst after Sustained Decode. This is intentionally disabled by default      │
│ because it adds another full decode matrix.                                  │
╰──────────────────────────────────────────────────────────────────────────────╯

╭────────────────────────────── Primary Summary ───────────────────────────────╮
│ Primary matrices repeated last so the important numbers are visible without  │
│ scrolling back through diagnostics.                                          │
╰──────────────────────────────────────────────────────────────────────────────╯
Aggregate decode tok/s                                                       
╭────────────┬──────┬───────┬───────┬───────┬───────┬───────┬───────┬───────╮
│ ctx \ conc │    1 │     2 │     3 │     4 │     5 │     6 │     7 │     8 │
├────────────┼──────┼───────┼───────┼───────┼───────┼───────┼───────┼───────┤
│ 0          │ 86.6 │ 143.3 │ 184.7 │ 217.2 │ 241.2 │ 270.3 │ 289.6 │ 312.2 │
╰────────────┴──────┴───────┴───────┴───────┴───────┴───────┴───────┴───────╯

Results saved to 
/tmp/claude-1000/-home-brandonmusic-KLC-SANDBOXES/50980f6d-56ae-4115-a6bf-0d1737
7be8eb/scratchpad/testsuite/results2/decode-c1c8-fp8.json
```
</details>

<details><summary>Decode C1 dedicated 30 s — nvfp4_ds_mla log</summary>

```text
╭─────────────────────────────── Configuration ────────────────────────────────╮
│ LLM Inference Benchmark                                                      │
│ Model: GLM-5.2-EXL3-TR3-3.0bpw @ localhost:8000                              │
│ Decode concurrency: [1]                                                      │
│ Decode contexts: ['0']                                                       │
│ Duration: 30.0s per decode test | Max tokens: 8192                           │
│ Pre-decode warmup: C=1 max-runnable context for 3s                           │
│ Prefill: skipped | Sustained decode: 1 cells                                 │
╰──────────────────────────────────────────────────────────────────────────────╯
Engine: vLLM 
0.11.2.dev280+gilded.gnosis.v20.vllm3e731bc.si1a88b38.fi801d57a.cu132.20260722  
Models: ['GLM-5.2-EXL3-TR3-3.0bpw']
KV cache budget (vLLM metrics): 262,144 tokens (1024 blocks × 64; local 65,536 ×
CP 4; CP source: local process)
Model context length: 262,144 tokens
Prefill tests: skipped
Done.



llm-decode-bench v0.4.29
╭────────────────────────────────── Phase 2 ───────────────────────────────────╮
│ Sustained Decode                                                             │
│ Steady-state decode throughput after the engine has admitted the requested   │
│ concurrency and passed warmup. Use this as the main tuning/regression signal │
│ for kernels, NCCL, DCP, MTP, and scheduler changes.                          │
╰──────────────────────────────────────────────────────────────────────────────╯
Aggregate tok/s  + TTFT/ITL 
╭────────────┬─────────────╮
│ ctx \ conc │           1 │
├────────────┼─────────────┤
│ 0          │ 83.5 160/12 │
╰────────────┴─────────────╯
Sustained Decode: aggregate tok/s uses OpenAI stream usage by default 
(continuous completion_tokens when the server supports it). Prometheus is kept 
as validation/scheduler data.
Aggregate source(s): openai_continuous_usage
Per-Request tok/s    
╭────────────┬──────╮
│ ctx \ conc │    1 │
├────────────┼──────┤
│ 0          │ 83.5 │
╰────────────┴──────╯
Client request      
latency: p50 / p90  
ms                  
╭────────────┬─────╮
│ ctx \ conc │   1 │
├────────────┼─────┤
│ 0          │ —/— │
╰────────────┴─────╯
Aggregate cells show dim detail as TTFT ms / ITL ms for the same ctx/conc 
coordinate. ITL is computed from observed generated tokens, including streams 
stopped at the measurement boundary; a missing ITL means no stream produced at 
least two measured output tokens. Per-request tok/s and request latency are 
shown in separate per-cell matrices. Completion/sample counts and full 
request-level distributions remain in JSON under request_samples.
Sustained mode: client latency metrics explain request UX variance; aggregate 
tok/s remains the primary throughput signal. 
ITL=(last_token_time-first_token_time)/(output_tokens-1), user tok/s=1/ITL.
Hardware Summary                                                                
╭───┬─┬───────┬───────────┬───────┬─────────┬─────┬──────┬─────┬───────────────╮
│ … │ │ mode  │ GPU avg/… │ Mem … │ W avg/… │ T … │ CPU… │ VR… │ PCIe rx/tx a… │
├───┼─┼───────┼───────────┼───────┼─────────┼─────┼──────┼─────┼───────────────┤
│ 0 │ │ sust… │  100/100% │   47% │ 1071/1… │ 86C │  74C │ 94… │   10791/10730 │
╰───┴─┴───────┴───────────┴───────┴─────────┴─────┴──────┴─────┴───────────────╯
╭────────────────────── Whole-run GPU Power ──────────────────────╮
│ avg 996 W | max 1,099 W | limit 1,200 W | over 48s | 21 samples │
╰─────────────────────────────────────────────────────────────────╯
Hardware summary is sampled from nvidia-smi during the measured part of each 
cell. Whole-run GPU power is the sampled sum of GPU power draw across the 
complete benchmark run, not wall-outlet system power. PCIe rx/tx is MB/s and is 
a coarse live diagnostic, not a per-kernel NCCL profiler.

╭────────────────────────────────── Phase 3 ───────────────────────────────────╮
│ Burst / E2E Decode                                                           │
│ Not run. Re-run with --run-burst to append a finite client-facing request    │
│ burst after Sustained Decode. This is intentionally disabled by default      │
│ because it adds another full decode matrix.                                  │
╰──────────────────────────────────────────────────────────────────────────────╯

╭────────────────────────────── Primary Summary ───────────────────────────────╮
│ Primary matrices repeated last so the important numbers are visible without  │
│ scrolling back through diagnostics.                                          │
╰──────────────────────────────────────────────────────────────────────────────╯
Aggregate decode     
tok/s                
╭────────────┬──────╮
│ ctx \ conc │    1 │
├────────────┼──────┤
│ 0          │ 83.5 │
╰────────────┴──────╯

Results saved to 
/tmp/claude-1000/-home-brandonmusic-KLC-SANDBOXES/50980f6d-56ae-4115-a6bf-0d1737
7be8eb/scratchpad/testsuite/results2/decode-c1-nvfp4_ds_mla.json
```
</details>

<details><summary>Decode C1 dedicated 30 s — fp8 log</summary>

```text
╭─────────────────────────────── Configuration ────────────────────────────────╮
│ LLM Inference Benchmark                                                      │
│ Model: GLM-5.2-EXL3-TR3-3.0bpw @ localhost:8000                              │
│ Decode concurrency: [1]                                                      │
│ Decode contexts: ['0']                                                       │
│ Duration: 30.0s per decode test | Max tokens: 8192                           │
│ Pre-decode warmup: C=1 max-runnable context for 3s                           │
│ Prefill: skipped | Sustained decode: 1 cells                                 │
╰──────────────────────────────────────────────────────────────────────────────╯
Engine: vLLM 
0.11.2.dev280+gilded.gnosis.v20.vllm3e731bc.si1a88b38.fi801d57a.cu132.20260722  
Models: ['GLM-5.2-EXL3-TR3-3.0bpw']
KV cache budget (vLLM metrics): 262,144 tokens (1024 blocks × 64; local 65,536 ×
CP 4; CP source: local process)
Model context length: 262,144 tokens
Prefill tests: skipped
Done.



llm-decode-bench v0.4.29
╭────────────────────────────────── Phase 2 ───────────────────────────────────╮
│ Sustained Decode                                                             │
│ Steady-state decode throughput after the engine has admitted the requested   │
│ concurrency and passed warmup. Use this as the main tuning/regression signal │
│ for kernels, NCCL, DCP, MTP, and scheduler changes.                          │
╰──────────────────────────────────────────────────────────────────────────────╯
Aggregate tok/s  + TTFT/ITL 
╭────────────┬─────────────╮
│ ctx \ conc │           1 │
├────────────┼─────────────┤
│ 0          │ 82.4 153/12 │
╰────────────┴─────────────╯
Sustained Decode: aggregate tok/s uses OpenAI stream usage by default 
(continuous completion_tokens when the server supports it). Prometheus is kept 
as validation/scheduler data.
Aggregate source(s): openai_continuous_usage
Per-Request tok/s    
╭────────────┬──────╮
│ ctx \ conc │    1 │
├────────────┼──────┤
│ 0          │ 82.4 │
╰────────────┴──────╯
Client request      
latency: p50 / p90  
ms                  
╭────────────┬─────╮
│ ctx \ conc │   1 │
├────────────┼─────┤
│ 0          │ —/— │
╰────────────┴─────╯
Aggregate cells show dim detail as TTFT ms / ITL ms for the same ctx/conc 
coordinate. ITL is computed from observed generated tokens, including streams 
stopped at the measurement boundary; a missing ITL means no stream produced at 
least two measured output tokens. Per-request tok/s and request latency are 
shown in separate per-cell matrices. Completion/sample counts and full 
request-level distributions remain in JSON under request_samples.
Sustained mode: client latency metrics explain request UX variance; aggregate 
tok/s remains the primary throughput signal. 
ITL=(last_token_time-first_token_time)/(output_tokens-1), user tok/s=1/ITL.
Hardware Summary                                                                
╭───┬─┬───────┬───────────┬───────┬─────────┬─────┬──────┬─────┬───────────────╮
│ … │ │ mode  │ GPU avg/… │ Mem … │ W avg/… │ T … │ CPU… │ VR… │ PCIe rx/tx a… │
├───┼─┼───────┼───────────┼───────┼─────────┼─────┼──────┼─────┼───────────────┤
│ 0 │ │ sust… │  100/100% │   46% │ 1066/1… │ 86C │  75C │ 95… │   10586/10558 │
╰───┴─┴───────┴───────────┴───────┴─────────┴─────┴──────┴─────┴───────────────╯
╭────────────────────── Whole-run GPU Power ──────────────────────╮
│ avg 993 W | max 1,099 W | limit 1,200 W | over 48s | 21 samples │
╰─────────────────────────────────────────────────────────────────╯
Hardware summary is sampled from nvidia-smi during the measured part of each 
cell. Whole-run GPU power is the sampled sum of GPU power draw across the 
complete benchmark run, not wall-outlet system power. PCIe rx/tx is MB/s and is 
a coarse live diagnostic, not a per-kernel NCCL profiler.

╭────────────────────────────────── Phase 3 ───────────────────────────────────╮
│ Burst / E2E Decode                                                           │
│ Not run. Re-run with --run-burst to append a finite client-facing request    │
│ burst after Sustained Decode. This is intentionally disabled by default      │
│ because it adds another full decode matrix.                                  │
╰──────────────────────────────────────────────────────────────────────────────╯

╭────────────────────────────── Primary Summary ───────────────────────────────╮
│ Primary matrices repeated last so the important numbers are visible without  │
│ scrolling back through diagnostics.                                          │
╰──────────────────────────────────────────────────────────────────────────────╯
Aggregate decode     
tok/s                
╭────────────┬──────╮
│ ctx \ conc │    1 │
├────────────┼──────┤
│ 0          │ 82.4 │
╰────────────┴──────╯

Results saved to 
/tmp/claude-1000/-home-brandonmusic-KLC-SANDBOXES/50980f6d-56ae-4115-a6bf-0d1737
7be8eb/scratchpad/testsuite/results2/decode-c1-fp8.json
```
</details>

# 7. Power

Whole-run GPU power across the 4-GPU set (1,200 W aggregate limit):

| Run | avg | max | duration |
| --- | ---: | ---: | ---: |
| LAVD nvfp4 | 1,130 W | 1,143 W | 19m 20s |
| LAVD fp8 | 1,127 W | 1,138 W | 19m 45s |
| Estonia nvfp4 | 1,105 W | 1,124 W | 10m 56s |
| Estonia fp8 | 1,101 W | 1,127 W | 14m 28s |

Sustained draw is 92-94% of the aggregate limit. Two of the four cards are 600 W-capable parts
software-limited to 300 W, so single-stream decode is partly bounded by the slowest rank's clock.

---

# 8. Independent third-party evaluation

Run and published by **[malaiwah/glm52-exl3-vast](https://github.com/malaiwah/glm52-exl3-vast/tree/main)**
on 2026-07-23/24, independently of this repository's author. The original write-up, the client
harness, and the raw per-question summary are mirrored under
[`independent-eval/`](./independent-eval).

## Results (pass@1, pooled across repeats)

| Benchmark | Questions x repeats = n | EXL3 3.0bpw | GLM-5.2 BF16 (Z.ai published) | ~95% CI |
| --- | ---: | ---: | ---: | ---: |
| AIME 2026 | 30 x 4 = **120** | **99.2** | 99.2 | ±1.6 |
| HMMT Feb 2026 | 33 x 4 = **132** | **95.5** | 92.5 | ±3.6 |
| GPQA Diamond | 198 x 2 = **396** | **91.4** | 91.2 | ±2.8 |

**All three land within sampling noise of the BF16 reference — no measurable reasoning degradation
was detected at 3.0 bits per weight.**

GPQA per-question stability across its 2 repeats: **173 of 198 questions correct both times**,
16 split 1-of-2, 9 wrong both times. Across all 396 generations: **0 truncations, 0 errors**,
1 sample with no extractable answer, 10,561 avg completion tokens,
15.9 h wall time.

## How it was run

- **Sampling** matched to Z.ai's published eval settings: `temperature=1.0`, `top_p=0.95`
- **Max generation**: 163,840 tokens (math), 131,072 (GPQA). Zero truncations occurred.
- **No thinking-effort override** — server default reasoning mode
- **Math prompt**: Z.ai's `Explanation: / Exact Answer: / Confidence:` system prompt
- **Math datasets**: `MathArena/aime_2026`, `MathArena/hmmt_feb_2026`
- **Math grading**: `math-verify` symbolic equivalence; fallback chain `Exact Answer:` line -> last \boxed{} -> none
- **GPQA**: `Idavidrein/gpqa` (gpqa_diamond), simple-evals / Artificial-Analysis MCQ template,
  options deterministically shuffled per (question, repeat), regex letter extraction
- **Client**: async Python harness, 32 concurrent requests (16 GPQA + 8 AIME + 8 HMMT) with all
  three benchmarks running simultaneously; ~65 tok/s aggregate under that mixed long-reasoning
  load; **8.63M completion tokens over ~16 h**
- **pass@1** computed over all repeats pooled

## How this run differed from the shipped preset

Reported by the runner as **fp8 KV cache**, `max_model_len` 524288, server version
`0.17.0rc1.dev4499+g60c82d972` — corresponding to the earlier image tag
`v1-gg-60c82d972-spi1937274-cu132-sm120a` rather than the published `v20-gg6722c1d-si1a88b38`.
Same model weights, same compose and server script, MTP-3 enabled (speculative decoding affects
throughput, not the output distribution).

## Reading these numbers fairly

- The BF16 column is **Z.ai's published figures, not a re-measurement on this harness.** The math
  grading used `math-verify` symbolic equivalence rather than Z.ai's GPT-5.5 judge. This is
  therefore measured-versus-published, not a controlled head-to-head.
- **HMMT +3.0 over BF16 should not be read as the quant beating full precision** — a quantization
  cannot exceed its source in expectation. With 33 questions, a ±3.6 interval and a different
  grader, that gap is noise plus methodology.
- Confidence intervals are simple binomial approximations. Repeats of the same question are
  correlated, so true intervals are somewhat wider.
- AIME and HMMT are small sets (30 and 33 questions). GPQA Diamond at 198 x 2 is the most
  statistically solid of the three.

## Known reproducible quirks (seen on multiple quants, likely model-level)

- **HMMT Q20**: a common reasoning path converges on `1100` where the gold answer is `20460`.
  Reproduced across different quantizations of this base model; this quant scored 2 of 4 repeats.
- **GPQA idx 79** (dataset order): triggers unusually long reasoning chains.

## Incident note from the runner

Three requests stalled mid-run on dropped server connections — client sockets stayed ESTABLISHED
while the server no longer tracked the request. All three hit the client read timeout, auto-retried
and completed. **Zero lost or errored samples in the final data.** Suggested hardening: TCP
keepalives plus a tighter per-request timeout.

Reproduce with the mirrored harness (point `--base-url` at any OpenAI-compatible endpoint):

```bash
python independent-eval/mathbench.py  --dataset MathArena/aime_2026     --repeats 4 --concurrency 8
python independent-eval/mathbench.py  --dataset MathArena/hmmt_feb_2026 --repeats 4 --concurrency 8
python independent-eval/gpqa_bench.py --repeats 2 --concurrency 16
```

---

## Reproducing the serving benchmarks

> **Note (2026-07-25):** the `docker-compose.yml` and `server.sh` embedded further
> below reproduce the BF16-MTP Sections 1-8. The repo's live `server.sh` /
> `docker-compose.yml` are the **tr3-MTP build** (v21 image, `NUM_GPU_BLOCKS_OVERRIDE`
> empty → auto-profile, `VLLM_EXL3_TRELLIS_MIN_M=1`, `MAX_MODEL_LEN=524288`);
> `./server.sh start` below pulls and runs that current preset.

```bash
hf download brandonmusic/GLM-5.2-EXL3-TR3-3.0bpw --local-dir "$HOME/models/GLM-5.2-EXL3-TR3-3.0bpw"
cd "$HOME/models/GLM-5.2-EXL3-TR3-3.0bpw"
chmod +x server.sh && ./server.sh start
```

<details><summary>Full docker-compose.yml (all serve flags)</summary>

```yaml
services:
  glm52:
    image: ${IMAGE:-verdictai/glm52-exl3-sparkinfer:v31-gg-v20-sic3828fd-vllm0c79e41-cu132-sm120a@sha256:0433ae94665b769b78dd301f952d907508a3ba80bce47a1630ec20ade8812dff}
    container_name: glm52-exl3-sparkinfer
    ports:
      - "${BIND_ADDRESS:-127.0.0.1}:${PORT:-8000}:8000"
    gpus: all
    shm_size: "32g"
    ipc: host
    ulimits:
      memlock: -1
      nofile: 1048576
    environment:
      CUDA_VISIBLE_DEVICES: "${CUDA_VISIBLE_DEVICES:-3,1,2,0}"
      CUDA_DEVICE_ORDER: PCI_BUS_ID
      CUDA_DEVICE_MAX_CONNECTIONS: "32"
      CUTE_DSL_ARCH: sm_120a
      TORCH_CUDA_ARCH_LIST: 12.0a
      FLASHINFER_CUDA_ARCH_LIST: 12.0f
      FLASHINFER_DISABLE_VERSION_CHECK: "1"
      OMP_NUM_THREADS: "16"
      PYTORCH_CUDA_ALLOC_CONF: expandable_segments:True
      SAFETENSORS_FAST_GPU: "1"
      NCCL_IB_DISABLE: "1"
      NCCL_P2P_LEVEL: SYS
      NCCL_PROTO: LL,LL128,Simple
      VLLM_USE_FLASHINFER_SAMPLER: "1"
      VLLM_USE_B12X_FP8_GEMM: "1"
      VLLM_USE_B12X_SPARSE_INDEXER: "1"
      VLLM_USE_B12X_MOE: "1"
      VLLM_USE_V2_MODEL_RUNNER: "1"
      VLLM_ENABLE_PCIE_ALLREDUCE: "1"
      VLLM_PCIE_ALLREDUCE_BACKEND: b12x
      VLLM_PCIE_ONESHOT_ALLREDUCE_MAX_SIZE: "${VLLM_PCIE_ONESHOT_ALLREDUCE_MAX_SIZE:-64KB}"
      VLLM_PCIE_ONESHOT_FUSED_ADD_RMS_NORM_MAX_SIZE: "${VLLM_PCIE_ONESHOT_FUSED_ADD_RMS_NORM_MAX_SIZE:-84KB}"
      VLLM_PCIE_DMA_FP8: ag
      B12X_PCIE_DMA_FP8: ag
      VLLM_CPP_AR_1STAGE_NCCL_CUTOFF: 56KB
      VLLM_CPP_AR_IGNORE_CUTOFF_MAX_ROWS: "0"
      VLLM_RTX6K_FUSED_ALLREDUCE_ADD: "0"
      VLLM_RTX6K_FUSED_ALLREDUCE_ADD_END_BARRIER: "0"
      VLLM_USE_AOT_COMPILE: "1"
      VLLM_USE_BREAKABLE_CUDAGRAPH: "0"
      VLLM_USE_FUSED_MOE_GROUPED_TOPK: "1"
      VLLM_USE_B12X_MHC: "1"
      B12X_MHC_MAX_TOKENS: "16384"
      VLLM_USE_B12X_WO_PROJECTION: "1"
      B12X_MLA_SM120_UNIFIED: "1"
      B12X_DENSE_SPLITK_TURBO: "1"
      B12X_W4A16_TC_DECODE: "1"
      B12X_MOE_FORCE_A16: "1"
      VLLM_DISABLE_SHARED_EXPERTS_STREAM: "${VLLM_DISABLE_SHARED_EXPERTS_STREAM:-1}"
      VLLM_DISABLED_KERNELS: MarlinFP8ScaledMMLinearKernel
      VLLM_B12X_MLA_SPEC_EXTEND_AS_DECODE: "${VLLM_B12X_MLA_SPEC_EXTEND_AS_DECODE:-0}"
      VLLM_B12X_MLA_SPEC_DECODE_MAX_Q: "8"
      VLLM_USE_B12X_DCP_A2A: "1"
      VLLM_DCP_A2A_MAX_TOKENS: "16"
      VLLM_DCP_A2A_LARGE_BACKEND: ag_rs
      VLLM_DCP_GLOBAL_TOPK: "${VLLM_DCP_GLOBAL_TOPK:-1}"
      VLLM_DCP_SHARD_DRAFT: "${VLLM_DCP_SHARD_DRAFT:-1}"
      VLLM_DCP_QUERY_SPLIT: "0"
      VLLM_B12X_MLA_CKV_GATHER: "1"
      VLLM_B12X_MLA_CKV_GATHER_MIN_TOKENS: "512"
      VLLM_B12X_MLA_CKV_GATHER_MAX_TOKENS: "16384"
      ENABLE_MTP: "${ENABLE_MTP:-1}"
      MTP_TOKENS: "${MTP_TOKENS:-3}"
      MTP_DRAFT_SAMPLE_METHOD: "${MTP_DRAFT_SAMPLE_METHOD:-greedy}"
      ENABLE_ASYNC_SCHEDULING: "${ENABLE_ASYNC_SCHEDULING:-0}"
      GLM52_INDEX_TOPK_PATTERN: "${GLM52_INDEX_TOPK_PATTERN:-FFFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSS}"
      NUM_GPU_BLOCKS_OVERRIDE: "${NUM_GPU_BLOCKS_OVERRIDE:-1024}"
      MAX_NUM_BATCHED_TOKENS: "${MAX_NUM_BATCHED_TOKENS:-3072}"
      VLLM_EXL3_TRELLIS_MIN_M: "4"
      VLLM_EXL3_TRELLIS_MAX_M: "32"
      VLLM_EXL3_TRELLIS_BLOCK_M: "8"
      VLLM_EXL3_PREFILL_CHUNK: "128"
      VLLM_CACHE_DIR: /cache/jit/vllm
      TRITON_CACHE_DIR: /cache/jit/triton
      TORCH_EXTENSIONS_DIR: /cache/jit/torch_extensions
      TORCHINDUCTOR_CACHE_DIR: /cache/jit/torchinductor
      FLASHINFER_WORKSPACE_BASE: /cache/jit/flashinfer
      XDG_CACHE_HOME: /cache/jit
      TVM_FFI_CACHE_DIR: /cache/jit/tvm-ffi
      VLLM_MEMORY_PROFILE_INCLUDE_ATTN: "1"
      VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS: "1"
      VLLM_DEBUG_WORKSPACE: "${VLLM_DEBUG_WORKSPACE:-0}"
    volumes:
      - ${MODEL_DIR:-/home/brandonmusic/models/GLM-5.2-EXL3-TR3-3.0bpw}:/model:ro
      - ${CACHE_DIR:-/home/brandonmusic/.cache/glm52-tr3-release}:/cache:rw
    entrypoint:
      - /bin/bash
      - -lc
    command:
      - |
        unset NCCL_GRAPH_FILE NCCL_GRAPH_DUMP_FILE VLLM_B12X_MLA_EXTEND_MAX_CHUNKS
        spec_args=()
        if [[ "$${ENABLE_MTP:-1}" == "1" ]]; then
          printf -v spec_config '{"method":"mtp","num_speculative_tokens":%s,"moe_backend":"triton","draft_sample_method":"%s"}' \
            "$${MTP_TOKENS:-3}" "$${MTP_DRAFT_SAMPLE_METHOD:-greedy}"
          spec_args=(--speculative-config "$$spec_config")
        fi
        if [[ "$${ENABLE_ASYNC_SCHEDULING:-0}" == "1" ]]; then
          async_args=(--async-scheduling)
        else
          async_args=(--no-async-scheduling)
        fi
        index_pattern="$${GLM52_INDEX_TOPK_PATTERN}"
        if [[ "$${#index_pattern}" -ne 78 ]]; then
          printf 'GLM-5.2 index_topk_pattern must cover all 78 layers (got %s)\n' "$${#index_pattern}" >&2
          exit 2
        fi
        printf -v hf_overrides '{"use_index_cache":true,"index_topk_pattern":"%s"}' "$${index_pattern}"
        block_args=()
        if [[ -n "$${NUM_GPU_BLOCKS_OVERRIDE:-}" ]]; then
          block_args=(--num-gpu-blocks-override "$${NUM_GPU_BLOCKS_OVERRIDE}")
        fi
        exec vllm serve /model \
          --served-model-name GLM-5.2-EXL3-TR3-3.0bpw \
          --host 0.0.0.0 --port 8000 --trust-remote-code \
          --tensor-parallel-size 4 \
          --decode-context-parallel-size 4 \
          --dcp-comm-backend a2a \
          --dcp-kv-cache-interleave-size ${DCP_KV_CACHE_INTERLEAVE_SIZE:-64} \
          --seed 0 \
          --quantization exl3 \
          --kv-cache-dtype ${KV_CACHE_DTYPE:-nvfp4_ds_mla} \
          --attention-backend B12X_MLA_SPARSE \
          --moe-backend b12x \
          --load-format safetensors \
          --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE","cudagraph_capture_sizes":[4,8,12,16,20,24,28,32],"custom_ops":["all"],"pass_config":{"fuse_allreduce_rms":true}}' \
          --gpu-memory-utilization ${GPU_MEMORY_UTILIZATION:-0.96} \
          --max-model-len ${MAX_MODEL_LEN:-262144} \
          --max-num-seqs 8 \
          --max-num-batched-tokens $${MAX_NUM_BATCHED_TOKENS:-3072} \
          --max-cudagraph-capture-size 32 \
          --enable-chunked-prefill \
          --enable-prefix-caching \
          --enable-auto-tool-choice \
          --tool-call-parser glm47 \
          --reasoning-parser glm45 \
          --default-chat-template-kwargs '{"reasoning_effort":"high"}' \
          --hf-overrides "$${hf_overrides}" \
          "$${block_args[@]}" \
          "$${async_args[@]}" \
          "$${spec_args[@]}"

```
</details>

<details><summary>Full server.sh</summary>

```bash
#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

export IMAGE="${IMAGE:-verdictai/glm52-exl3-sparkinfer:v31-gg-v20-sic3828fd-vllm0c79e41-cu132-sm120a@sha256:0433ae94665b769b78dd301f952d907508a3ba80bce47a1630ec20ade8812dff}"
export MODEL_DIR="${MODEL_DIR:-$SCRIPT_DIR}"
export CACHE_DIR="${CACHE_DIR:-$HOME/.cache/glm52-exl3-sparkinfer}"
export PORT="${PORT:-8000}"
export BIND_ADDRESS="${BIND_ADDRESS:-127.0.0.1}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-3,1,2,0}"
export GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.96}"
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-262144}"
export DCP_KV_CACHE_INTERLEAVE_SIZE="${DCP_KV_CACHE_INTERLEAVE_SIZE:-64}"
export VLLM_B12X_MLA_SPEC_EXTEND_AS_DECODE="${VLLM_B12X_MLA_SPEC_EXTEND_AS_DECODE:-0}"
export VLLM_DCP_SHARD_DRAFT="${VLLM_DCP_SHARD_DRAFT:-1}"
export VLLM_DCP_GLOBAL_TOPK="${VLLM_DCP_GLOBAL_TOPK:-1}"
export VLLM_DISABLE_SHARED_EXPERTS_STREAM="${VLLM_DISABLE_SHARED_EXPERTS_STREAM:-1}"
export VLLM_PCIE_ONESHOT_ALLREDUCE_MAX_SIZE="${VLLM_PCIE_ONESHOT_ALLREDUCE_MAX_SIZE:-64KB}"
export VLLM_PCIE_ONESHOT_FUSED_ADD_RMS_NORM_MAX_SIZE="${VLLM_PCIE_ONESHOT_FUSED_ADD_RMS_NORM_MAX_SIZE:-84KB}"
export ENABLE_MTP="${ENABLE_MTP:-1}"
export MTP_TOKENS="${MTP_TOKENS:-3}"
export MTP_DRAFT_SAMPLE_METHOD="${MTP_DRAFT_SAMPLE_METHOD:-greedy}"
export ENABLE_ASYNC_SCHEDULING="${ENABLE_ASYNC_SCHEDULING:-0}"
export GLM52_INDEX_TOPK_PATTERN="${GLM52_INDEX_TOPK_PATTERN:-FFFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSS}"
export NUM_GPU_BLOCKS_OVERRIDE="${NUM_GPU_BLOCKS_OVERRIDE:-1024}"
export MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-3072}"
export COMPOSE_PROJECT_NAME="${COMPOSE_PROJECT_NAME:-glm52-exl3-sparkinfer}"

COMPOSE_FILE="${COMPOSE_FILE:-/tmp/claude-1000/-home-brandonmusic-KLC-SANDBOXES/50980f6d-56ae-4115-a6bf-0d17377be8eb/scratchpad/testsuite/config/docker-compose.yml}"
COMPOSE=(docker compose -f "$COMPOSE_FILE")

usage() {
  cat <<'EOF'
Usage: ./server.sh [start|stop|restart|logs|status|pull]

Environment overrides:
  IMAGE, MODEL_DIR, CACHE_DIR, PORT, BIND_ADDRESS, CUDA_VISIBLE_DEVICES,
  GPU_MEMORY_UTILIZATION, MAX_MODEL_LEN, DCP_KV_CACHE_INTERLEAVE_SIZE,
  VLLM_B12X_MLA_SPEC_EXTEND_AS_DECODE, VLLM_DCP_SHARD_DRAFT,
  VLLM_DCP_GLOBAL_TOPK,
  VLLM_DISABLE_SHARED_EXPERTS_STREAM,
  VLLM_PCIE_ONESHOT_ALLREDUCE_MAX_SIZE,
  VLLM_PCIE_ONESHOT_FUSED_ADD_RMS_NORM_MAX_SIZE,
  ENABLE_MTP, MTP_TOKENS, MTP_DRAFT_SAMPLE_METHOD, ENABLE_ASYNC_SCHEDULING,
  GLM52_INDEX_TOPK_PATTERN,
  NUM_GPU_BLOCKS_OVERRIDE,
  MAX_NUM_BATCHED_TOKENS,
  COMPOSE_PROJECT_NAME, COMPOSE_FILE
EOF
}

require_runtime() {
  command -v docker >/dev/null 2>&1 || {
    echo "docker is required" >&2
    exit 1
  }
  docker compose version >/dev/null
  [[ -f "$COMPOSE_FILE" ]] || {
    echo "Compose file not found: $COMPOSE_FILE" >&2
    exit 1
  }
}

require_model() {
  [[ -f "$MODEL_DIR/config.json" ]] || {
    echo "Model config not found: $MODEL_DIR/config.json" >&2
    exit 1
  }
  [[ -f "$MODEL_DIR/model.safetensors.index.json" ]] || {
    echo "Model index not found: $MODEL_DIR/model.safetensors.index.json" >&2
    exit 1
  }
  mkdir -p "$CACHE_DIR"
}

action="${1:-start}"
require_runtime

case "$action" in
  start)
    require_model
    docker pull "$IMAGE"
    "${COMPOSE[@]}" up -d --force-recreate
    echo "Starting on http://localhost:$PORT"
    echo "Follow startup with: $0 logs"
    ;;
  stop)
    "${COMPOSE[@]}" down
    ;;
  restart)
    require_model
    docker pull "$IMAGE"
    "${COMPOSE[@]}" up -d --force-recreate
    echo "Restarting on http://localhost:$PORT"
    ;;
  logs)
    "${COMPOSE[@]}" logs --tail 100 -f glm52
    ;;
  status)
    "${COMPOSE[@]}" ps
    curl -fsS "http://localhost:$PORT/v1/models" || true
    printf '\n'
    ;;
  pull)
    docker pull "$IMAGE"
    ;;
  -h|--help|help)
    usage
    ;;
  *)
    usage >&2
    exit 2
    ;;
esac

```
</details>


## Long-context needle-in-a-haystack (v28, 2026-07-26)

Run on 4x RTX PRO 6000 Blackwell (SM120), TP4 / DCP4, `nvfp4_ds_mla` KV,
MTP-3 draft at layer 78, `VLLM_EXL3_TRELLIS_MIN_M=1`, `max-model-len 524288`.
A unique authorization code is planted at three depths (0.1 / 0.5 / 0.9) in a
filler document and requested back with greedy decoding.

### `nvfp4_ds_mla` KV

| target ctx | real prompt tokens | depth 0.1 | depth 0.5 | depth 0.9 |
|---|---|---|---|---|
| 8k    |   4,844 | HIT | HIT | HIT |
| 32k   |  19,316 | HIT | HIT | HIT |
| 65k   |  39,188 | HIT | HIT | HIT |
| 128k  |  77,168 | HIT | HIT | HIT |
| 200k  | 199,783 | HIT | HIT | HIT |
| 300k  | 299,648 | HIT | HIT | HIT |
| 400k  | 399,512 | HIT | HIT | HIT |
| 480k  | 479,396 | HIT | HIT | HIT |

**24/24 needles recovered.** GPU KV cache 959,744-998,400 tokens. Deepest 480k
probe 315 s.

### `fp8` KV

Identical checkpoint, weights, draft and flags; only `--kv-cache-dtype` changed.

| target ctx | real prompt tokens | depth 0.1 | depth 0.5 | depth 0.9 |
|---|---|---|---|---|
| 8k    |   8,010 | HIT | HIT | HIT |
| 65k   |  64,965 | HIT | HIT | HIT |
| 128k  | 127,854 | HIT | HIT | HIT |
| 200k  | 199,784 | HIT | HIT | HIT |
| 300k  | 299,648 | HIT | HIT | HIT |
| 480k  | 479,396 | HIT | HIT | HIT |

**18/18 needles recovered.** GPU KV cache 648,192 tokens (8-bit vs 4-bit, so a
smaller pool than `nvfp4_ds_mla` at the same utilization). Deepest 480k probe
311 s.

**Combined: 42/42 across both KV dtypes**, three depths each, to ~480k real
prompt tokens -- within ~45k of the 524,288 `max-model-len` ceiling. No garbled
output and zero engine restarts in either lane.

Context: vLLM issue #183 reported that `VLLM_EXL3_TRELLIS_MIN_M=1` silently
corrupts long-context output. That did not reproduce here in either KV dtype. Independently, the
fused Trellis MoE was verified bitwise-correct at m=1,2,3 at this exact geometry
(tile 64x256x64x256, `block_size_m=8`, capacity 32, topk=8) with the scratch
arena NaN-poisoned. Note that `MIN_M=1` widens the Trellis window so the draft's
m=1..3 GEMMs stay on the fused, graph-capturable path; it is not a per-token
capture and carries no throughput penalty.


## Boot without the MIN_M workaround (v29, 2026-07-27)

Historically, serving an EXL3 rank-sliced tr3 MTP draft required setting
`VLLM_EXL3_TRELLIS_MIN_M=1` by hand; without it the engine could not start
(vLLM issue #183):

```text
RuntimeError: EXL3 eager parity path entered during CUDA graph capture (m=3);
              capture sizes must lie inside the Trellis window [4, 32]
```

v29 removes the requirement. The backend stamps each layer's draft/target role
at construction (`runner_type == "draft"`), where the vllm-config context is
live, and defaults draft layers' Trellis window to `MIN_CAPTURABLE_TRELLIS_M=1`
automatically. Target layers keep the historical default of 4; an explicit
`VLLM_EXL3_TRELLIS_MIN_M` still overrides both.

Validation on this rig (4x RTX PRO 6000 SM120, TP4/DCP4, tr3 MTP-78, MTP-3),
with `VLLM_EXL3_TRELLIS_MIN_M` entirely unset:

| gate | result |
|---|---|
| engine boot + serve | PASS (previously guaranteed startup failure) |
| capture-time window error in logs | 0 occurrences |
| blank-env `int('')` startup crash | 0 occurrences (blank now means unset) |
| greedy inference | PASS |
| needle 8k/65k/128k x depths 0.1/0.5/0.9 | 9/9 recovered |

The compose files in this repo now leave `VLLM_EXL3_TRELLIS_MIN_M` unset by
default. Fix commits: vLLM PR #139 `239ba678b5` + `796ea923f1`.


## v30 (2026-07-27): env-knob registration + sparkinfer PR#79 module

Delta vs v29 (which carries all correctness fixes): (1) the nine EXL3 env knobs are registered
in vllm envs.py -- startup 'Unknown vLLM environment variable' warnings drop 15 -> 8 (the
remaining 8 are base-runtime-owned), and the knobs join the torch.compile cache-key factors
(one-time ~70 s recompile after changing one); (2) the SparkInfer wheel is rebuilt from the
PR#49 branch rebased onto master AFTER PR#79 ("perf(pcie): add exact DCP top-k owner exchange"),
so the CUDA-IPC owner-exchange module ships in the image. It is DORMANT here: PR#79 has zero
overlap with the EXL3/MoE lane (only +1 line outside its new files), and the vLLM-side owner
algorithm is not yet in this image's pinned base. Gates on the pinned v30: boot with
VLLM_EXL3_TRELLIS_MIN_M unset PASS, 0 capture-window errors, warnings 15->8 confirmed,
greedy inference PASS, tool-calls 4/4.


## v31 (2026-07-27): unified v20 base refresh (SparkInfer c3828fd)

Base bump only on the SparkInfer axis (vLLM pin unchanged at 0c79e41, so the 13-file vLLM
overlay is byte-identical to v30). Wheel rebuilt from PR#49 (11 commits) on the new
integration pin c3828fd and verified a strict superset of the base's canonical SparkInfer
(168/168 source files present). Gates on the pinned digest: boot with
VLLM_EXL3_TRELLIS_MIN_M unset PASS, warnings 8, 0 capture errors, inference PASS,
tool-calls 4/4, KV 963,840.

Correction note for v30: its wheel was built from SparkInfer master rather than the base's
integration pin, so the pip install replaced the base's canonical SparkInfer with a tree
missing the integration-only PCIe calibration commits. No effect on the published serving
configs (they pin DCP controls explicitly, and calibration only engages on 'auto'), but
helper/auto-calibration users should prefer v31.
