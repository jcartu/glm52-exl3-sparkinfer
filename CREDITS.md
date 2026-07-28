# Credits, ownership, and license boundaries

This release is an integration of many independent projects. No single person or team created the
whole stack. This page says plainly who made each layer, what this repository added, and which
license applies where.

If a name or project is missing or described incorrectly, please open an issue or pull request.
Attribution corrections are treated as release fixes.

## The credit chain at a glance

| Layer | Created or maintained by | What this release uses |
|---|---|---|
| Foundation model | [Z.ai / GLM-5 Team](https://github.com/zai-org/GLM-5) | GLM-5.2 architecture, weights, tokenizer, chat behavior, and research |
| Quantized checkpoint | [Brandon Music](https://huggingface.co/brandonmusic) | The published 3.0 bpw EXL3-TR3, four-rank checkpoint |
| Quantization format and reference kernels | [turboderp](https://github.com/turboderp) and [ExLlamaV3 contributors](https://github.com/turboderp-org/exllamav3/graphs/contributors) | EXL3/Trellis format, packing, metadata, and CUDA implementation |
| Serving engine | [vLLM Team and contributors](https://github.com/vllm-project/vllm/graphs/contributors) | Distributed model loading, scheduling, LoRA framework, and OpenAI-compatible API |
| Blackwell kernels | [Luke Alonso](https://github.com/lukealonso) and [Sparkinfer contributors](https://github.com/local-inference-lab/sparkinfer/graphs/contributors) | Staged Trellis MoE, sparse attention, and PCIe-oriented Blackwell execution |
| EXL3 integration | [Brandon Music](https://github.com/brandonmmusic-max), [David Young](https://github.com/davidsyoung), and the [local-inference-lab contributors](https://github.com/orgs/local-inference-lab/repositories) | Rank-sliced EXL3 loading, Trellis execution, and the prefill path used by this lineage |
| MTP head and independent evaluation | [malaiwah](https://github.com/malaiwah) | Rank-sliced MTP-78 work and independently produced evaluation material |
| Dynamic LoRA release | [Josh Cartu (`jcartu`)](https://github.com/jcartu), building on Brandon Music's repository | LoRA bridge integration, source repair, qualification, reproducible image, deployment, documentation, and publication |

## 1. Model and research

**Z.ai and the GLM-5 Team created GLM-5.2.** This repository did not train the model and does not
redistribute its weights.

Primary sources:

- [Official GLM-5.2 model card and license](https://huggingface.co/zai-org/GLM-5.2)
- [GLM-5 source repository](https://github.com/zai-org/GLM-5)
- [Technical report: *GLM-5: from Vibe Coding to Agentic Engineering*](https://arxiv.org/abs/2602.15763)

<details>
<summary>Full author list from the technical report</summary>

GLM-5-Team; Aohan Zeng; Xin Lv; Zhenyu Hou; Zhengxiao Du; Qinkai Zheng; Bin Chen; Da Yin;
Chendi Ge; Chenghua Huang; Chengxing Xie; Chenzheng Zhu; Congfeng Yin; Cunxiang Wang;
Gengzheng Pan; Hao Zeng; Haoke Zhang; Haoran Wang; Huilong Chen; Jiajie Zhang; Jian Jiao;
Jiaqi Guo; Jingsen Wang; Jingzhao Du; Jinzhu Wu; Kedong Wang; Lei Li; Lin Fan; Lucen Zhong;
Mingdao Liu; Mingming Zhao; Pengfan Du; Qian Dong; Rui Lu; Shuang-Li; Shulin Cao; Song Liu;
Ting Jiang; Xiaodong Chen; Xiaohan Zhang; Xuancheng Huang; Xuezhen Dong; Yabo Xu; Yao Wei;
Yifan An; Yilin Niu; Yitong Zhu; Yuanhao Wen; Yukuo Cen; Yushi Bai; Zhongpei Qiao; Zihan Wang;
Zikang Wang; Zilin Zhu; Ziqiang Liu; Zixuan Li; Bojie Wang; Bosi Wen; Can Huang; Changpeng Cai;
Chao Yu; Chen Li; Chengwei Hu; Chenhui Zhang; Dan Zhang; Daoyan Lin; Dayong Yang; Di Wang;
Ding Ai; Erle Zhu; Fangzhou Yi; Feiyu Chen; Guohong Wen; Hailong Sun; Haisha Zhao; Haiyi Hu;
Hanchen Zhang; Hanrui Liu; Hanyu Zhang; Hao Peng; Hao Tai; Haobo Zhang; He Liu; Hongwei Wang;
Hongxi Yan; Hongyu Ge; Huan Liu; Huanpeng Chu; Jia'ni Zhao; Jiachen Wang; Jiajing Zhao;
Jiamin Ren; Jiapeng Wang; Jiaxin Zhang; Jiayi Gui; Jiayue Zhao; Jijie Li; Jing An; Jing Li;
Jingwei Yuan; Jinhua Du; Jinxin Liu; Junkai Zhi; Junwen Duan; Kaiyue Zhou; Kangjian Wei;
Ke Wang; Keyun Luo; Laiqiang Zhang; Leigang Sha; Liang Xu; Lindong Wu; Lintao Ding; Lu Chen;
Minghao Li; Nianyi Lin; Pan Ta; Qiang Zou; Rongjun Song; Ruiqi Yang; Shangqing Tu;
Shangtong Yang; Shaoxiang Wu; Shengyan Zhang; Shijie Li; Shuang Li; Shuyi Fan; Wei Qin;
Wei Tian; Weining Zhang; Wenbo Yu; Wenjie Liang; Xiang Kuang; Xiangmeng Cheng; Xiangyang Li;
Xiaoquan Yan; Xiaowei Hu; Xiaoying Ling; Xing Fan; Xingye Xia; Xinyuan Zhang; Xinze Zhang;
Xirui Pan; Xu Zou; Xunkai Zhang; Yadi Liu; Yandong Wu; Yanfu Li; Yidong Wang; Yifan Zhu;
Yijun Tan; Yilin Zhou; Yiming Pan; Ying Zhang; Yinpei Su; Yipeng Geng; Yong Yan; Yonglin Tan;
Yuean Bi; Yuhan Shen; Yuhao Yang; Yujiang Li; Yunan Liu; Yunqing Wang; Yuntao Li; Yurong Wu;
Yutao Zhang; Yuxi Duan; Yuxuan Zhang; Zezhen Liu; Zhengtao Jiang; Zhenhe Yan; Zheyu Zhang;
Zhixiang Wei; Zhuo Chen; Zhuoer Feng; Zijun Yao; Ziwei Chai; Ziyuan Wang; Zuzhou Zhang;
Bin Xu; Minlie Huang; Hongning Wang; Juanzi Li; Yuxiao Dong; and Jie Tang.

The authoritative spelling and ordering are in the linked report.

</details>

## 2. Checkpoint, format, and adapter boundary

### Quantized checkpoint

[Brandon Music](https://huggingface.co/brandonmusic) created and published
[`brandonmusic/GLM-5.2-EXL3-TR3-3.0bpw`](https://huggingface.co/brandonmusic/GLM-5.2-EXL3-TR3-3.0bpw).
It is the 332.19 GB, rank-sliced EXL3-TR3 checkpoint used by the deployment. Brandon also created
the original repository, test-suite lineage, and the principal EXL3 vLLM/Sparkinfer integration
on which this dynamic-LoRA release is built.

### EXL3 and Trellis

[turboderp](https://github.com/turboderp) and every
[ExLlamaV3 contributor](https://github.com/turboderp-org/exllamav3/graphs/contributors) are
credited for the EXL3/Trellis quantization format, tools, metadata contracts, and reference CUDA
implementation. The installed extension is derived from that project; this repository does not
claim authorship of EXL3.

### Qualification adapter

The BF16 rank-16 adapter used during qualification was a local test artifact. Its metadata named
neither a creator nor a license. It is **not included or redistributed**, and this project makes
no ownership claim over it. Users must supply an adapter they are authorized to use. The published
contract describes compatibility only: PEFT safetensors, BF16, rank 16, alpha 32, and the documented
GLM target modules.

## 3. Serving and kernel projects

### vLLM

The [vLLM Team and all vLLM contributors](https://github.com/vllm-project/vllm/graphs/contributors)
created and maintain the distributed serving engine, scheduler, model loader, LoRA framework,
OpenAI-compatible server, prefix cache, speculative decoding framework, and much of the runtime
used here.

Release-specific EXL3 and LoRA work is visible in:

- [`local-inference-lab/vllm` pull request 139](https://github.com/local-inference-lab/vllm/pull/139), authored by [Brandon Music](https://github.com/brandonmmusic-max)
- [`jcartu/vllm` commit `95d7914de1df93b39fe44957377311ddb752bd2f`](https://github.com/jcartu/vllm/commit/95d7914de1df93b39fe44957377311ddb752bd2f), the immutable source used by release r2

[David Young](https://github.com/davidsyoung) is explicitly credited for the planned Trellis
prefill work incorporated into the EXL3 path.

### Sparkinfer

[Luke Alonso](https://github.com/lukealonso) and every
[Sparkinfer contributor](https://github.com/local-inference-lab/sparkinfer/graphs/contributors)
created and maintain the Blackwell-focused kernels and APIs used for Trellis MoE, sparse attention,
and PCIe execution.

Release-specific work is visible in:

- [`local-inference-lab/sparkinfer` pull request 49](https://github.com/local-inference-lab/sparkinfer/pull/49), authored by [Brandon Music](https://github.com/brandonmmusic-max)
- [`jcartu/sparkinfer` commit `fc8051efee755563e2c7a4ce87ce8b683db58381`](https://github.com/jcartu/sparkinfer/commit/fc8051efee755563e2c7a4ce87ce8b683db58381), the immutable source used by release r2

### Runtime lineage

The digest-pinned base runtime combines work published by
[local-inference-lab](https://github.com/local-inference-lab),
[voipmonitor](https://github.com/voipmonitor), and
[Verdict AI](https://hub.docker.com/u/verdictai). Their contributors established the Blackwell
vLLM/Sparkinfer image lineage on which this smaller release overlay is built.

[malaiwah](https://github.com/malaiwah) is credited for the LDLQ-calibrated, rank-sliced MTP-78
head work and for the independent evaluation preserved under
[`docs/independent-eval/`](docs/independent-eval/). The upstream evaluation repository is
[`malaiwah/glm52-exl3-vast`](https://github.com/malaiwah/glm52-exl3-vast).

## 4. This repository's integration work

[Brandon Music](https://github.com/brandonmmusic-max) created the original deployment repository,
release lineage, validation suite, and documentation. [Josh Cartu (`jcartu`)](https://github.com/jcartu)
continued that work to integrate, repair, qualify, package, document, and publish dynamic,
fully-sharded LoRA serving.

The local integration commits use the Git author identity **Sisyphus**. In this release history,
that is Josh Cartu's implementation identity, not a separate upstream project. The complete record
of repository authors and changes remains available in the
[commit history](https://github.com/jcartu/glm52-exl3-sparkinfer/commits/main/) and
[contributor graph](https://github.com/jcartu/glm52-exl3-sparkinfer/graphs/contributors).

The release-specific contribution is the glue between existing layers: rank-local adapter loading,
MLA attention corrections, an adapter-aware EXL3 expert plan, staged Sparkinfer execution,
dynamic lifecycle behavior, reproducible container construction, deployment presets, hardware
qualification, rollback, and this documentation. It does not transfer ownership of any upstream
model, checkpoint, format, engine, kernel, or library.

## 5. Supporting software and infrastructure

The container also relies directly or transitively on the following projects. Credit belongs to
each project's maintainers and complete contributor community:

- [NVIDIA CUDA](https://developer.nvidia.com/cuda-toolkit), [cuBLAS](https://developer.nvidia.com/cublas), [cuDNN](https://developer.nvidia.com/cudnn), [NCCL](https://github.com/NVIDIA/nccl), and [CUTLASS](https://github.com/NVIDIA/cutlass)
- [PyTorch](https://github.com/pytorch/pytorch/graphs/contributors) and [Triton](https://github.com/triton-lang/triton/graphs/contributors)
- [FlashInfer](https://github.com/flashinfer-ai/flashinfer/graphs/contributors)
- [DeepGEMM](https://github.com/deepseek-ai/DeepGEMM/graphs/contributors)
- [InstantTensor](https://github.com/scitix/InstantTensor/graphs/contributors)
- [TokenSpeed MLA](https://github.com/modal-projects/tokenspeed-mla/graphs/contributors)
- [TVM FFI](https://github.com/apache/tvm-ffi/graphs/contributors)
- [Hugging Face Transformers](https://github.com/huggingface/transformers/graphs/contributors), [PEFT](https://github.com/huggingface/peft/graphs/contributors), [Safetensors](https://github.com/huggingface/safetensors/graphs/contributors), and [Hugging Face Hub](https://github.com/huggingface/huggingface_hub/graphs/contributors)
- [Docker Engine](https://github.com/moby/moby/graphs/contributors), [BuildKit](https://github.com/moby/buildkit/graphs/contributors), and [Docker Compose](https://github.com/docker/compose/graphs/contributors)
- [GitHub](https://github.com/about), GitHub Actions, GitHub Container Registry, and all open-source Actions used for publication

This list explains the material runtime lineage; it is not a replacement for the package notices
and license files inside the container, which remain authoritative for transitive dependencies.

## 6. Review and AI-tool disclosure

Automated tools assisted people; they are not presented as owners or independent validators.

- [CodeRabbit](https://coderabbit.ai/) supplied automated review findings on the EXL3 integration pull requests.
- [OpenAI Codex](https://openai.com/codex/) and [Anthropic Claude Code](https://www.anthropic.com/claude-code) assisted implementation and review during the upstream EXL3 work, as disclosed on the pull requests.
- OpenAI Codex, operated through [Oh My Pi](https://github.com/can1357/oh-my-pi), assisted the dynamic-LoRA integration, provenance repair, documentation, and publication work.

Humans selected the architecture, accepted or rejected suggestions, controlled the source history,
and ran the reported tests on the physical four-GPU machine. AI involvement does not change the
licenses or ownership of the human-created projects above.

## 7. License boundaries

| Material | License or boundary | Source |
|---|---|---|
| This repository's original scripts, configuration, tests, and documentation | MIT; retain Brandon Music's and Josh Cartu's notices | [`LICENSE`](LICENSE) |
| GLM-5.2 model | MIT according to the official model card; model files are not stored here | [model card](https://huggingface.co/zai-org/GLM-5.2) |
| Brandon Music's EXL3-TR3 checkpoint | MIT according to its model card; checkpoint files are not stored here | [checkpoint card](https://huggingface.co/brandonmusic/GLM-5.2-EXL3-TR3-3.0bpw) |
| vLLM | Apache License 2.0 | [license](https://github.com/vllm-project/vllm/blob/main/LICENSE) |
| Sparkinfer | Apache License 2.0 | [license](https://github.com/local-inference-lab/sparkinfer/blob/main/LICENSE) |
| ExLlamaV3 | MIT, copyright Turboderp and contributors | [license](https://github.com/turboderp-org/exllamav3/blob/master/LICENSE) |
| Qualification adapter | Unknown from local metadata; not redistributed and no ownership claimed | user-supplied material only |
| NVIDIA libraries and every supporting dependency | Their own upstream terms; inclusion in one image does not relicense them | links in the section above and notices inside the image |

The repository's MIT license does **not** relicense the model, checkpoint, adapter, container base,
CUDA stack, native extensions, Python packages, or any other dependency. Anyone redistributing the
container or model files is responsible for preserving all applicable notices and satisfying every
upstream license.