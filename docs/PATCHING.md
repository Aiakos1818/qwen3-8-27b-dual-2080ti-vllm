# 从干净上游源码构建补丁版环境

以下步骤锁定的是这套服务实测使用的提交。先在测试机验证，再替换生产服务。

## 方式 A：直接用已推送的 fork 分支（最省事）

本 fork 的改动已推送为可直接克隆的分支：

~~~bash
# vLLM：base(sm75/qwen3.8) + KV 优化两个提交，基于 v0.27.1
git clone -b sm75-qwen3.8-kv https://github.com/Aiakos1818/vllm.git
cd vllm
python -m pip install -U pip
python -m pip install -e .

# FlashQLA-SM70-SM75：SM75 本地改动
git clone -b sm75-qwen3.8 https://github.com/Aiakos1818/FlashQLA-SM70-SM75.git
cd FlashQLA-SM70-SM75
python -m pip install -e .
~~~

- `Aiakos1818/vllm@sm75-qwen3.8-kv`（Apache-2.0）
- `Aiakos1818/FlashQLA-SM70-SM75@sm75-qwen3.8`（MIT）

## 方式 B：从上游 + 本仓库 patch 构建

### vLLM

~~~bash
git clone https://github.com/vllm-project/vllm.git
cd vllm
git checkout v0.27.1
git apply /path/to/qwen3-8-27b-dual-2080ti-vllm/patches/vllm-v0.27.1-sm75-qwen3.8.patch
git apply /path/to/qwen3-8-27b-dual-2080ti-vllm/patches/vllm-v0.27.1-kv-offload-2080ti.patch
python -m pip install -U pip
python -m pip install -e .
~~~

`vllm-v0.27.1-kv-offload-2080ti.patch` 是 **KV 优化补丁**（会话保活、Mamba 锚点、
GPU↔RAM/SSD 分层 offload 与分块流式、指标面板），必须**在基础补丁之后**应用。
设计与实测见 `docs/kv-optimization/`。

该补丁同时携带 3 个从上游 v0.28/v0.29 手工移植的 mamba/GDN 修复（#51812 投机解码
gate 对齐、#56196 短 prefill chunk 的 conv state 落块、#49436 state-copy Triton 3D
tiling）。上游 #52789 只对 Kimi-K3 KDA 生效、#51674/#52539 需要 sm80+，均未移植；
详见 `docs/kv-optimization/README.md` 的「上游同步」一节。

本次验证环境：Python 3.12.3 + PyTorch 2.13.0+cu130。

### FlashQLA-SM70-SM75

~~~bash
git clone https://github.com/weicj/FlashQLA-SM70-SM75.git
cd FlashQLA-SM70-SM75
git checkout 3ab27d77d8ca01d7a4718903b726add1a8886c0e
git apply /path/to/qwen3-8-27b-dual-2080ti-vllm/patches/flashqla-sm70-sm75-local.patch
python -m pip install -e .
~~~

运行服务前，确保 FLASHQLA_PATH 指向该目录；启动脚本会把它加入 PYTHONPATH。

## FlashInfer

本次运行环境使用 flashinfer-python==0.6.16.post3。SM75 对 FlashInfer/vLLM 版本较敏感，升级 FlashInfer、vLLM、CUDA、驱动或模型后，都要重新测试短文本、长上下文、工具调用和显存上限。
