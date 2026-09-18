# 从干净上游源码构建补丁版环境

以下步骤锁定的是这套服务实测使用的提交。先在测试机验证，再替换生产服务。

两条路线：

- **基础部署（本项目原始路线）**：上游 vLLM `v0.27.1` + `patches/vllm-v0.27.1-sm75-qwen3.8.patch`。
- **本分支路线（`2080ti_dual_qwen38-27B`）**：上游 vLLM `main` + SM75/Qwen3.8 移植提交（9 文件）。
  开分支、移植范围、编译环境修补（CUDA 工具链、`.deps` 路径、cutlass）与部署记录见
  [upstream-branch.md](upstream-branch.md)。

## vLLM（上游 v0.27.1 + 本仓库 patch）

~~~bash
git clone https://github.com/vllm-project/vllm.git
cd vllm
git checkout v0.27.1
git apply /path/to/qwen3-8-27b-dual-2080ti-vllm/patches/vllm-v0.27.1-sm75-qwen3.8.patch
python -m pip install -U pip
python -m pip install -e .
~~~

`vllm-v0.27.1-sm75-qwen3.8.patch` 即 SM75 / Qwen3.8 部署改动（FlashQLA legacy GDN
prefill、Qwen3.5 MTP、SM75 spec-decode 同步、FlashInfer 的 SM75 支持判定等）。

本次验证环境：Python 3.12.3 + PyTorch 2.13.0+cu130。

### 源码内构建（改用上游 main 时）

从上游 `main` 起步时不要直接套用上面那份 patch（它基于 v0.27.1），而是按
[upstream-branch.md](upstream-branch.md) §2 的移植清单把同样的 9 处改动落到目标提交上，
并按 §3 修补编译环境。

## FlashQLA-SM70-SM75

~~~bash
git clone https://github.com/weicj/FlashQLA-SM70-SM75.git
cd FlashQLA-SM70-SM75
git checkout 3ab27d77d8ca01d7a4718903b726add1a8886c0e
git apply /path/to/qwen3-8-27b-dual-2080ti-vllm/patches/flashqla-sm70-sm75-local.patch
python -m pip install -e .
~~~

运行服务前，确保 FLASHQLA_PATH 指向该目录；启动脚本会把它加入 PYTHONPATH。

## FlashInfer

基础部署使用 `flashinfer-python==0.6.16.post3`；本分支（上游 main）实测使用
`0.6.18.post1`。SM75 对 FlashInfer/vLLM 版本较敏感，升级 FlashInfer、vLLM、CUDA、驱动
或模型后，都要重新测试短文本、长上下文、工具调用和显存上限。
