# 运行环境锁定清单

本分支（`2080ti_dual_qwen38-27B`：上游 vLLM `main` + SM75 移植）运行服务的关键 Python/CUDA 包版本
（Intel Xeon E5-2696 v3 / 15 GiB 机器）。优先保持这些版本不变，直到先跑通基准。

~~~text
Python                 3.12.3
NVIDIA Driver          580.173.02
Driver CUDA Runtime    13.0
torch                  2.13.0+cu130
vllm                   0.26.1rc1.dev2278+g49f68ba24（editable，分支 2080ti_dual_qwen38-27B）
transformers           5.16.1
flashinfer-python      0.6.18.post1
triton                 3.7.1
numpy                  2.3.5
tokenizers             0.23.1
safetensors            0.8.0
xgrammar               0.2.3
tilelang               0.1.12
apache-tvm-ffi         0.1.11
nvidia-nccl-cu13       2.29.7
nvidia-cudnn-cu13      9.20.0.48
nvidia-cuda-runtime    13.0.96
nvidia-cublas          13.1.1.3
nvidia-cusparselt-cu13 0.8.1
~~~

从源码编译 vLLM 时额外依赖 pip 的 CUDA 工具链（`cuda-toolkit 13.0.3.0`），其中
`nvidia-cuda-nvcc` / `nvidia-nvvm` / `nvidia-cuda-crt` 需固定为 **13.0.88**（13.3 的
头文件与 torch cu130 不匹配）。构建侧的其余修补见
[upstream-branch.md](upstream-branch.md) §3。

`flash-qla` 的 `setup.py` 原先把上述两个包钉死在 `tilelang==0.1.8` /
`apache-tvm-ffi==0.1.9`，与本环境（0.1.12 / 0.1.11，与 flashinfer 0.6.18 共存）冲突；
已放宽为 `>=`，`pip check` 无冲突，见 [upstream-branch.md](upstream-branch.md) §3。

系统 nvcc 是否在 PATH 并不是 vLLM 运行的唯一判断条件；本环境依靠 PyTorch 的 CUDA 运行时。若从源码编译 vLLM、FlashInfer 或 FlashQLA，仍需要准备匹配的 CUDA 编译工具链。
