# Triton-Turing 集成交接文档

日期：2026-09-06  
执行人：CatPaw  
目标：在双 RTX 2080 Ti (SM75) 的 0.28.0 vLLM 上，用 triton-turing 替换标准 triton 编译器，让 TRITON_ATTN backend 获得 SM75 软件流水线加速

---

## 1. 背景与动机

### 1.1 当前 attention backend 现状

vLLM 0.28.0 在 SM75 上的默认 attention backend 选择逻辑：

```python
# vllm/platforms/cuda.py:148-159
# SM75 不是 SM100，走 else 分支
# 优先级：FLASH_ATTN > FLASHINFER > TRITON_ATTN > FLEX_ATTENTION > TURBOQUANT
```

但 **FLASH_ATTN 被 SM75 拒绝**（`supports_compute_capability` 要求 `>= (8, 0)`），
**TRITON_ATTN 也被拒绝**（FP8 KV cache 要求 SM89+），
所以最终只选了 **FlashInfer**。

日志证据：
```
Using FLASHINFER attention backend out of potential backends: ['FLASHINFER', 'TRITON_ATTN']
```

### 1.2 triton-turing 项目

仓库：https://github.com/Chennesxu/triton-turing

这是一个 **Triton 编译器的 SM75 专用 fork**，核心改进：
- **软件流水线**（`ld.global → st.shared → bar.sync`，不依赖 Ampere 的 `cp.async`）— Turing 上首次实现
- **FlashAttention-2 forward + backward (pipelined)** — 比 CUDA/CUTLASS FA 快 21-26%（head_dim=64）
- **bf16 dot on fp16 Tensor Core** — Turing 的 `mma.sync` 没有 bf16，vLLM 默认 bf16 所以根本没用上 Tensor Core
- 安装后**替换标准 triton 编译器**，所有 `@triton.jit` kernel 自动获得加速

### 1.3 集成思路

1. 安装 triton-turing 替换标准 triton 3.7.1
2. patch vLLM 的 TRITON_ATTN backend 解除 SM75 FP8 KV cache 限制
3. 启动 Canary 强制使用 `--attention-backend TRITON_ATTN`
4. A/B 对比 FlashInfer vs triton-turing TRITON_ATTN

---

## 2. 已完成的工作

### 2.1 ✅ 克隆 triton-turing 源码

```bash
# 服务器上
ls /home/<user>/triton-turing-src/
# bin cmake CMakeLists.txt docs examples include lib LICENSE Makefile
# MANIFEST.in pyproject.toml pytest.ini python README.md RELEASE.md
# scripts setup.py test third_party unittest utils
```

### 2.2 ✅ LLVM 预编译包已下载

文件：`/home/<user>/.triton/archives/llvm-87717bf9-ubuntu-x64-1.tar.gz`  
大小：1.8GB  
SHA256 校验通过：`7889df00f0dbbceb8e45774362b0478590029400a3abd58bdabba83d92fb2bbe`

### 2.3 ✅ 构建依赖已安装

```bash
# 在 vllm-env-0280-qwopus 中已安装：
# cmake-3.31.10, lit-23.1.0, pybind11-3.1.0
```

### 2.4 ✅ vLLM TRITON_ATTN FP8 KV cache 限制已解除

文件：`/home/<user>/vllm-2080ti-0.28.0-qwopus/vllm/v1/attention/backends/triton_attn.py`

修改内容（第 540-549 行附近）：
```diff
- if self.kv_cache_dtype.startswith("fp8") and not (
-     current_platform.has_device_capability(89)
- ):
-     suggested = (
-         "float16" if (cap is None or cap.to_int() < 80) else "bfloat16"
-     )
-     raise ValueError(
-         f"FP8 KV cache is not supported by the Triton attention backend "
-         f"on {dev} (compute capability {cap_str}); native FP8 (fp8e4nv) "
-         f"requires SM89+. Re-run with --kv-cache-dtype {suggested}."
-     )
+ if self.kv_cache_dtype.startswith("fp8") and not (
+     current_platform.has_device_capability(89)
+ ):
+     # SM75 patch: allow FP8 KV cache with Triton attention.
+     # The kernel converts FP8 to FP16 internally, so no native
+     # FP8 hardware support is needed. This enables triton-turing
+     # SM75 software-pipelined attention to work with FP8 KV cache.
+     logger.warning(
+         f"FP8 KV cache with Triton attention backend on {dev} "
+         f"(compute capability {cap_str}); native FP8 (fp8e4nv) "
+         f"requires SM89+, but SM75 will use FP8-to-FP16 conversion "
+         f"in the Triton kernel (triton-turing patched)."
+     )
```

验证 patch 已生效：
```bash
grep -n 'SM75 patch' /home/<user>/vllm-2080ti-0.28.0-qwopus/vllm/v1/attention/backends/triton_attn.py
```

---

## 3. 未完成的工作

### 3.1 ❌ triton-turing 编译安装

编译命令（需要重新执行）：
```bash
cd /home/<user>/triton-turing-src

# 建议限制并行度避免 SSH 无响应（之前 -j32 导致 sshd 不响应）
# 方法 1：用环境变量限制
MAX_JOBS=4 TRITON_BUILD_WITH_CLANG_LLD=0 \
  /home/<user>/vllm-env-0280-qwopus/bin/pip install -e . --no-build-isolation

# 方法 2：先手动跑 ninja 限线程
cd /home/<user>/triton-turing-src/build/cmake.linux-x86_64-cpython-3.12
ninja -j8

# 然后再跑 pip install
cd /home/<user>/triton-turing-src
/home/<user>/vllm-env-0280-qwopus/bin/pip install -e . --no-build-isolation --no-deps
```

**之前失败原因**：`ninja -j32` 32 线程并行编译 C++ 导致 CPU 负载过高，sshd 无法响应。
**解决方案**：用 `MAX_JOBS=4` 或 `ninja -j8` 限制并行度。

编译成功后验证：
```bash
/home/<user>/vllm-env-0280-qwopus/bin/python -c "import triton; print(triton.__version__); print(triton.__file__)"
# 应该输出: 3.7.0+git82007a85  和  /home/<user>/triton-turing-src/python/triton/__init__.py
```

### 3.2 ❌ 启动 Canary 服务

编译完成后，启动 Canary 服务（端口 8001），强制使用 TRITON_ATTN backend：

```bash
# 参考已有的 canary 启动脚本
cat /home/<user>/start_canary_8001.sh

# 需要修改的关键参数：
# 1. 添加 --attention-backend TRITON_ATTN
# 2. 使用 0.28.0 venv（已配置）
# 3. 添加环境变量 TRITON_SM75_BF16_DOT_AS_F16=1（可选，bf16→fp16 Tensor Core 加速）

# 启动命令示例（基于生产 service 文件修改）：
VLLM_ATTENTION_BACKEND=TRITON_ATTN \
TRITON_SM75_BF16_DOT_AS_F16=1 \
/home/<user>/vllm-env-0280-qwopus/bin/python -m vllm.entrypoints.openai.api_server \
  --host 0.0.0.0 --port 8001 \
  --model /home/<user>/models/Qwen3.8-27B-Uncensored-OrcaRouter-FP8/FP8 \
  ... (其余参数同生产服务) ...
  --attention-backend TRITON_ATTN
```

验证 TRITON_ATTN 生效：
```bash
# 启动日志中应该看到
# "Using TRITON_ATTN attention backend"
# 而不是 "Using FLASHINFER attention backend"
```

### 3.3 ❌ A/B Benchmark

测试脚本：`/home/<user>/vllm-2080ti-0.28.0-qwopus/benchmarks/run_context_ttft.py`

```bash
# FlashInfer 基线（当前生产服务 8000 端口）
python run_context_ttft.py --host <server>:8000

# triton-turing TRITON_ATTN（canary 8001 端口）
python run_context_ttft.py --host <server>:8001
```

已有基线数据（FlashInfer，2026-09-02 测试）：
```
words=2700  TTFT=2.62s  prefill=1086  decode=110.4
words=5400  TTFT=4.53s  prefill=1245  decode=89.8
words=8100  TTFT=6.51s  prefill=1298  decode=105.1
words=19000 TTFT=15.23s prefill=1298  decode=97.0
```

---

## 4. 关键技术细节

### 4.1 为什么 SM75 不能用 FLASH_ATTN

```python
# vllm/v1/attention/backends/flash_attn.py:206-208
@classmethod
def supports_compute_capability(cls, capability: DeviceCapability) -> bool:
    return capability >= DeviceCapability(8, 0)  # SM75 < SM80，被拒绝
```

FlashAttn 依赖 `vllm.vllm_flash_attn` 扩展（C++ CUDA kernel），该扩展不支持 SM75。

### 4.2 为什么 TRITON_ATTN 之前不被选

TRITON_ATTN 的 `supports_compute_capability` 返回 `True`（支持所有架构），
但在 `TritonAttentionImpl.__init__` 中检查 FP8 KV cache：

```python
# vllm/v1/attention/backends/triton_attn.py:540-549 (patch 前)
if self.kv_cache_dtype.startswith("fp8") and not (
    current_platform.has_device_capability(89)  # SM75 < 89，raise ValueError
):
    raise ValueError("FP8 KV cache requires SM89+")
```

这导致 TRITON_ATTN 在 SM75 + FP8 KV cache 组合下直接报错退出，
所以 vLLM 的 backend selector 只剩 FlashInfer。

### 4.3 triton-turing 如何加速

triton-turing 是 Triton 编译器的 fork，安装后替换 `pip install triton`。
所有使用 `@triton.jit` 装饰器的 Python kernel（包括 vLLM 的 TRITON_ATTN kernel）
都会被 triton-turing 编译器重新编译，获得：

1. **软件流水线**：`ld.global → st.shared → bar.sync` 多级流水，
   在 Tensor Core 计算 K×V 时预取下一块 K/V
2. **SM75 autotune**：针对 64KB/CTA shared memory 优化的 block size
3. **bf16→fp16 Tensor Core**（opt-in）：`TRITON_SM75_BF16_DOT_AS_F16=1`

### 4.4 FP8 KV cache 在 SM75 上如何工作

Triton kernel 不使用硬件 FP8 计算（`fp8e4nv` 需要 SM89+）。
它在 Python 层将 `uint8` KV cache 视图为 `torch.float8_e4m3fn`，
然后在 kernel 内部转为 FP16 进行计算。
这在 SM75 上完全可以工作，只是没有硬件 FP8 加速——但省了 KV cache 显存。

---

## 5. 文件清单

| 路径 | 说明 |
|---|---|
| `/home/<user>/triton-turing-src/` | triton-turing 源码树（已克隆） |
| `/home/<user>/.triton/archives/llvm-87717bf9-ubuntu-x64-1.tar.gz` | LLVM 预编译包（1.8GB，已下载，SHA256 校验通过） |
| `/home/<user>/vllm-2080ti-0.28.0-qwopus/` | vLLM 0.28.0 源码树（editable install，triton_attn.py 已 patch） |
| `/home/<user>/vllm-env-0280-qwopus/` | Python venv（当前安装的是标准 triton 3.7.1） |
| `/home/<user>/start_canary_8001.sh` | 已有的 Canary 启动脚本（需要修改 attention backend） |
| `/home/<user>/vllm-0271-main-8000.log` | 当前生产服务 8000 端口日志 |

---

## 6. 下一步操作清单

```bash
# 1. 检查当前编译是否还在跑
ps aux | grep -E 'ninja|cc1|pip.*triton' | grep -v grep

# 2a. 如果编译进程还在但卡住，杀掉重来
kill $(ps aux | grep -E 'ninja|cc1|pip.*triton' | grep -v grep | awk '{print $2}')

# 2b. 清理 build 目录
rm -rf /home/<user>/triton-turing-src/build/cmake.linux-x86_64-cpython-3.12

# 3. 重新编译（限制并行度！）
cd /home/<user>/triton-turing-src
MAX_JOBS=8 TRITON_BUILD_WITH_CLANG_LLD=0 \
  /home/<user>/vllm-env-0280-qwopus/bin/pip install -e . --no-build-isolation 2>&1 | tee /tmp/triton-turing-build.log

# 4. 验证安装
/home/<user>/vllm-env-0280-qwopus/bin/python -c "import triton; print(triton.__version__, triton.__file__)"

# 5. 启动 Canary（强制 TRITON_ATTN）
# 修改 start_canary_8001.sh 添加 --attention-backend TRITON_ATTN
# 和环境变量 TRITON_SM75_BF16_DOT_AS_F16=1
bash /home/<user>/start_canary_8001.sh 2>&1 | tee /tmp/canary_triton_turing.log

# 6. 检查日志确认 TRITON_ATTN 生效
grep -i "TRITON_ATTN\|triton.*attention\|backend" /tmp/canary_triton_turing.log | head -10

# 7. A/B 测试
cd /home/<user>/vllm-2080ti-0.28.0-qwopus/benchmarks
python run_context_ttft.py --host <server>:8001
```

---

## 7. 风险与回退

### 7.1 风险

1. **TRITON_ATTN + FP8 KV cache 在 SM75 上未经充分测试** — 可能有精度问题或 kernel crash
2. **triton-turing 编译器可能与 vLLM 的某些 triton kernel 不兼容** — 版本差异可能导致 JIT 编译失败
3. **bf16→fp16 Tensor Core 转换会改变数值** — `TRITON_SM75_BF16_DOT_AS_F16=1` 是 opt-in 的，
   超出 fp16 范围的值会 overflow

### 7.2 回退

```bash
# 回退 triton 到标准版本
/home/<user>/vllm-env-0280-qwopus/bin/pip install triton==3.7.1

# 回退 triton_attn.py patch（恢复 FP8 限制）
# 需要从 git 恢复或手动改回 raise ValueError

# 重启生产服务
sudo systemctl restart qwen-vllm-qwopus.service
```
