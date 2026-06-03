# Quest 显存观测实验工具

观测单请求 / 单 batch 推理时三层显存（NVML / PyTorch / Quest 池内驻留），
量化 Quest+CPU 卸载相对 Dense 节省的"实际使用显存"。

设计文档：`docs/superpowers/specs/2026-06-01-quest-memory-probe-design.md`

## 前置环境

- 一定要 venv（不要 conda env）
- env vars: `HF_HUB_OFFLINE=1 PATH="$PWD/.venv/bin:$PATH"`
- GPU 几乎全空（Quest 配置约吃 22-25 GiB）。建议用 `CUDA_VISIBLE_DEVICES=N` 选一张干净的卡
- HF cache 已有 `meta-llama/Llama-3.2-3B-Instruct`

## 三个子命令

### A. Dense vs Quest

```bash
HF_HUB_OFFLINE=1 PATH="$PWD/.venv/bin:$PATH" \
    .venv/bin/python -m benchmarks.quest_memory_probe \
    compare-dense-vs-quest \
    --samples longbench:narrativeqa:lengths=short,medium:n=2 \
    --top-k 16 \
    --quest-pool 128 \
    --out-dir results/$(date +%Y%m%d-%H%M%S)
```

### B. Pool size sweep（看卸载激烈程度对显存的影响）

```bash
HF_HUB_OFFLINE=1 PATH="$PWD/.venv/bin:$PATH" \
    .venv/bin/python -m benchmarks.quest_memory_probe \
    compare-pool-size \
    --samples longbench:narrativeqa:lengths=short,medium:n=2 \
    --top-k 16 \
    --pool-sizes 512,256,128,32,16 \
    --out-dir results/$(date +%Y%m%d-%H%M%S)
```

### C. OOM threshold sweep

```bash
HF_HUB_OFFLINE=1 PATH="$PWD/.venv/bin:$PATH" \
    .venv/bin/python -m benchmarks.quest_memory_probe \
    oom-sweep \
    --samples longbench:narrativeqa:lengths=short,medium,long:n=4 \
    --top-k 16 \
    --quest-pool 128 \
    --out-dir results/$(date +%Y%m%d-%H%M%S)
```

## 强制 synthetic 模式

如果机器没法访问 LongBench-v2 数据集，加 env var：

```bash
QUEST_MEM_PROBE_FORCE_SYNTHETIC=1 ... 同上 ...
```

合成 prompt 用 10 段不同主题文本拼成目标长度，确定性可复现。

## 输出产物

```text
out_dir/
├── manifest.json                  # CLI 参数 + commit
├── <cfg>/
│   ├── samples.csv                # 时间序列采样
│   ├── summary.json               # 样本级聚合
│   └── stdout.log
├── plots/
│   ├── memory_timeline_<cfg>.png  # 三层堆叠时间序列（主图）
│   ├── memory_peak_bar.png        # 跨 cfg 峰值柱
│   ├── kv_pool_breakdown.png      # 工程价值的可视化（kv_useful vs slack）
│   ├── topk_hit_ratio_<cfg>.png   # 仅 quest cfg
│   └── oom_threshold.png          # 仅 oom-sweep
└── report.md
```

## 显存分布与关键字段

Stage 2A+ 起 Quest arena 是私有 `torch.empty` 分配（见
`vllm/v1/attention/backends/quest/backend.py:305,327`），与 vLLM engine
预留的 KV 池是**两块独立显存**：

```text
torch.allocated_bytes
  = weights + workspace          ← vllm.engine_essential_bytes
  + engine_kv_pool_total         ← vllm.kv_pool_total_bytes
  + quest_arena_total            ← quest.arena_total_bytes (Quest 模式)
```

- **`vllm.actual_used_bytes`** = `essential + arena_total`（Quest）
  / `essential + kv_useful`（Dense）。
  这是"vLLM 进程实际持有的、必须的 GPU 显存"——含 weights、workspace、
  以及 Quest 自己用 `torch.empty` 申请来给 decode 用的整块 KV 缓存。
- **`vllm.actual_used_peak_bytes`** — 同上，但 essential 部分用
  `torch.peak_allocated_bytes`，能反映 prefill spike 期间的最高水位。
- **`quest.arena_total_bytes`** — Quest 私有 arena 字节总和（K+V，跨所有
  Quest 层）。常量，由 `gpu_cache_blocks_per_seq × bpb × num_quest_layers`
  决定。
- **`vllm.gpu_kv_useful_bytes`** (= arena_resident in Quest mode)
  — arena 里被某个 (seq, block) 占着的 slot 字节数。辅助观察 arena 内部
  占用率，**不进入 actual_used**。
- **`quest.topk_hit_ratio`** — 每次 selection 时 top-k 已驻留 GPU 的比例。
  池越小 → 命中率越低 → H2D 越频繁。

### 采样时机限制

当前 probe 在每个样本窗口里只在 `sample_start` 之前 + `generate()` 之后
各采一次（`runner.py:163,193,202`），不是周期采样——因为 worker 的
collective_rpc 输入 socket 与 engine 请求流共用，并发会让 IPC frame 崩。
所以：

- `peak_actual_used_bytes`（窗口取 max，口径 A）= 这两个采样点的较大者，
  **不会捕到 prefill 中段峰**。
- `peak_actual_used_peak_bytes`（基于 `torch.peak_allocated_bytes`，
  口径 B）= 真实的 generate 期间最高水位，含 prefill 峰。runner 在每个
  sample 前调 `reset_peak_stats`，所以这个值就是该样本期间的真峰。

要看真峰用口径 B；口径 A 留作对照（generate 前后 settled 状态）。

## 手动验证 checklist

跑完一次 `compare-pool-size` 后：

- [ ] dense cfg 的 `actual_used_bytes` ≈ `weights + small workspace +
      kv_useful`，应略大于 `weights_bytes`
- [ ] quest cfg 随 `gpu_cache_blocks_per_seq` 缩小，`actual_used_bytes`
      单调下降，但不低于 `weights_bytes`
- [ ] `actual_used_peak_bytes` ≥ `actual_used_bytes`（峰值 ≥ settled）
- [ ] top-k 命中率随 pool 缩小而下降的趋势是平滑的，不应突然跌到 0

如果观察到的是反的（pool 缩小后 slack 没增加），先排查：

1. 检查 `<cfg>/stdout.log` 看 `[runner] discovered N TierManager(s)`，
   `N` 应该 = quest 配置下的非 full_kv layer 数（Llama-3.2-3B 是 26）。
   如果是 0，说明 `_collect_tier_managers` fallback 路径没找到 layer。
2. 检查 `samples.csv` 里 `quest.evict_d2h` / `quest.load_h2d` 是不是有值；
   全 0 说明卸载没真正发生（可能是 `gpu_pool_aliases_kv_cache=True` 那条
   分支，详见 `vllm/v1/attention/backends/quest/cache/tier_manager.py:230` 注释）。

## 跑 e2e smoke（GPU 必需）

```bash
CUDA_VISIBLE_DEVICES=1 HF_HUB_OFFLINE=1 PATH="$PWD/.venv/bin:$PATH" \
    .venv/bin/python -m pytest \
    tests/v1/attention/quest/test_memory_probe.py::TestE2ESmoke -v
```

需要 ~1 张 ≥40 GiB GPU、~1 分钟。

## 常见错误

| 症状 | 原因 | 修复 |
| --- | --- | --- |
| `ModuleNotFoundError: vllm._C` | 用了 conda env 而非 venv | 用 `.venv/bin/python` |
| `ninja: not found` 在第一次 forward 时 | PATH 没含 `.venv/bin` | export PATH 见前置 |
| 卡 5 分钟然后 timeout | HF_HUB_OFFLINE 没设 | export 之 |
| 跨 config OOM | EngineCore CUDA leak | 已经用 spawn 子进程隔离；如果还出，加 `--probe-interval-ms 500` 减负 |
| `Engine core initialization failed` 于 cpu_cache | RunConfig.cpu_cache_blocks/gib 都是 0 | 用默认值（8192/8）或显式给非零 |
