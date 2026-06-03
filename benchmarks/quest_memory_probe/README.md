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

## 三个关键字段

- **`vllm.gpu_kv_useful_bytes`** — KV 池里真正承载有效 KV 数据的字节数。
  Quest 卸载越激烈这个越小。
- **`vllm.kv_pool_slack_bytes`** — vLLM 预留池里没被占用的部分。
  Dense ≈ 0；Quest+offload 应显著 > 0。
- **`quest.topk_hit_ratio`** — 每次 selection 时 top-k 已驻留 GPU 的比例。
  池越小 → 命中率越低 → H2D 越频繁。

## 手动验证 checklist

跑完一次 `compare-pool-size` 后看 `kv_pool_breakdown.png`：

- [ ] dense cfg 的 `kv_slack` ≈ 0（vLLM 默认把池占满）
- [ ] quest pool=16 的 `kv_slack` 显著 > 0（卸载工程价值的实证）
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
