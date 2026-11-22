# 当前 PD 部署

这份说明对的是 2026-11-22 下午仍在服务的这一套，不是 `scripts/launch_pd_*.sh` 里的旧默认值。那些脚本还在，但正在跑的进程没有走它们。

两边都以用户 `aaron` 跑，用户 systemd 的运行目录是 `/run/user/1003`。Python 是 `/home/aaron/pd/.venv-sglang`。`PYTHONPATH` 把 `/home/aaron/pd/sglang/python` 放在前面，所以改 checkout 里的文件会在下次启动时生效。已经起来的进程不会重新读磁盘上的 `.py`。

## 拓扑

| 角色 | 机器 | 进程入口 | 监听 |
|---|---|---|---|
| Prefill | s109，`10.230.0.109` / RDMA `192.168.100.109` | `pd_run/start_prefill_block3.sh` | `0.0.0.0:30000`，bootstrap `8998` |
| Decode | s107，`10.230.0.107` / RDMA `192.168.100.107` | `pd_run/start_decode_pcie.sh` | `0.0.0.0:30001` |
| Router | s107 | `/home/aaron/pd/scripts/launch_pd_router.sh` | 本机 `127.0.0.1:8001` |
| 拒绝代理 | s107，由 router 脚本拉起 | `/home/aaron/pd/scripts/reject_192_168_105_1.py` | `0.0.0.0:8000`，转发到 `127.0.0.1:8001`，拒绝 `192.168.105.1` |

模型是 DeepSeek-V4.1-Flash-1017-NVFP4。Prefill 读 `/data/models/nvidia/DeepSeek-V4.1-Flash-1017-NVFP4`，decode 读 `/data/aaron/models/nvidia--DeepSeek-V4.1-Flash-1017-NVFP4`。两边都是 TP8、EP8、attention DP 1、DSpark block 3、同一份权重当 draft。Prefill 只跑 encoder 层（环境变量 `SGLANG_DSV4_ENCODER_ONLY_PREFILL=1`，层 0–19），decode 跑 0–39。KV 用 Mooncake RDMA。Prefill 网卡 `rocep37s0f0`，decode 网卡 `mlx5_0`，`MC_GID_INDEX=3`。

显卡：prefill 是 RTX PRO 5000 72GB ×8，decode 是 RTX 6000D ×8。两边 `mem-fraction-static` 都是 0.82，`max-running-requests` 64，上下文 1M。Prefill 的 chunk 是 4096。Decode 启动参数里的 chunk 是 8192；3k/1k 和 32k 的实测都是 prefill 这一侧的 4096 在切请求。

本次对外提交的 40 点在 ardelva.ai 已核验记录 #22。

可复现的入口在 sglang 仓库的 `pd_run/`，不要用 `scripts/launch_pd_*.sh` 的默认值。检出包含本文件的那次提交后：

1. s107 直接用仓库里的 `pd_run/start_decode_pcie.sh`。它读 `pd_run/decode-env.sh`，GEMM tile 已经在这棵树的 `configs/` 里。Decode 用仓库里的 `conn.py` 和 `kv_wait_clock.py`。
2. s109 在启动前把 `pd_run/live-prefill-conn.py` 覆盖到 `python/sglang/srt/disaggregation/common/conn.py`。2026-11-22 这次实测的 prefill 加载的是这一份，里面没有 `kv_wait_clock.py`。不要在 s107 上做这个覆盖。
3. 然后按下面的 systemd 命令启动。`/tmp/start_*.sh` 是这次进程实际的入口，内容和 `pd_run/` 里的服务命令一致。

`bak-pre-m48/` 和 `freeze_replay.py` 不在这次提交里，复现时不需要。

## 正在跑的进程

2026-11-22 下午核对时：

- Prefill：`pd-prefill-block3.service`，05:18 由 `/tmp/start_prefill_block3.sh` 拉起。日志指针在 `/tmp/prefill-block3.logpath`，文件在 `/home/aaron/pd/logs/pd-prefill-109-block3-*.log`。
- Decode：`pd-decode-pcie.service`，04:53 由 `/tmp/start_decode_pcie.sh` 拉起。日志指针在 `/tmp/decode-pcieipc.logpath`，文件在 `/home/aaron/pd/logs/pd-decode-107-pcieipc-*.log`。
- Router：`pd-router.service`，11 月 21 日 23:42 起，还在跑。

这三个都是 `systemd-run --user` 的临时单元，`CollectMode=inactive-or-failed`。`systemctl --user stop` 之后单元定义会消失，需要按下面的命令重新创建，不能只 `start`。

## 启动顺序

先 prefill 和 decode，两边 `/health` 都是 200 之后再看 router。Router 如果还在，不要为了重启 worker 去动它。

在 s109 上：

```bash
export XDG_RUNTIME_DIR=/run/user/1003
cp /home/aaron/pd/sglang/pd_run/live-prefill-conn.py \
  /home/aaron/pd/sglang/python/sglang/srt/disaggregation/common/conn.py
systemctl --user stop pd-prefill-block3.service
systemd-run --user --unit pd-prefill-block3 --collect \
  /home/aaron/pd/sglang/pd_run/start_prefill_block3.sh
```

在 s107 上：

```bash
export XDG_RUNTIME_DIR=/run/user/1003
systemctl --user stop pd-decode-pcie.service
systemd-run --user --unit pd-decode-pcie --collect \
  /home/aaron/pd/sglang/pd_run/start_decode_pcie.sh
```

Router 只在它没在跑的时候重建：

```bash
export XDG_RUNTIME_DIR=/run/user/1003
systemd-run --user --unit pd-router --collect \
  /usr/bin/bash -lc /home/aaron/pd/scripts/launch_pd_router.sh
```

健康检查：

```bash
curl -s -o /dev/null -w "prefill:%{http_code}\n" --max-time 5 http://10.230.0.109:30000/health
curl -s -o /dev/null -w "decode:%{http_code}\n"  --max-time 5 http://127.0.0.1:30001/health
curl -s -o /dev/null -w "router:%{http_code}\n"  --max-time 5 http://127.0.0.1:8001/health
```

对外入口是 s107 的 `:8000`。本机压测打 `http://127.0.0.1:8000` 也会进拒绝代理，再转到 `127.0.0.1:8001`。

## 两个启动脚本实际做了什么

`/tmp/start_prefill_block3.sh` 自己写死环境，再 exec `sglang serve`。关键项：

- `SGLANG_DSV4_ENCODER_ONLY_PREFILL=1`
- `CHUNKED_PREFILL=4096`，命令行 `--chunked-prefill-size 4096`
- `MOONCAKE_PROTOCOL=rdma`，`MOONCAKE_DEVICE=rocep37s0f0`，`SGLANG_HOST_IP=192.168.100.109`
- `MC_MAX_MR_SIZE=134217728`，`MC_RDMA_DATA_DIRECT=0`，`MC_GID_INDEX=3`
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False`
- `SGLANG_ENABLE_JIT_DEEPGEMM=1`，`SGLANG_RAGGED_VERIFY_MODE=static`
- `--dsv4-prefill-backend flashmla_sparse`，`--disable-prefill-cuda-graph`
- `--attention-backend dsv4`，`--moe-runner-backend flashinfer_cutlass`
- `--speculative-algorithm DSPARK`，`--speculative-dspark-block-size 3`
- `--schedule-conservativeness 0.3`，`--cuda-graph-max-bs-decode 64`
- `--disaggregation-mode prefill`，`--disaggregation-transfer-backend mooncake`

`/tmp/start_decode_pcie.sh` 先 `source /tmp/decode-env.txt`，再 exec。命令行和 prefill 的差别：

- `--disaggregation-mode decode`，`--disaggregation-ib-device mlx5_0`，端口 30001
- `--chunked-prefill-size 8192`
- 没有 encoder-only，没有 `--disable-prefill-cuda-graph`，没有 `--dsv4-prefill-backend`
- `/tmp/decode-env.txt` 里有 `SGLANG_ENABLE_PCIE_IPC_ALLREDUCE=1`，`SGLANG_HOST_IP=192.168.100.107`，`MOONCAKE_DEVICE=mlx5_0`
- 同一份 env 里：`SGLANG_DSV41_SPAN_BUCKETS=0`，`SGLANG_DSV41_SHORT_COLS=1`，`SGLANG_OPT_DSV41_TARGET_VERIFY_METADATA_GLUE=0`，`SGLANG_OPT_DSV41_TARGET_METADATA_EARLY=0`，`SGLANG_DSV4_DECODER_TAIL_GRAPH_MAX_ROWS=0`

`/home/aaron/pd/scripts/launch_pd_prefill_109.sh` 和 `launch_pd_decode_107.sh` 是更早的封装。它们的默认值不是现在这套：prefill 脚本默认 chunk 32768、attention DP 8；decode 脚本默认 mem 0.72、chunk 4096、attention DP 8。旁边的 `*.bak*` 是改这些默认值时留下的，不要拿来启动。

`sglang/pd_run/` 里有一份更短的 `launch_prefill_109.sh`、`launch_decode_107.sh`、`launch_pd_router.sh`、`reject_192_168_105_1.py`。Router 那份和 `scripts/launch_pd_router.sh` 是同一类入口。当前单元没有调用 `pd_run/launch_*.sh`。

## 和这套部署有关的代码

两边 checkout 都是分支 `host-v0523-sm120-engram`，但不是同一个提交。

s107 decode 的 HEAD 是 `982054a6a6`（比 origin 超前 1 个提交，还没 push）。提交说明是 SM120 decode path，以及 router 拒绝 `192.168.105.1`。这个进程 04:53 启动，用的就是这个 HEAD，再加上下面那些 01:18 改过的 GEMM JSON。

s109 prefill 的 HEAD 是 `e512d6816d`，比 origin 落后 3 个提交。进程 05:18 启动，加载的是这个工作区，不是 s107 的 HEAD。

下面这些 `.py` 两边内容相同，都已经被对应进程加载。s107 上它们在已跟踪的历史里；s109 上除了 `pd_stage.py` 显示为未跟踪，其余是已跟踪但未提交的改动。内容一样，只是 git 状态不一样：

| 文件 | 在这套部署里做什么 |
|---|---|
| `python/sglang/srt/disaggregation/prefill.py` | prefill 结束和 KV 发布完成时打 `PDSTAGE` |
| `python/sglang/srt/disaggregation/decode.py` | decode 第一次 KV ready 时打 `PDSTAGE` |
| `python/sglang/srt/managers/scheduler.py` | 启动时打一行 import 标记；prefill 请求第一次进 forward 时记录排队时间 |
| `python/sglang/srt/managers/scheduler_components/batch_result_processor.py` | decode 侧记录首 token |
| `python/sglang/srt/observability/pd_stage.py` | 上面这些日志的实现。s109 上是未跟踪文件，s107 上已跟踪。两边 md5 相同 |

`conn.py` 两边不一样，这是当前部署里唯一一份分叉的传输代码：

- s107 `python/sglang/srt/disaggregation/common/conn.py` 配 `kv_wait_clock.py`。在未推送的 `982054a6a6` 里。300 秒传输时钟从最后一块 KV 的通知起算，不从 metadata 发布时间起算。Decode 04:53 加载的是这一份。
- s109 的 `conn.py` md5 不同，没有 `kv_wait_clock.py`。early allocation 的 deadline 逻辑直接写在这个文件里。Prefill 05:18 加载的是这一份。

s107 那个未推送提交里、decode 进程会导入的其他文件：

- `python/sglang/srt/environ.py`
- `python/sglang/srt/layers/attention/dsv4/v41_indexer/scoring.py`
- `python/sglang/srt/layers/attention/graph_variants.py`
- `python/sglang/srt/model_executor/graph_deps_diag.py`
- `python/sglang/srt/model_executor/runner/decode_cuda_graph_runner.py`
- `python/sglang/srt/speculative/dspark_components/dspark_verify.py`
- `pd_run/launch_pd_router.sh`
- `pd_run/reject_192_168_105_1.py`

## GEMM tile

这 10 个文件在 s107 上是已跟踪、未提交的改动，mtime 2026-11-22 01:18，早于 decode 的 04:53 启动。`fp8_kernel.py` 的 `get_w8a8_block_fp8_configs` 用 `configs/` 加精确文件名打开，所以 decode 用的就是它们。文件名里的设备是 `NVIDIA_RTX_6000D`，prefill 的 RTX PRO 5000 不会读这组。

目录：`/home/aaron/pd/sglang/python/sglang/kernels/ops/gemm/configs/`

- `N=512,K=5120,device_name=NVIDIA_RTX_6000D,dtype=fp8_w8a8,block_shape=[32, 32].json`
- `N=576,K=5120,...`
- `N=1280,K=5120,...`
- `N=1536,K=5120,...`
- `N=1792,K=5120,...`
- `N=4096,K=1280,...`
- `N=5120,K=288,...`
- `N=5120,K=1024,...`
- `N=5120,K=15360,...`
- `N=25600,K=6144,...`

改的是 M 档上的 `BLOCK_SIZE_M` / `BLOCK_SIZE_N`，以及部分档的 `SWAP_AB`、`SPLIT_K`。这是保留下来的 M=256 FP8 tile，不是后来撤回的 tactic 15 或 2-block ring。

加载器不扫子目录。`configs/bak-pre-m48/` 是未跟踪备份，里面 4 个旧 JSON，运行时不会读。

Decode 还有两份运行时缓存，不在 git 里，但 04:53 这次启动用过：

- FlashInfer autotune：`/home/aaron/.cache/sglang/flashinfer/autotune/0.7.2`
- PCIe IPC JIT：`/home/aaron/.cache/flashinfer/0.7.2/120f/cached_ops/pcie_ipc_comm`

PCIe allreduce 在超过 2 张、且只有 PCIe 没有 NVLink 的 GPU 上不会走 CustomAllReduceV2。当前 decode 走的是 FlashInfer 的 PCIe IPC staged ring（variant 2，4 个 block，512 线程）。不要把已经撤回的 mHC epilogue 再编进这个 JIT 缓存。

## 未跟踪、但和这套部署无关

- s107 `python/sglang/kernels/ops/gemm/configs/bak-pre-m48/`：备份，见上。
- s109 `python/sglang/srt/layers/attention/dsv4/v41_indexer/freeze_replay.py`：701 行。文件写明只有设置 `SGLANG_OPT_DSV41_INDEXER_FREEZE_DIR` 才会被导入。正在跑的 prefill 进程没有这个环境变量，scheduler 也没有 import 它。

## 压测从哪打

压测机是 s107。解释器 `/home/aaron/pd/.venv-sglang/bin/python`，驱动 `/home/aaron/pd/bench/ardelva_bench_pd.py`。`--suite` 跑 40 个点，不会自动 `--post`。官方核验 key 在 `/home/aaron/pd/bench/.ardelva-official.key`（`~/.config/ardelva/official.key` 是同一把）。合并后的 40 点 CSV 和本地预览在 `/home/aaron/pd/bench/runs/suite-chunk4096-20261122-0523/`。

`bench/` 本身不是 git 仓库。
