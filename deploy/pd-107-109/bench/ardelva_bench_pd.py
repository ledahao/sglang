#!/usr/bin/env python3
"""
ardelva_bench_pd.py — 本机 PD 环境适配版（SGLang Prefill@109 + Decode@107 + router:8000）。

相对上游 https://ardelva.ai/bench/ardelva_bench.py：
  1. 压测引擎改为 `python -m sglang.benchmark.serving`（无 `vllm bench serve`）
  2. **默认扫描矩阵与上游一致**（3k/60k/100k × 命中/冷启动），部署已支持 1M / C64
  3. 在上游并发列表末尾按新规格补上 64（短/中点）；长 Prefill 点保持上游 [1,8]
  4. 采样默认与上游对齐：min_requests=32、req_per_conc=4、warmup=8、seed=1、
     stop_ttft=3000、slo=ttft:3000,tpot:50
  5. 无外网：优先本地 ShareGPT JSON（~/pd/bench/data/ShareGPT_local_mini.json）跑
     dataset=random；否则回退 random-ids。命中路径用 generated-shared-prefix

快速跑：
  ~/pd/bench/run_pd_bench.sh --smoke

标准套件（与上游默认矩阵对齐，约 16 点）：
  ~/pd/bench/run_pd_bench.sh --out ~/pd/bench/runs/std1

自定义（语法同上游，命中写百分数或 cold）：
  ~/pd/bench/run_pd_bench.sh --matrix '3000:1000:0:1,8,32,64;60000:1000:95:1,8,32'
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

COLD = None  # hit=None → 完全冷启动

# 与上游 ardelva_bench.DEFAULT_MATRIX 对齐的物理标定矩阵；
# 新规格 max_running=64：在原 [1,8,32] 上补 64（100k 长 Prefill 仍保持 [1,8]）。
DEFAULT_MATRIX = [
    (3000, 1000, 0.0, [1, 8, 32, 64]),     # 短上下文：解码、MoE、通信
    # 60k×C64 易打爆 Prefill(72GB) indexer workspace；基线先到 C32
    (60000, 1000, 0.95, [1, 8]),       # Prefill池~1.9M：60k×32顶满，基线先 C1/C8       # 长上下文 + 前缀命中
    (100000, 1000, 0.0, [1, 8]),           # 长输入 Prefill
    (3000, 1000, COLD, [1, 8, 32, 64]),    # 完全冷启动
    (100000, 1000, COLD, [1, 8]),          # 冷启动长 Prefill
]

# 本地 ShareGPT（避免 HF 下载）；不存在则 random → random-ids
LOCAL_SHAREGPT = os.path.expanduser(
    '~/pd/bench/data/ShareGPT_local_mini.json'
)

PD_DEFAULTS = {
    'base_url': 'http://127.0.0.1:8000',
    'model': '/data/aaron/models/nvidia--DeepSeek-V4.1-Flash-1017-NVFP4',
    'tokenizer': '/data/aaron/models/nvidia--DeepSeek-V4.1-Flash-1017-NVFP4',
    'backend': 'sglang-oai-chat',
    'engine': 'SGLang 0.5.23 PD 1M/C64 (P@109 + D@107, mooncake_tcp)',
}

# 与上游 CSV 列兼容（后几列可选）
CSV_COLS = [
    'isl', 'osl', 'hit', 'c', 'nreq', 'out_tps', 'ttft_p50', 'tpot_p50',
    'ttft_p99', 'tpot_p99', 'total_tps', 'success', 'accept', 'ttft_t1', 'ttft_tn',
    'ds', 'rate', 'slo_ttft', 'slo_tpot', 'slo_pct', 'good_tps', 'goodput',
    'in_tps', 'after_ttft_tps', 'after_ttft_wave_tps',
    'ttft_p95', 'tpot_p95', 'itl_p95', 'itl_p99',
    'active_decode_est',
]
MAX_POINTS = 96


# ---------------------------------------------------------------------------
# matrix / scrape
# ---------------------------------------------------------------------------

def parse_matrix(s: str):
    """与上游一致：'isl:osl:hit:c1,c2;...'；hit 为百分数，或 cold。"""
    rows = []
    for part in s.split(';'):
        part = part.strip()
        if not part:
            continue
        isl, osl, hit, conc = (part.split(':') + ['', '', '', ''])[:4]
        if not isl or not osl or not conc:
            raise SystemExit(f'矩阵项格式错误：{part!r}（需要 isl:osl:hit:c1,c2）')
        hit_s = hit.strip().lower()
        if hit_s in ('', 'cold', 'none', '-'):
            h = COLD
        else:
            # 上游：float(hit)/100；兼容写成 0.95
            v = float(hit_s)
            h = v if 0.0 < v <= 1.0 else v / 100.0
            if v == 0.0:
                h = 0.0
        cs = [int(x) for x in conc.split(',') if x.strip()]
        rows.append((int(isl), int(osl), h, cs))
    return rows


def scrape(base_url: str) -> dict:
    try:
        req = urllib.request.Request(base_url.rstrip('/') + '/metrics')
        if os.environ.get('OPENAI_API_KEY'):
            req.add_header('authorization', 'Bearer ' + os.environ['OPENAI_API_KEY'])
        with urllib.request.urlopen(req, timeout=10) as r:
            text = r.read().decode()
    except Exception:
        return {}
    m = {}
    for line in text.splitlines():
        if line.startswith('#'):
            continue
        mm = re.match(r'^([a-zA-Z_:][\w:]*)(\{[^}]*\})?\s+([-+0-9.eE]+|NaN)$', line.strip())
        if mm and mm.group(3) != 'NaN':
            m[mm.group(1)] = m.get(mm.group(1), 0.0) + float(mm.group(3))
    return m


def delta(after, before, *names):
    for n in names:
        if n in after:
            return after[n] - before.get(n, 0.0)
    return None


def reset_prefix_cache(base_url: str) -> bool:
    """尽量清前缀缓存：vLLM /reset_prefix_cache 或 SGLang /flush_cache。"""
    root = re.sub(r'/v1/?$', '', base_url.rstrip('/'))
    for path in ('/reset_prefix_cache', '/flush_cache'):
        try:
            urllib.request.urlopen(
                urllib.request.Request(root + path, method='POST', data=b''), timeout=10
            ).read()
            return True
        except Exception:
            continue
    return False



def wait_healthy(
    base_url: str,
    prefill_url: str = 'http://10.230.0.109:30000',
    timeout: int = 180,
    model: str | None = None,
) -> bool:
    """路由器 + Prefill/Decode 健康才继续；并用一次短 chat 探测 circuit-open。"""
    deadline = time.time() + timeout
    root = base_url.rstrip('/')
    paths = [
        root + '/v1/models',
        'http://127.0.0.1:30001/health',
        prefill_url.rstrip('/') + '/health',
    ]
    body = json.dumps(
        {
            'model': model
            or '/data/aaron/models/nvidia--DeepSeek-V4.1-Flash-1017-NVFP4',
            'messages': [{'role': 'user', 'content': 'ping'}],
            'max_tokens': 1,
            'temperature': 0,
        }
    ).encode()
    while time.time() < deadline:
        ok = True
        for url in paths:
            try:
                urllib.request.urlopen(url, timeout=5)
            except Exception:
                ok = False
                break
        if ok:
            try:
                req = urllib.request.Request(
                    root + '/v1/chat/completions',
                    data=body,
                    headers={'Content-Type': 'application/json'},
                )
                with urllib.request.urlopen(req, timeout=60) as r:
                    if 200 <= r.status < 300:
                        return True
            except Exception:
                ok = False
        time.sleep(5)
    return False

def resolve_python(a) -> list[str]:
    """返回启动 sglang.benchmark.serving 的命令前缀。"""
    if a.python:
        return [a.python, '-m', 'sglang.benchmark.serving']
    # 优先当前解释器（通常已 source activate-sglang.sh）
    return [sys.executable, '-m', 'sglang.benchmark.serving']


# ---------------------------------------------------------------------------
# dataset → sglang.benchmark.serving args
# ---------------------------------------------------------------------------

def _random_dataset_cli(isl: int, osl: int) -> list[str]:
    """上游用 ShareGPT 采样再截断到 isl；有本地 JSON 则 dataset=random，否则 random-ids。"""
    base = [
        '--random-input-len', str(isl),
        '--random-output-len', str(osl),
        '--random-range-ratio', os.environ.get('BENCH_RANGE_RATIO', '0'),
    ]
    if os.path.isfile(LOCAL_SHAREGPT):
        return [
            '--dataset-name', 'random',
            '--dataset-path', LOCAL_SHAREGPT,
            *base,
        ]
    return ['--dataset-name', 'random-ids', *base]


def dataset_args(a, isl, osl, hit):
    """返回 (CLI 参数列表, 结果文件名前缀)。语义对齐上游 random / prefix。"""
    kind, _, rest = a.dataset.partition(':')
    if kind == 'random':
        if hit is COLD:
            return _random_dataset_cli(isl, osl), f'isl{isl}_osl{osl}_cold'
        if hit and hit > 0:
            # 对齐上游 random-prefix-len：共享前缀 ≈ isl*hit
            # 长上下文：少 groups + fast-prepare，避免本地造 57k×128 条卡死
            prefix = max(1, int(round(isl * hit)))
            suffix = max(1, isl - prefix)
            groups = '2' if isl >= 30000 else '8'
            ppg = '8' if isl >= 30000 else '16'
            x = [
                '--dataset-name', 'generated-shared-prefix',
                '--gsp-num-groups', groups,
                '--gsp-prompts-per-group', ppg,
                '--gsp-system-prompt-len', str(prefix),
                '--gsp-question-len', str(suffix),
                '--gsp-output-len', str(osl),
            ]
            if isl >= 30000:
                x.append('--gsp-fast-prepare')
            return x, f'isl{isl}_osl{osl}_hit{int(hit * 100)}'

        return _random_dataset_cli(isl, osl), f'isl{isl}_osl{osl}_hit0'
    if kind == 'sharegpt':
        x = ['--dataset-name', 'sharegpt', '--dataset-path', rest]
        if a.output_len:
            x += ['--sharegpt-output-len', str(a.output_len)]
        return x, 'sharegpt'
    if kind == 'custom':
        # sglang custom 无独立 output-len flag；需要固定长度时用 --extra
        return (
            ['--dataset-name', 'custom', '--dataset-path', rest],
            'custom',
        )
    if kind == 'prefix':
        # 复用 gsp：prefix_len:suffix_len:num_prefixes
        pl, sl, npf = (rest.split(':') + ['', '', ''])[:3]
        pl_i = int(pl or '256')
        sl_i = int(sl or '128')
        np_i = int(npf or '4')
        return (
            [
                '--dataset-name', 'generated-shared-prefix',
                '--gsp-num-groups', str(np_i),
                '--gsp-prompts-per-group', '16',
                '--gsp-system-prompt-len', str(pl_i),
                '--gsp-question-len', str(sl_i),
                '--gsp-output-len', str(a.output_len or osl or 128),
            ],
            f'prefix{pl_i}_{sl_i}_{np_i}',
        )
    raise SystemExit(
        f'不认识的 --dataset：{a.dataset}（本适配版支持 random / sharegpt / custom / prefix）'
    )


def ds_label(ds: str) -> str:
    m = re.match(r'^custom:.*?realtext_([\w.-]+)\.jsonl$', ds or '')
    return f'realtext:{m.group(1)}' if m else ds


def nominal_hit(a) -> float:
    if a.dataset.startswith('prefix:'):
        pl, sl = (a.dataset.split(':')[1:] + ['256', '128'])[:2]
        return round(100 * float(pl or 256) / (float(pl or 256) + float(sl or 128)), 1)
    return 0.0


# ---------------------------------------------------------------------------
# result helpers
# ---------------------------------------------------------------------------

def load_result(path: Path) -> dict:
    """sglang 默认写 JSONL（一行一个 JSON）；也兼容单文件 JSON。"""
    text = path.read_text().strip()
    if not text:
        raise ValueError(f'空结果文件：{path}')
    if path.suffix == '.jsonl' or '\n' in text:
        # 取最后一行非空
        last = [ln for ln in text.splitlines() if ln.strip()][-1]
        return json.loads(last)
    return json.loads(text)


def parse_slo(s: str | None):
    """与上游一致：'ttft:3000,tpot:50' → (3000.0, 50.0)；也兼容 '3000,50'。"""
    if not s:
        return 3000.0, 50.0
    if ':' in s:
        d = dict(x.split(':') for x in s.split(',') if ':' in x)
        return float(d.get('ttft', 3000)), float(d.get('tpot', 50))
    a, b = [float(x) for x in s.split(',')[:2]]
    return a, b


def slo_fields(reqs, total, duration, slo):
    """reqs: [(ttft_ms, tpot_ms|None, out_tokens)]。

    total 用「有明细的完成请求数」归一，避免 GSP 实际完成数 > --num-prompts
    时出现 slo_pct>100。
    """
    t_ms, p_ms = slo
    good = [(f, t, o) for f, t, o in reqs if f <= t_ms and (t is None or t <= p_ms)]
    denom = max(len(reqs), int(total or 0), 1)
    if not duration:
        return {
            'slo_ttft': t_ms,
            'slo_tpot': p_ms,
            'slo_pct': round(100 * len(good) / denom, 1),
        }
    return {
        'slo_ttft': t_ms,
        'slo_tpot': p_ms,
        'slo_pct': round(100 * len(good) / denom, 1),
        'good_tps': round(sum(o for _, _, o in good) / duration, 2),
        'goodput': round(len(good) / duration, 3),
    }


def slo_from_detailed(res: dict, nreq: int, slo) -> dict:
    """优先用 sglang --output-details 的 ttfts/itls/output_lens；否则试 dict 列表。"""
    if isinstance(slo, str):
        slo = parse_slo(slo)
    ttfts, itls, outs = res.get('ttfts'), res.get('itls'), res.get('output_lens')
    if ttfts and outs:
        errs = res.get('errors') or [''] * len(ttfts)
        med = statistics.median([x for x in ttfts if x and x > 0] or [0])
        k = (
            1000.0
            if med
            and res.get('median_ttft_ms')
            and abs(med * 1000 - res['median_ttft_ms'])
            < abs(med - res['median_ttft_ms'])
            else 1.0
        )
        reqs = []
        for i, f in enumerate(ttfts):
            o = outs[i] if i < len(outs) else 0
            if not f or f <= 0 or not o or (i < len(errs) and errs[i]):
                continue
            it = itls[i] if itls and i < len(itls) else None
            tp = (sum(it) * k / (o - 1)) if it and o > 1 else None
            reqs.append((f * k, tp, o))
        # 分母取明细条数，与 nreq 解耦（GSP 常生成 groups*ppg > num-prompts）
        return slo_fields(reqs, len(reqs) or nreq, res.get('duration'), slo)

    # fallback：goodput 字段（--goodput 时 sglang 可能直接给出）
    if res.get('goodput') is not None or res.get('slo_attainment') is not None:
        t_ms, p_ms = slo
        out = {'slo_ttft': t_ms, 'slo_tpot': p_ms}
        if res.get('goodput') is not None:
            out['goodput'] = round(float(res['goodput']), 3)
        if res.get('slo_attainment') is not None:
            v = float(res['slo_attainment'])
            pct = round(v * 100 if v <= 1.0 else v, 1)
            out['slo_pct'] = min(pct, 100.0)
        return out
    return {'slo_ttft': slo[0], 'slo_tpot': slo[1]}


def pick(res: dict, *keys, default=0):
    for k in keys:
        if k in res and res[k] is not None:
            return res[k]
    return default


# ---------------------------------------------------------------------------
# run one matrix point
# ---------------------------------------------------------------------------

def run_point(a, isl, osl, hit, c, outdir: Path, rate: float | None = None):
    cold = hit is COLD
    dargs, prefix = dataset_args(a, isl, osl, hit if not cold else COLD)
    tag = prefix
    # 与上游：nreq = max(min_requests, req_per_conc * c)；速率点 = rate * rate_seconds
    if rate:
        nreq = max(a.min_requests, int(round(rate * a.rate_seconds)))
    else:
        nreq = max(a.min_requests, a.req_per_conc * c)
    # 长上下文在本机（SM120 torch indexer）TTFT 极高：压请求数保证基线可完成
    if isl and isl >= 60000:
        nreq = min(nreq, max(4, c * 2))
    # gsp：保证 groups*prompts ≥ nreq
    if 'generated-shared-prefix' in dargs:
        try:
            gi = dargs.index('--gsp-prompts-per-group')
            ni = dargs.index('--gsp-num-groups')
            need = max(1, (nreq + int(dargs[ni + 1]) - 1) // int(dargs[ni + 1]))
            if int(dargs[gi + 1]) < need:
                dargs[gi + 1] = str(need)
        except (ValueError, IndexError):
            pass
    seed_m = a.seed + (0 if not cold else hash((isl, osl, c, 'cold')) % 100000)
    seed_w = seed_m + 17
    name = f'{prefix}_c{c}' + (f'_r{rate:g}' if rate else '') + '.jsonl'
    py = resolve_python(a)
    slo_pair = a.slo if isinstance(a.slo, tuple) else parse_slo(a.slo)

    def cmd(n: int, save: bool, flush: bool = False):
        # sglang.benchmark.serving：--model / --tokenizer；默认 ignore_eos=True
        x = py + [
            '--backend', a.backend,
            '--base-url', a.base_url,
            '--model', a.model,
            *dargs,
            '--num-prompts', str(n),
            '--max-concurrency', str(c),
            '--request-rate', (f'{rate:g}' if rate and save else 'inf'),
            '--seed', str(seed_m if save else seed_w),
            '--disable-tqdm',
            '--ready-check-timeout-sec', '60',
        ]
        if a.tokenizer:
            x += ['--tokenizer', a.tokenizer]
        if a.pd_separated:
            x.append('--pd-separated')
        if a.cache_report and save:
            x.append('--cache-report')
        if flush or (save and cold):
            x.append('--flush-cache')
        if save:
            x += [
                '--goodput', f'ttft:{slo_pair[0]:g}', f'tpot:{slo_pair[1]:g}',
                '--output-file', str(outdir / name),
                '--output-details',
            ]
        if a.extra:
            x += a.extra.split()
        return x

    if cold:
        reset_prefix_cache(a.base_url)

    warmup_n = a.warmup
    if isl and isl >= 60000:
        warmup_n = min(warmup_n, 1)  # 长上下文预热减负（60k TTFT 极高）
    if warmup_n > 0:
        print(f'  预热 C{c} ×{warmup_n} …', flush=True)
        subprocess.run(
            cmd(max(min(c, 8) if not rate else min(c, 8), warmup_n), False, flush=cold),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    before = scrape(a.base_url)
    print(
        f'  测量 {tag} '
        + (f'速率={rate:g}/s（并发上限 {c}）' if rate else f'C={c}')
        + f' 请求数={nreq}',
        flush=True,
    )
    t0 = time.time()
    p = subprocess.run(cmd(nreq, True), capture_output=True, text=True)
    out_path = outdir / name
    if p.returncode != 0 or not out_path.exists():
        err = (p.stdout or '')[-2000:] + '\n' + (p.stderr or '')[-2000:]
        print(err, file=sys.stderr)
        # 注意：Namespace dump 里常有 apply_chat_template=False，勿用裸 'chat_template' 匹配
        low = err.lower()
        if (
            'does not have a chat template' in low
            or 'tokenizer does not have a chat_template' in low
            or 'missing chat template' in low
        ):
            a._tmpl_err = True
            print(
                '  ✗ tokenizer 缺 chat_template：检查 --tokenizer，'
                '或换 --backend sglang-oai',
                file=sys.stderr,
                flush=True,
            )
        elif 'localentrynotfounderror' in low or 'huggingface' in low:
            print(
                '  ✗ 数据集需要从 HuggingFace 下载（本机无外网）。'
                '默认矩阵已用 random-ids / gsp；若仍出现请检查 --extra/--dataset-path',
                file=sys.stderr,
                flush=True,
            )
        else:
            print(f'  ✗ 测量失败（rc={p.returncode}）', file=sys.stderr, flush=True)
        return None

    after = scrape(a.base_url)
    try:
        res = load_result(out_path)
    except Exception as e:
        print(f'  ✗ 解析结果失败：{e}', file=sys.stderr, flush=True)
        return None

    done = int(pick(res, 'completed', 'completed_requests', default=0) or 0)
    n_prompts = int(pick(res, 'num_prompts', default=nreq) or nreq)
    # GSP 等路径常实际完成 > --num-prompts；分母取真实完成与标称请求的较大值
    n_denom = max(n_prompts, done, nreq, 1)
    total_in = pick(res, 'total_input_tokens', default=0)
    total_out = pick(res, 'total_output_tokens', default=0)
    # 矩阵名义 isl/osl 作为主键；实测均值仅在合理范围内覆盖（避免 GSP 只计 suffix→isl≈1）
    measured_isl = round(total_in / done) if done and total_in else 0
    measured_osl = round(total_out / done) if done and total_out else 0
    use_isl = measured_isl if measured_isl >= max(16, int(isl * 0.2)) else isl
    use_osl = measured_osl if measured_osl >= max(1, int(osl * 0.05)) else osl
    target_hit = (
        0.0
        if cold
        else (
            round((hit or 0) * 100, 1)
            if a.dataset == 'random'
            else nominal_hit(a)
        )
    )

    row = {
        'isl': use_isl,
        'osl': use_osl,
        'hit': target_hit,
        'c': c,
        'nreq': n_denom,
        'out_tps': round(pick(res, 'output_throughput', default=0), 2),
        'in_tps': round(pick(res, 'input_throughput', default=0), 1),
        'ttft_p50': round(pick(res, 'median_ttft_ms', default=0), 2),
        'tpot_p50': round(pick(res, 'median_tpot_ms', default=0), 3),
        'ttft_p95': round(pick(res, 'p95_ttft_ms', default=0), 2),
        'tpot_p95': round(pick(res, 'p95_tpot_ms', default=0), 3),
        'itl_p95': round(pick(res, 'p95_itl_ms', default=0), 3),
        'itl_p99': round(pick(res, 'p99_itl_ms', default=0), 3),
        'ttft_p99': round(pick(res, 'p99_ttft_ms', default=0), 2) or '',
        'tpot_p99': round(pick(res, 'p99_tpot_ms', default=0), 3) or '',
        'total_tps': round(
            pick(res, 'total_throughput', 'total_token_throughput', default=0), 2
        ) or '',
        'success': round(100.0 * min(done, n_denom) / n_denom, 2),
        'accept': '',
        'ds': ds_label(a.dataset) if a.dataset != 'random' else ('random:cold' if cold else 'random'),
    }
    if rate:
        row['rate'] = rate
    row.update(slo_from_detailed(res, n_denom, a.slo))
    if 'slo_pct' in row and row['slo_pct'] is not None:
        try:
            row['slo_pct'] = min(float(row['slo_pct']), 100.0)
        except (TypeError, ValueError):
            pass

    # 服务端命中率（若开了 metrics）；目标命中>0 时勿被 0 查询噪声覆盖
    hits = delta(
        after, before,
        'vllm:prefix_cache_hits_total', 'vllm:prefix_cache_hits',
        'sglang:cache_hit_total', 'sglang:prefix_cache_hits',
    )
    qs = delta(
        after, before,
        'vllm:prefix_cache_queries_total', 'vllm:prefix_cache_queries',
        'sglang:cache_query_total', 'sglang:prefix_cache_queries',
    )
    if hits is not None and qs and qs > 0:
        measured_hit = round(100 * hits / qs, 1)
        if target_hit <= 0 or measured_hit > 0:
            row['hit'] = measured_hit
    # --cache-report：仅在有正命中或无目标命中时覆盖（避免 GSP 目标95被报告0冲掉）
    cr = res.get('cache_report') or {}
    if isinstance(cr, dict) and cr.get('cache_hit_rate_pct') is not None:
        cr_hit = round(float(cr['cache_hit_rate_pct']), 1)
        if target_hit <= 0 or cr_hit > 0:
            row['hit'] = cr_hit
        else:
            row['hit'] = target_hit
    elif target_hit > 0 and float(row.get('hit') or 0) <= 0:
        row['hit'] = target_hit
    if res.get('accept_length') is not None:
        row['accept'] = round(float(res['accept_length']), 2)

    tpot_ms = float(pick(res, 'median_tpot_ms', default=0) or 0)
    dur = float(pick(res, 'duration', default=0) or 0)
    ttfts = [float(x) for x in (res.get('ttfts') or []) if x is not None]
    ttft_min = min(ttfts) if ttfts else float(pick(res, 'median_ttft_ms', default=0) or 0) / 1000.0
    total_out = float(pick(res, 'total_output_tokens', default=0) or 0)
    # 单流首字后：TPOT 的倒数。TPOT = (e2e - ttft) / (output_len - 1)。
    row['after_ttft_tps'] = round(1000.0 / tpot_ms, 2) if tpot_ms > 0 else ''
    wave = dur - ttft_min
    # 从本轮最早一条首字到测量结束的总输出速率。并发一批时接近 decode 阶段吞吐。
    row['after_ttft_wave_tps'] = round(total_out / wave, 2) if wave > 0.05 else ''
    out_tps = float(pick(res, 'output_throughput', default=0) or 0)
    row['active_decode_est'] = round(out_tps * tpot_ms / 1000.0, 2) if tpot_ms > 0 else ''

    print(
        f'    → out {row["out_tps"]} tok/s · in {row["in_tps"]} tok/s · '
        f'首字后单流 {row["after_ttft_tps"]} tok/s · '
        f'首字后整段 {row["after_ttft_wave_tps"]} tok/s · '
        f'活跃decode约 {row["active_decode_est"]} · '
        f'TTFT p50/p95 {row["ttft_p50"]}/{row["ttft_p95"]} ms · '
        f'TPOT p50/p95 {row["tpot_p50"]}/{row["tpot_p95"]} ms · '
        f'ITL p95/p99 {row["itl_p95"]}/{row["itl_p99"]} ms · 命中 {row["hit"]}%'
        + (
            f' · 达标 {row["slo_pct"]}%（达标产能 {row["good_tps"]} tok/s）'
            if 'slo_pct' in row
            else ''
        )
        + f' · 用时 {time.time() - t0:.0f}s',
        flush=True,
    )
    return row


# ---------------------------------------------------------------------------
# post to ardelva (optional, upstream-compatible)
# ---------------------------------------------------------------------------

def post(a, rows: list[dict]):
    payload = {
        'engine': a.engine,
        'gpu': a.gpu,
        'note': a.note,
        'points': [
            {k: r.get(k, '') for k in CSV_COLS if k in r or k in ('isl', 'osl', 'hit', 'c', 'out_tps')}
            for r in rows
        ],
    }
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        a.post_url,
        data=data,
        headers={'Content-Type': 'application/json'},
        method='POST',
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            print(f'已提交 {len(rows)} 点 → {a.post_url} 状态 {r.status}')
    except Exception as e:
        print(f'提交失败：{e}', file=sys.stderr)


# ---------------------------------------------------------------------------
# CLI / main
# ---------------------------------------------------------------------------

def build_parser():
    p = argparse.ArgumentParser(
        description='ardelva_bench PD 适配版：用 sglang.benchmark.serving 扫矩阵，写兼容 CSV',
    )
    g = p.add_argument_group('服务')
    g.add_argument('--base-url', default=PD_DEFAULTS['base_url'])
    g.add_argument('--model', default=PD_DEFAULTS['model'],
                   help='传给 sglang bench --model（served model id / 路径）')
    g.add_argument('--tokenizer', default=PD_DEFAULTS['tokenizer'])
    g.add_argument('--backend', default=PD_DEFAULTS['backend'],
                   help='sglang-oai-chat / sglang-oai / vllm-chat …')
    g.add_argument('--python', default='',
                   help='python 可执行文件；默认 sys.executable（请先 source activate-sglang.sh）')
    g.add_argument('--extra', default='', help='额外传给 serving bench 的参数（空格分隔）')
    g.add_argument('--pd-separated', action='store_true', default=True,
                   help='开启 PD 分离计时（默认开）')
    g.add_argument('--no-pd-separated', action='store_false', dest='pd_separated')
    g.add_argument('--cache-report', action='store_true', default=True,
                   help='开启 cache-report（默认开）')
    g.add_argument('--no-cache-report', action='store_false', dest='cache_report')

    g2 = p.add_argument_group('矩阵')
    g2.add_argument('--dataset', default='random',
                    help='random | sharegpt:<path> | custom:<jsonl> | prefix:pl:sl:np')
    g2.add_argument('--matrix', default='',
                    help="覆盖默认矩阵，如 '3000:1000:0:1,8,32,64;60000:1000:95:1,8,32'")
    g2.add_argument(
        '--conc', default='',
        help='非 random 数据集的并发列表（默认 1,4,8,16,32,64）',
    )
    g2.add_argument('--output-len', type=int, default=0)
    g2.add_argument('--min-requests', type=int, default=32,
                    help='每个点至少的请求数（默认 32，与上游对齐）')
    g2.add_argument(
        '--req-per-conc', '--requests-per-concurrency',
        dest='req_per_conc', type=int, default=4,
        help='每个点的请求数 = 并发 × 这个值（默认 4；别名 --requests-per-concurrency）',
    )
    g2.add_argument('--warmup', type=int, default=8, help='每个点预热请求数（0 关闭）')
    g2.add_argument('--seed', type=int, default=1)
    g2.add_argument('--rate', default='', help='可选：按速率扫描，逗号分隔，如 1,2,4')
    g2.add_argument('--rate-cap', type=int, default=256,
                    help='按速率压测时的并发上限（默认 256）')
    g2.add_argument('--rate-seconds', type=int, default=120,
                    help='按速率压测时每个点的目标时长（秒），请求数 = 速率 × 时长')
    g2.add_argument('--only-rate', action='store_true')
    g2.add_argument(
        '--slo', default='ttft:3000,tpot:50',
        help='服务目标（ms），如 ttft:3000,tpot:50（默认与上游一致）',
    )
    g2.add_argument(
        '--stop-ttft', type=float, default=3000,
        help='过载即停：p50 TTFT(ms) 超过此值且达标率低于 --stop-slo 则跳过更高并发；0=关闭',
    )
    g2.add_argument('--stop-slo', type=float, default=50,
                    help='过载即停的达标率门槛 %%（默认 50）')

    g3 = p.add_argument_group('输出 / 提交')
    g3.add_argument('--out', default='~/pd/bench/runs/latest')
    g3.add_argument('--engine', default=PD_DEFAULTS['engine'])
    g3.add_argument('--gpu', default='8×RTX6000D Decode(107) + 8×RTXPRO5000 Prefill(109)')
    g3.add_argument('--note', default='PD mooncake_tcp ctx=1M max_running=64')
    g3.add_argument('--post', action='store_true')
    g3.add_argument('--post-url', default='https://ardelva.ai/api/bench')

    g4 = p.add_argument_group('冒烟')
    g4.add_argument(
        '--smoke', action='store_true',
        help='只跑矩阵首点规格 3000:1000 C=1，并压低 nreq/warmup 以保持快速验证',
    )
    return p


def main():
    a = build_parser().parse_args()
    a._tmpl_err = False
    a._skipped = []
    a.slo = parse_slo(a.slo)
    a.out = os.path.expanduser(a.out)
    outdir = Path(a.out).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    # 快速自检：bench 模块可导入
    chk = subprocess.run(
        resolve_python(a) + ['--help'],
        capture_output=True, text=True,
    )
    if chk.returncode != 0:
        print(
            '无法启动 sglang.benchmark.serving。请先：\n'
            '  source ~/pd/activate-sglang.sh\n'
            f'stderr: {(chk.stderr or "")[-500:]}',
            file=sys.stderr,
        )
        sys.exit(2)

    if a.smoke:
        # 规格与上游 DEFAULT_MATRIX 首点一致（3000/1000），但压低样本量保持冒烟快
        matrix = [(3000, 1000, 0.0, [1])]
        a.min_requests = min(a.min_requests, 4)
        a.req_per_conc = min(a.req_per_conc, 2)
        a.warmup = min(a.warmup, 1)
    elif a.dataset == 'random':
        matrix = parse_matrix(a.matrix) if a.matrix else DEFAULT_MATRIX
    else:
        matrix = [(0, a.output_len or 0, 0.0,
                   [int(c) for c in (a.conc or '1,4,8,16,32,64').split(',')])]

    print(
        f'PD bench → {a.base_url}  backend={a.backend}\n'
        f'  model={a.model}\n'
        f'  tokenizer={a.tokenizer}\n'
        f'  out={outdir}\n'
        f'  matrix points≈{sum(len(cs) for *_, cs in matrix)}',
        flush=True,
    )

    rows = []
    failed = []
    for isl, osl, hit, cs in matrix:
        if a.dataset == 'random':
            print(
                f'== 输入 {isl} / 输出 {osl} / 目标命中 '
                + ('完全冷启动' if hit is COLD else f'{hit:.0%}'),
                flush=True,
            )
        else:
            print(f'== 数据集 {a.dataset}', flush=True)
        for c in ([] if a.only_rate and a.rate else cs):
            if not wait_healthy(a.base_url):
                print('  ✗ 服务不健康，中止本轮矩阵（需重启 Prefill/Decode）', file=sys.stderr, flush=True)
                failed.append('infra_unhealthy')
                a._tmpl_err = True
                break
            r = run_point(a, isl, osl, hit, c, outdir)
            if r:
                rows.append(r)
                # 逐点落盘，避免长矩阵中途崩溃丢全部 CSV
                csv_path = outdir / 'ardelva_points.csv'
                with open(csv_path, 'w', newline='') as f:
                    w = csv.DictWriter(f, fieldnames=CSV_COLS, extrasaction='ignore')
                    w.writeheader()
                    w.writerows(rows)
            if a._tmpl_err:
                break
            rest = [x for x in cs if x > c]
            if (
                r and rest and a.stop_ttft
                and r.get('ttft_p50', 0) > a.stop_ttft
                and r.get('slo_pct', 0) < a.stop_slo
            ):
                msg = (
                    f"random {isl}/{osl} C{'/C'.join(map(str, rest))} 跳过"
                    f"（C{c} 已过载：TTFT p50 {r['ttft_p50']/1000:.1f}s）"
                )
                print('  ⏭ ' + msg, flush=True)
                a._skipped.append(msg)
                break
        if a.rate and not a._tmpl_err:
            for rt in [float(x) for x in a.rate.split(',') if x.strip()]:
                r = run_point(a, isl, osl, hit, a.rate_cap, outdir, rate=rt)
                if r:
                    rows.append(r)
                    csv_path = outdir / 'ardelva_points.csv'
                    with open(csv_path, 'w', newline='') as f:
                        w = csv.DictWriter(f, fieldnames=CSV_COLS, extrasaction='ignore')
                        w.writeheader()
                        w.writerows(rows)
        if a.dataset == 'random' and not any(
            row.get('isl') == isl and row.get('osl') == osl for row in rows[-len(cs) - 5:]
        ):
            # 粗略检测本行是否全挂
            pass
        if a._tmpl_err:
            failed.append(ds_label(a.dataset))
            break

    csv_path = outdir / 'ardelva_points.csv'
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLS, extrasaction='ignore')
        w.writeheader()
        w.writerows(rows)
    print(f'\n共 {len(rows)} 个点，CSV：{csv_path}')
    if a.post and rows:
        post(a, rows[:MAX_POINTS])
    if not rows:
        print('0 个点成功', file=sys.stderr)
        sys.exit(3)


if __name__ == '__main__':
    main()
