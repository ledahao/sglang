#!/usr/bin/env python3
"""
按「输入长度 × 并发」扫描 vLLM 推理服务，输出 ardelva.ai 部署测算的实测 CSV，可选直接提交。

每个点调用一次 `vllm bench serve`（random 数据集、固定输入输出长度、--ignore-eos、固定并发闭环），
从服务端 /metrics 读取每个点的真实前缀缓存命中率和投机解码接受长度。

用法：
  # 1) 只测，生成 CSV（之后可在部署测算「提交本配置的实测数据」里粘贴导入）
  python3 ardelva_bench.py --base-url http://10.0.0.10:8000 --model deepseek-v4.1-flash \
      --tokenizer /models/DeepSeek-V4.1-Flash --out run1

  # 2) 测完直接提交到 ardelva.ai（三把 key 从部署测算「方案标识」里复制）
  ARDELVA_KEY=YOUR_KEY python3 ardelva_bench.py ... --post \
      --model-key '…' --deploy-key '…' --hw-key '…' \
      --submitter 'Ardelva Lab' --engine 'vLLM 0.19 (Ardelva a1b2c3)'

  # 3) 用真实数据集（长度由数据决定，只扫并发；命中率从服务端读）
  python3 ardelva_bench.py ... --dataset hf:likaixin/InstructCoder --conc 1,4,8,16,32,64 --output-len 512
  python3 ardelva_bench.py ... --dataset hf:vdaita/edit_10k_char --conc 4,16,32
  python3 ardelva_bench.py ... --dataset sharegpt:/data/ShareGPT_V3_unfiltered_cleaned_split.json
  python3 ardelva_bench.py ... --dataset custom:/data/agent_traces.jsonl --skip-chat-template
  python3 ardelva_bench.py ... --dataset prefix:57000:3000:8 --output-len 1000   # 8 个 57K 共享前缀，模拟多个代码库上下文

  # 标准套件：random 矩阵（和估算逐点对照）+ 真实编程数据集 InstructCoder（真实长度、真实投机接受长度）
  python3 ardelva_bench.py ... --hf-coding            # 默认矩阵之后再跑 InstructCoder，C1/4/8/16/32，输出 1000

  # 正式负载（产能与成本按它们计算）：先用 tools/build_realtext.py 生成 3K/10K/32K 真实文本，再
  python3 ardelva_bench.py ... --realtext rt --multiturn --think     # 默认 random 标定 13 点（含 5 个完全冷启动点）+ 真实文本 3K/10K/32K + 多轮 60K + 思考模式 10K→4K

  # 多轮 Agent：每个会话起始 60K，每轮新增 5K（上一轮回复 500 + 工具结果 4500），共 5 轮；同一会话的轮次串行
  python3 ardelva_bench.py ... --multiturn            # 默认 multiturn:60000:5000:5，C1/4/8/16/32
  python3 ardelva_bench.py ... --dataset multiturn:80000:8000:6:800 --conc 4,8,16   # 起始:每轮:轮数[:输出[:共享前缀]]

  # 按请求速率扫描（泊松到达），统计服务目标达标率：找「每秒多少请求时还能守住 TTFT / TPOT」
  python3 ardelva_bench.py ... --matrix '3000:1000:0:1' --rate 0.5,1,1.5,2 --slo ttft:3000,tpot:50
  #    所有点都会按 --slo（默认 ttft:3000,tpot:50，即首字 ≤3 s、每用户 ≥20 tok/s）统计达标请求占比与达标产能

  # 精度检查：压测前在同一服务上跑固定题集（ardelva-qc-v1，40 题，temperature 0），防止「快了但错了」
  python3 ardelva_bench.py ... --accuracy --accuracy-min 90         # 低于 90% 就不压测
  python3 ardelva_bench.py ... --accuracy --accuracy-save ref.json   # 基线版本存一份参考答案
  python3 ardelva_bench.py ... --accuracy --accuracy-ref ref.json    # 换算子后与参考答案逐题比对
  python3 ardelva_bench.py ... --accuracy-only                       # 只跑精度检查

  # 4) 已经测完、当时没带 key：不用重测，直接提交已有结果
  ARDELVA_KEY=YOUR_KEY python3 ardelva_bench.py --from-csv run1/ardelva_points.csv --date 2026-10-28 \
      --model-key '…' --deploy-key '…' --hw-key '…' --engine 'vLLM 0.19'
  #    用原生 `vllm bench serve --save-result` 跑的：指向结果 JSON（目录或文件，可多个）
  ARDELVA_KEY=YOUR_KEY python3 ardelva_bench.py --from-json old_runs/ --hit 95 --date 2026-10-28 ...

只需要 Python 3.8+ 标准库和装好 vLLM（提供 `vllm bench serve` 命令）的环境。
"""
import argparse, csv, json, os, random, re, statistics, subprocess, sys, time, urllib.parse, urllib.request
from datetime import date
from pathlib import Path

# 默认扫描矩阵：(平均输入, 输出, 目标命中率, 并发列表)
# random 只做物理标定（部署测算逐点对照、跨平台对比）：三种物理情形各取几个并发即可；
# 产能与成本按真实文本（--realtext）与多轮（--multiturn）的完整并发扫描计算
DEFAULT_MATRIX = [
    (3000, 1000, 0.0, [1, 8, 32]),        # 短上下文：解码、MoE、通信
    (60000, 1000, 0.95, [1, 8, 32]),      # 长上下文解码 + 前缀缓存命中路径
    (100000, 1000, 0.0, [1, 8]),          # 长输入 Prefill 吞吐与 TTFT（与前面各点共用种子，会有部分前缀命中，接近真实使用）
    (3000, 1000, None, [1, 8, 32]),       # 完全冷启动：每个点独立种子、预热换种子、先清前缀缓存，命中 ≈ 0
    (100000, 1000, None, [1, 8]),         # 完全冷启动的长输入 Prefill
]
COLD = None   # 矩阵里命中写 cold（或 None）= 完全冷启动；结果的数据集记为 random:cold
CSV_COLS = ['isl', 'osl', 'hit', 'c', 'nreq', 'out_tps', 'ttft_p50', 'tpot_p50',
            'ttft_p99', 'tpot_p99', 'total_tps', 'success', 'accept', 'ttft_t1', 'ttft_tn', 'ds',
            'rate', 'slo_ttft', 'slo_tpot', 'slo_pct', 'good_tps', 'goodput']   # 后 6 列为新增（可选），旧 CSV 照常导入
HF_CODING = 'hf:likaixin/InstructCoder'   # --hf-coding 默认的真实编程数据集
MULTITURN = 'multiturn:60000:5000:5'       # --multiturn 默认：起始 60K，每轮 +5K，共 5 轮（模拟编码 Agent）
MT_COMMIT = '5d55ef1590c92213e3893dc8e8712f7f2e0bc508'   # vLLM benchmarks/multi_turn 固定版本
MT_FILES = ['benchmark_serving_multi_turn.py', 'bench_dataset.py', 'bench_utils.py']
UA = 'Mozilla/5.0 (compatible; ardelva_bench/1.3; +https://ardelva.ai/tests/bench/)'
MAX_POINTS = 96


def parse_matrix(s):
    """--matrix '3000:1000:50:1,4,8;60000:1000:95:8,16,32' → 列表"""
    out = []
    for part in s.split(';'):
        isl, osl, hit, cs = part.split(':')
        out.append((int(isl), int(osl), COLD if hit.strip().lower() == 'cold' else float(hit) / 100, [int(c) for c in cs.split(',')]))
    return out


def reset_prefix_cache(base_url):
    root = re.sub(r'/v1/?$', '', base_url.rstrip('/'))
    try:
        urllib.request.urlopen(urllib.request.Request(root + '/reset_prefix_cache', method='POST', data=b''), timeout=10).read()
        return True
    except Exception:
        return False


def scrape(base_url):
    """读 Prometheus /metrics，返回 {指标名: 数值}（同名不同标签的累加）"""
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


def dataset_args(a, isl, osl, hit):
    """返回 (vllm bench 的数据集参数, 结果文件名前缀)。random 按矩阵造数据；其余数据集长度由数据决定，
    这里只控制输出长度（--output-len，配合 --ignore-eos）。"""
    kind, _, rest = a.dataset.partition(':')
    if kind == 'random':
        if hit is COLD:                       # 完全冷启动：没有共享前缀，全部随机
            return (['--dataset-name', 'random', '--random-input-len', str(isl), '--random-output-len', str(osl),
                     '--random-range-ratio', '0', '--random-prefix-len', '0'], f'isl{isl}_osl{osl}_cold')
        prefix = int(round(isl * hit))
        rand_in = max(1, isl - prefix)        # vLLM random 数据集：总长度 = 共享前缀 + 随机部分
        return (['--dataset-name', 'random', '--random-input-len', str(rand_in), '--random-output-len', str(osl),
                 '--random-range-ratio', '0', '--random-prefix-len', str(prefix)],
                f'isl{isl}_osl{osl}_hit{int(hit * 100)}')
    ol = [str(a.output_len)] if a.output_len else []
    if kind == 'hf':                          # hf:<repo 或本地路径>[:split[:subset]]
        parts = rest.split(':')
        x = ['--dataset-name', 'hf', '--dataset-path', parts[0], '--hf-split', parts[1] if len(parts) > 1 and parts[1] else 'train']
        if len(parts) > 2 and parts[2]:
            x += ['--hf-subset', parts[2]]
        if a.hf_name:
            x += ['--hf-name', a.hf_name]
        if ol:
            x += ['--hf-output-len'] + ol
        if a.skip_chat_template:
            x.append('--skip-chat-template')
        return x, 'hf_' + re.sub(r'\W+', '_', parts[0])[-40:]
    if kind == 'sharegpt':                    # sharegpt:<json 路径>
        return ['--dataset-name', 'sharegpt', '--dataset-path', rest] + (['--sharegpt-output-len'] + ol if ol else []), 'sharegpt'
    if kind == 'custom':                      # custom:<jsonl 路径>，每行 {"prompt": "..."}
        x = ['--dataset-name', 'custom', '--dataset-path', rest] + (['--custom-output-len'] + ol if ol else [])
        if a.skip_chat_template:
            x.append('--skip-chat-template')
        return x, 'custom'
    if kind == 'prefix':                      # prefix:<共享前缀长>:<后缀长>:<前缀个数>，模拟同一份系统提示 / 代码库上下文
        pl, sl, npf = (rest.split(':') + ['', '', ''])[:3]
        return (['--dataset-name', 'prefix_repetition', '--prefix-repetition-prefix-len', pl or '8000',
                 '--prefix-repetition-suffix-len', sl or '1000', '--prefix-repetition-num-prefixes', npf or '4',
                 '--prefix-repetition-output-len', str(a.output_len or osl)],
                f'prefix{pl}_{sl}_{npf}')
    raise SystemExit(f'不认识的 --dataset：{a.dataset}')


def ds_label(ds):
    """提交用的数据集名：custom:<…/realtext_3k.jsonl> 记为 realtext:3k（tools/build_realtext.py 生成的真实文本），其余原样"""
    m = re.match(r'^custom:.*?realtext_([\w.-]+)\.jsonl$', ds or '')
    return f'realtext:{m.group(1)}' if m else ds


def nominal_hit(a):
    """读不到服务端命中率时的预设值：prefix 数据集按前缀占比，其他真实数据集按 0"""
    if a.dataset.startswith('prefix:'):
        pl, sl = (a.dataset.split(':')[1:] + ['8000', '1000'])[:2]
        return round(100 * float(pl or 8000) / (float(pl or 8000) + float(sl or 1000)), 1)
    return 0


def parse_slo(s):
    """--slo 'ttft:3000,tpot:50' → (3000.0, 50.0)，单位 ms"""
    d = dict(x.split(':') for x in s.split(',') if ':' in x)
    return float(d.get('ttft', 3000)), float(d.get('tpot', 50))


def slo_fields(reqs, total, duration, slo):
    """reqs：[(ttft_ms, tpot_ms 或 None, 输出 token 数)]，只含成功的请求；total：发出的请求数（失败算不达标）"""
    t_ms, p_ms = slo
    good = [(f, t, o) for f, t, o in reqs if f <= t_ms and (t is None or t <= p_ms)]
    if not total or not duration:
        return {}
    return {'slo_ttft': t_ms, 'slo_tpot': p_ms, 'slo_pct': round(100 * len(good) / total, 1),
            'good_tps': round(sum(o for _, _, o in good) / duration, 2), 'goodput': round(len(good) / duration, 3)}


def slo_from_detailed(res, total, slo):
    """从 vllm bench serve --save-detailed 的逐请求数据统计达标；ttfts / itls 的单位按中位数与 median_ttft_ms 对比判断"""
    ttfts, itls, outs = res.get('ttfts'), res.get('itls'), res.get('output_lens')
    if not ttfts or not outs:
        return {}
    errs = res.get('errors') or [''] * len(ttfts)
    med = statistics.median([x for x in ttfts if x and x > 0] or [0])
    k = 1000.0 if med and res.get('median_ttft_ms') and abs(med * 1000 - res['median_ttft_ms']) < abs(med - res['median_ttft_ms']) else 1.0
    reqs = []
    for i, f in enumerate(ttfts):
        o = outs[i] if i < len(outs) else 0
        if not f or f <= 0 or not o or (i < len(errs) and errs[i]):
            continue
        it = itls[i] if itls and i < len(itls) else None
        # MTP 一个 chunk 多个 token，不能用 mean(ITL)；按逐 token TPOT = (e2el-ttft)/(o-1)
        tp = (sum(it) * k / (o - 1)) if it and o > 1 else None
        reqs.append((f * k, tp, o))
    return slo_fields(reqs, total, res.get('duration'), slo)


# ---------- 精度检查：固定题集 ardelva-qc-v1（由种子确定性生成，答案由脚本直接算出）----------
QC_SET = 'ardelva-qc-v1'
QC_TAIL = '\n\nThink as needed, then give only the final answer on the last line in the form: ANSWER: <answer>'


def qc_items():
    rnd = random.Random(20261103); items = []
    for _ in range(12):
        x, y, z = rnd.randint(1000, 9999), rnd.randint(12, 98), rnd.randint(100, 9999)
        items.append(('num', f'Compute {x} * {y} + {z}.', str(x * y + z)))
    for _ in range(8):
        b, n, out = rnd.randint(40, 400), rnd.randint(6, 48), rnd.randint(50, 900)
        items.append(('num', f'A warehouse has {b} boxes and each box holds {n} parts. {out} parts are shipped out. How many parts remain in the warehouse?', str(b * n - out)))
    for _ in range(4):
        xs = [rnd.randint(1, 30) for _ in range(8)]
        items.append(('num', f'What does this Python code print?\n```python\nxs = {xs}\nprint(sum(x * x for x in xs if x % 2 == 0))\n```', str(sum(x * x for x in xs if x % 2 == 0))))
    for _ in range(4):
        n = rnd.randint(8, 20); r = 1
        for i in range(1, n + 1):
            r = r * i % 1000
        items.append(('num', f'What does this Python code print?\n```python\nr = 1\nfor i in range(1, {n} + 1):\n    r = r * i % 1000\nprint(r)\n```', str(r)))
    for _ in range(6):
        xs = rnd.sample(range(-60, 200), 9)
        items.append(('list', 'Sort these integers in ascending order and give them comma-separated: ' + ', '.join(map(str, xs)) + '.', ','.join(map(str, sorted(xs)))))
    syl = ['ra', 'ten', 'mo', 'lis', 'ver', 'qua', 'dor', 'sel']
    for _ in range(6):
        w = ''.join(rnd.choice(syl) for _ in range(rnd.randint(3, 5))); ch = rnd.choice(sorted(set(w)))
        items.append(('num', f'How many times does the letter "{ch}" appear in the string "{w}"?', str(w.count(ch))))
    return items


def qc_parse(text):
    text = re.sub(r'<think>.*?</think>', '', text or '', flags=re.S)
    m = re.findall(r'ANSWER:\s*(.+)', text)
    return m[-1].strip().strip('`*. ') if m else None


def qc_norm(kind, v):
    if v is None:
        return None
    if kind == 'num':
        m = re.findall(r'-?\d[\d,]*', v)
        return m[-1].replace(',', '') if m else v
    if kind == 'list':
        return ','.join(re.findall(r'-?\d+', v))
    return v


def qc_ask(a, prompt):
    body = json.dumps({'model': a.served_model_name or a.model, 'messages': [{'role': 'user', 'content': prompt}],
                       'temperature': 0, 'max_tokens': a.accuracy_max_tokens}).encode()
    req = urllib.request.Request(a.base_url.rstrip('/') + '/v1/chat/completions', data=body, method='POST',
                                 headers={'content-type': 'application/json'})
    if os.environ.get('OPENAI_API_KEY'):
        req.add_header('authorization', 'Bearer ' + os.environ['OPENAI_API_KEY'])
    try:
        with urllib.request.urlopen(req, timeout=900) as r:
            msg = json.loads(r.read().decode())['choices'][0]['message']
        return msg.get('content') or msg.get('reasoning_content') or msg.get('reasoning') or ''
    except Exception as e:
        return f'ERROR {e}'


def run_accuracy(a, outdir):
    from concurrent.futures import ThreadPoolExecutor
    items = qc_items()
    print(f'== 精度检查 {QC_SET}：{len(items)} 题，temperature 0，{a.accuracy_par} 路并行', flush=True)
    t0 = time.time()
    with ThreadPoolExecutor(a.accuracy_par) as ex:
        texts = list(ex.map(lambda it: qc_ask(a, it[1] + QC_TAIL), items))
    answers = [qc_norm(k, qc_parse(t)) for (k, _, _), t in zip(items, texts)]
    ok = [ans == qc_norm(k, gold) for (k, _, gold), ans in zip(items, answers)]
    res = {'acc_set': QC_SET, 'acc_n': len(items), 'acc_pct': round(100 * sum(ok) / len(items), 1)}
    if a.accuracy_ref:
        ref = json.loads(Path(a.accuracy_ref).read_text())
        if ref.get('set') == QC_SET and len(ref.get('answers', [])) == len(items):
            res['acc_ref_pct'] = round(100 * sum(x == y for x, y in zip(answers, ref['answers'])) / len(items), 1)
        else:
            print('  参考答案文件的题集版本不符，跳过逐题比对', file=sys.stderr)
    if a.accuracy_save:
        Path(a.accuracy_save).write_text(json.dumps({'set': QC_SET, 'model': a.model, 'date': date.today().isoformat(), 'answers': answers}, ensure_ascii=False, indent=1))
    (outdir / 'ardelva_accuracy.json').write_text(json.dumps({**res, 'detail': [
        {'q': q[:120], 'gold': g, 'got': ans, 'ok': o} for (_, q, g), ans, o in zip(items, answers, ok)]}, ensure_ascii=False, indent=1))
    wrong = [i + 1 for i, o in enumerate(ok) if not o]
    print(f"    → 答对 {res['acc_pct']}%" + (f"，与参考答案一致 {res['acc_ref_pct']}%" if 'acc_ref_pct' in res else '') +
          (f"；错题 {wrong}" if wrong else '') + f' · 用时 {time.time() - t0:.0f}s（明细 {outdir / "ardelva_accuracy.json"}）', flush=True)
    return res


def run_point(a, isl, osl, hit, c, outdir, rate=None):
    dargs, tag = dataset_args(a, isl, osl, hit)
    if rate:   # 按速率：请求数 = 速率 × 时长，c 为并发上限
        nreq = max(a.min_requests, int(round(rate * a.rate_seconds)))
        name = f'{tag}_r{rate:g}.json'
    else:
        nreq = max(a.min_requests, a.req_per_conc * c)
        name = f'{tag}_c{c}.json'

    cold = a.dataset == 'random' and hit is COLD
    if cold:   # 独立种子：不与前面任何点、也不与自己的预热生成同样的提示词
        a._cold_n = getattr(a, '_cold_n', 0) + 1
        seed_m, seed_w = a.seed + 100003 * a._cold_n, a.seed + 100003 * a._cold_n + 7
    else:
        seed_m = seed_w = a.seed

    def cmd(n, save):
        x = [a.vllm, 'bench', 'serve', '--backend', a.backend, '--base-url', a.base_url, '--endpoint', a.endpoint,
             '--model', a.model] + dargs + (['--ignore-eos'] if a.dataset == 'random' or a.output_len else []) + [
             '--num-prompts', str(n), '--max-concurrency', str(c), '--request-rate', f'{rate:g}' if rate and save else 'inf',
             '--percentile-metrics', 'ttft,tpot,itl,e2el', '--metric-percentiles', '50,99', '--seed', str(seed_m if save else seed_w)]
        if rate and save and a.burstiness != 1.0:
            x += ['--burstiness', str(a.burstiness)]
        if a.tokenizer:
            x += ['--tokenizer', a.tokenizer]
        if a.served_model_name:
            x += ['--served-model-name', a.served_model_name]
        if a.extra:
            x += a.extra.split()
        if save:
            x += ['--save-result', '--save-detailed', '--result-dir', str(outdir), '--result-filename', name]
        return x

    # 预热：同样的负载跑少量请求，不计入结果（也让共享前缀进入缓存）
    if cold:
        reset_prefix_cache(a.base_url)   # vLLM 以 VLLM_SERVER_DEV_MODE=1 启动才开放；不可用时靠独立种子
    if a.warmup > 0:
        print(f'  预热 C{c if not rate else min(c, 8)} …', flush=True)
        subprocess.run(cmd(max(c if not rate else min(c, 8), a.warmup), False), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    before = scrape(a.base_url)
    print(f'  测量 {tag} ' + (f'速率={rate:g}/s（并发上限 {c}）' if rate else f'C={c}') + f' 请求数={nreq}', flush=True)
    t0 = time.time()
    p = subprocess.run(cmd(nreq, True), capture_output=True, text=True)
    if p.returncode != 0 or not (outdir / name).exists():
        err = (p.stdout or '')[-2000:] + '\n' + (p.stderr or '')[-2000:]
        print(err, file=sys.stderr)
        if 'chat_template' in err or 'chat template' in err:
            a._tmpl_err = True
            print('  ✗ tokenizer 没有 chat_template：用 --tokenizer 指向带对话模板的 tokenizer 目录（原厂 HF 仓库即可），'
                  '或加 --skip-chat-template 直接发原文（接受长度会与正式对话略有差异）', file=sys.stderr, flush=True)
        else:
            print(f'  ✗ 测量失败（rc={p.returncode}）', file=sys.stderr, flush=True)
        return None
    after = scrape(a.base_url)
    res = json.loads((outdir / name).read_text())
    done = res.get('completed') or 0
    row = {
        'isl': round(res['total_input_tokens'] / done) if done else isl,
        'osl': round(res['total_output_tokens'] / done) if done else osl,
        'hit': (0.0 if cold else round(hit * 100, 1)) if a.dataset == 'random' else nominal_hit(a),
        'c': c, 'nreq': res.get('num_prompts', nreq),
        'out_tps': round(res['output_throughput'], 2),
        'ttft_p50': round(res['median_ttft_ms'], 2), 'tpot_p50': round(res['median_tpot_ms'], 3),
        'ttft_p99': round(res.get('p99_ttft_ms', 0), 2) or '', 'tpot_p99': round(res.get('p99_tpot_ms', 0), 3) or '',
        'total_tps': round(res.get('total_token_throughput', 0), 2) or '',
        'success': round(100 * done / res.get('num_prompts', nreq), 2) if res.get('num_prompts') else '',
        'accept': '',
        'ds': ds_label(a.dataset) if a.dataset != 'random' else ('random:cold' if cold else 'random'),
    }
    if rate:
        row['rate'] = rate
    row.update(slo_from_detailed(res, res.get('num_prompts', nreq), a.slo))
    # 服务端实测命中率（vLLM V1 指标名；不同版本名字略有差别，取到哪个用哪个）
    hits = delta(after, before, 'vllm:prefix_cache_hits_total', 'vllm:prefix_cache_hits', 'vllm:gpu_prefix_cache_hits_total')
    qs = delta(after, before, 'vllm:prefix_cache_queries_total', 'vllm:prefix_cache_queries', 'vllm:gpu_prefix_cache_queries_total')
    if hits is not None and qs:
        row['hit'] = round(100 * hits / qs, 1)
    # 投机解码平均接受长度 = 1 + 接受 token 数 / 草稿次数
    acc = delta(after, before, 'vllm:spec_decode_num_accepted_tokens_total', 'vllm:spec_decode_num_accepted_tokens')
    drafts = delta(after, before, 'vllm:spec_decode_num_drafts_total', 'vllm:spec_decode_num_drafts')
    if acc is not None and drafts:
        row['accept'] = round(1 + acc / drafts, 2)
    print(f'    → {row["out_tps"]} tok/s · TTFT p50 {row["ttft_p50"]} ms · TPOT p50 {row["tpot_p50"]} ms · '
          f'命中 {row["hit"]}%' + (f' · 达标 {row["slo_pct"]}%（达标产能 {row["good_tps"]} tok/s）' if 'slo_pct' in row else '') +
          f' · 用时 {time.time() - t0:.0f}s', flush=True)
    return row


def mt_spec(ds):
    """multiturn:<起始>:<每轮新增>:<轮数>[:<每轮输出>[:<共享前缀>]]"""
    v = [int(x) for x in ds.split(':')[1:] if x != '']
    start, step, rounds = (v + [60000, 5000, 5])[:3] if len(v) < 3 else v[:3]
    out = v[3] if len(v) > 3 else 500
    common = v[4] if len(v) > 4 else min(15000, start // 4)
    user = step - out
    prefix = start - max(common, user)   # vLLM 工具的首轮：共享前缀算在首轮用户消息的长度里，独有前缀另加
    if user <= 0 or prefix <= 0:
        raise SystemExit(f'multiturn 参数不合理：每轮新增 {step} 要大于输出 {out}，起始 {start} 要大于共享前缀 {common} 和每轮用户消息 {user}')
    return dict(start=start, step=step, rounds=rounds, out=out, common=common, user=user, prefix=prefix,
                label=f'multiturn:{start}+{step}x{rounds}')


def mt_nominal_hit(sp):
    """读不到服务端命中率时的理论值（按 token 加权）：首轮只命中共享前缀，之后每轮命中上一轮的全部上下文"""
    inp = [sp['start'] + k * sp['step'] for k in range(sp['rounds'])]
    hit = [sp['common']] + [x - sp['user'] for x in inp[1:]]
    return round(100 * sum(hit) / sum(inp), 1)


MT_PATCHES = [   # 对 vLLM 多轮工具的两处修补（幂等，带标记）
    ('        payload["max_tokens"] = max_tokens\n',
     '        payload["max_tokens"] = max_tokens\n        payload["ignore_eos"] = True  # ardelva patch: fixed output length, like --ignore-eos\n'),
    ('                    delta = data["choices"][0]["delta"]\n',
     '                    delta = data["choices"][0]["delta"]\n'
     '                    # ardelva patch: reasoning-parser servers stream thinking as delta.reasoning(_content); count it as output\n'
     '                    _txt = delta.get("content") or delta.get("reasoning") or delta.get("reasoning_content")\n'
     '                    if _txt:\n                        delta = {"content": _txt}\n'),
]


def mt_tool(a):
    """vLLM 的多轮压测工具：下载固定版本（或从 --mt-dir 拷贝）到缓存目录，再打两处补丁：
    固定输出长度（ignore_eos）、开了 reasoning parser 时把 reasoning 增量也算作输出"""
    d = Path.home() / '.cache' / 'ardelva' / ('vllm_multi_turn_' + MT_COMMIT[:12] + ('_local' if a.mt_dir else ''))
    d.mkdir(parents=True, exist_ok=True)
    for f in MT_FILES:
        if (d / f).exists():
            continue
        if a.mt_dir:
            (d / f).write_bytes((Path(a.mt_dir) / f).read_bytes())
        else:
            url = f'https://raw.githubusercontent.com/vllm-project/vllm/{MT_COMMIT}/benchmarks/multi_turn/{f}'
            print(f'  下载 {url}', flush=True)
            (d / f).write_bytes(urllib.request.urlopen(url, timeout=60).read())
    main = d / MT_FILES[0]
    src = main.read_text()
    if 'ardelva patch' not in src:
        for old, new in MT_PATCHES:
            if old not in src:
                raise SystemExit(f'多轮工具版本不符，找不到需要修补的位置：{old.strip()[:60]}（用固定版本或更新本脚本）')
            src = src.replace(old, new, 1)
        main.write_text(src)
    return main


def mt_corpus(path, tokens, seed):
    """生成足够长的语料（伪词）。多轮工具从语料里切片拼出每段上下文，内容由语料决定；
    每个测量点用不同的种子，否则同样的会话会命中上一个点（或冒烟测试）留在服务端的前缀缓存"""
    rnd = random.Random(seed); syl = ['ka', 'lo', 'mi', 'ten', 'ra', 'sul', 've', 'dor', 'an', 'qui', 'pe', 'zo', 'ri', 'mon', 'tal', 'es', 'vin', 'cor', 'ul', 'ba']
    words = [''.join(rnd.choice(syl) for _ in range(rnd.randint(1, 3))) for _ in range(4000)]
    with open(path, 'w') as f:
        written = 0
        while written < tokens * 9:   # 按每 token 至多约 9 个字符估，任何分词器都够用
            line = ' '.join(rnd.choice(words) for _ in range(40)) + '.\n'
            f.write(line); written += len(line)


def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(q / 100 * (len(xs) - 1))))] if xs else None


def run_mt_point(a, c, outdir):
    sp = mt_spec(a.dataset)
    tool = mt_tool(a)
    nconv = max(8, 2 * c)
    corpus = outdir / f'mt_corpus_c{c}.txt'
    mt_corpus(corpus, sp['common'] + nconv * (sp['prefix'] + 50) + sp['rounds'] * 2 * sp['user'] + 20000, f'{time.time_ns()}-{c}')
    conf = outdir / f'mt_c{c}.json'
    d = lambda v: {'distribution': 'constant', 'value': v}
    conf.write_text(json.dumps({'filetype': 'generate_conversations', 'num_conversations': nconv, 'text_files': [str(corpus)],
        'prompt_input': {'num_turns': d(2 * sp['rounds']), 'common_prefix_num_tokens': d(sp['common']),
                         'prefix_num_tokens': d(sp['prefix']), 'num_tokens': d(sp['user'])},
        'prompt_output': {'num_tokens': d(sp['out'])}}, indent=2))
    stats = outdir / f'mt_c{c}_stats.json'
    cmd = [sys.executable, str(tool), '-m', a.tokenizer or a.model, '--served-model-name', a.served_model_name or a.model,
           '-u', a.base_url, '-i', str(conf), '-p', str(c), '-k', str(c), '--seed', str(a.seed),
           '--limit-min-tokens', str(sp['out']), '--limit-max-tokens', str(sp['out']), '--request-timeout-sec', '1800',
           '--stats-json-output', str(stats)]
    if os.environ.get('OPENAI_API_KEY'):   # vLLM 用 --api-key 启动时
        cmd += ['--api-key', os.environ['OPENAI_API_KEY']]
    before = scrape(a.base_url)
    print(f"  测量 {sp['label']} C={c} 会话={nconv}", flush=True)
    t0 = time.time()
    p = subprocess.run(cmd, capture_output=True, text=True, cwd=str(tool.parent))
    if p.returncode != 0 or not stats.exists():
        print(p.stdout[-2000:], p.stderr[-3000:], sep='\n', file=sys.stderr)
        return None
    after = scrape(a.base_url)
    reqs = [r for r in json.loads(stats.read_text()) if r.get('ttft_ms', -1) > 0]
    if not reqs:
        return None
    span = (max(r['start_time_ms'] + r['latency_ms'] for r in reqs) - min(r['start_time_ms'] for r in reqs)) / 1000
    outs = sum(r['output_num_tokens'] for r in reqs); ins = sum(r['input_num_tokens'] for r in reqs)
    tpot = [(r['latency_ms'] - r['ttft_ms']) / max(1, r['output_num_tokens'] - 1) for r in reqs]
    ttft = [r['ttft_ms'] for r in reqs]
    first = [r['ttft_ms'] for r in reqs if r['input_num_turns'] <= 1]
    later = [r['ttft_ms'] for r in reqs if r['input_num_turns'] > 1]
    row = {'isl': round(ins / len(reqs)), 'osl': round(outs / len(reqs)),
           'hit': mt_nominal_hit(sp),
           'c': c, 'nreq': len(reqs), 'out_tps': round(outs / span, 2),
           'ttft_p50': round(pct(ttft, 50), 2), 'tpot_p50': round(pct(tpot, 50), 3),
           'ttft_p99': round(pct(ttft, 99), 2), 'tpot_p99': round(pct(tpot, 99), 3),
           'total_tps': round((ins + outs) / span, 2), 'success': '', 'accept': '',
           'ttft_t1': round(statistics.median(first), 1) if first else '', 'ttft_tn': round(statistics.median(later), 1) if later else '',
           'ds': sp['label']}
    row.update(slo_fields([(r['ttft_ms'], t, r['output_num_tokens']) for r, t in zip(reqs, tpot)], len(reqs), span, a.slo))
    hits = delta(after, before, 'vllm:prefix_cache_hits_total', 'vllm:prefix_cache_hits', 'vllm:gpu_prefix_cache_hits_total')
    qs = delta(after, before, 'vllm:prefix_cache_queries_total', 'vllm:prefix_cache_queries', 'vllm:gpu_prefix_cache_queries_total')
    if hits is not None and qs:
        row['hit'] = round(100 * hits / qs, 1)
    acc = delta(after, before, 'vllm:spec_decode_num_accepted_tokens_total', 'vllm:spec_decode_num_accepted_tokens')
    drafts = delta(after, before, 'vllm:spec_decode_num_drafts_total', 'vllm:spec_decode_num_drafts')
    if acc is not None and drafts:
        row['accept'] = round(1 + acc / drafts, 2)
    print(f"    → {row['out_tps']} tok/s · 平均输入 {row['isl']} · 命中 {row['hit']}% · TTFT 首轮 {row['ttft_t1']} ms / 后续 {row['ttft_tn']} ms · "
          f"TPOT p50 {row['tpot_p50']} ms · 用时 {time.time() - t0:.0f}s", flush=True)
    return row


def rows_from_csv(path):
    rows = []
    with open(path, newline='') as f:
        for r in csv.DictReader(f):
            row = {}
            for k in CSV_COLS:
                v = (r.get(k) or '').strip()
                if v == '':
                    continue
                row[k] = v if k == 'ds' else (float(v) if '.' in v or 'e' in v.lower() else int(v))
            if row:
                rows.append(row)
    return rows


def rows_from_json(paths, a):
    """把 `vllm bench serve --save-result` 的结果 JSON 转成测量点。并发取 max_concurrency；
    命中率优先从文件名里的 hitNN 读（本脚本的命名），否则用 --hit"""
    files = []
    for p in paths:
        p = Path(p)
        files += sorted(p.glob('*.json')) if p.is_dir() else [p]
    rows = []
    for fp in files:
        try:
            res = json.loads(fp.read_text())
        except Exception:
            continue
        if 'output_throughput' not in res:
            continue
        done = res.get('completed') or 0
        m = re.search(r'hit(\d+(?:\.\d+)?)', fp.name)
        c = res.get('max_concurrency') or (int(re.search(r'_c(\d+)', fp.name).group(1)) if re.search(r'_c(\d+)', fp.name) else None)
        if not c or not done:
            print(f'  跳过 {fp.name}：缺少并发或完成数', file=sys.stderr); continue
        rows.append({
            'isl': round(res['total_input_tokens'] / done), 'osl': round(res['total_output_tokens'] / done),
            'hit': float(m.group(1)) if m else a.hit, 'c': int(c), 'nreq': res.get('num_prompts', done),
            'out_tps': round(res['output_throughput'], 2), 'ds': a.ds or 'random',
            'ttft_p50': round(res['median_ttft_ms'], 2), 'tpot_p50': round(res['median_tpot_ms'], 3),
            **({'ttft_p99': round(res['p99_ttft_ms'], 2)} if res.get('p99_ttft_ms') else {}),
            **({'tpot_p99': round(res['p99_tpot_ms'], 3)} if res.get('p99_tpot_ms') else {}),
            **({'total_tps': round(res['total_token_throughput'], 2)} if res.get('total_token_throughput') else {}),
            **({'success': round(100 * done / res['num_prompts'], 2)} if res.get('num_prompts') else {}),
        })
    rows.sort(key=lambda r: (r['isl'], r['c']))
    return rows


def check_keys(a):
    """提交前校验三把 key：必须从部署测算「方案标识」复制（网络不通时跳过，提交时服务器还会再查）"""
    q = urllib.parse.urlencode({'m': a.model_key, 'd': a.deploy_key, 'h': a.hw_key})
    try:
        req = urllib.request.Request(a.site.rstrip('/') + '/api/keycheck?' + q, headers={'user-agent': UA})
        with urllib.request.urlopen(req, timeout=15) as r:
            j = json.loads(r.read().decode())
    except Exception as e:
        print(f'（未能预检 key：{e}；提交时再校验）', file=sys.stderr)
        return
    if not j.get('ok'):
        print('三把 key 不是部署测算的格式，部署测算匹配不到这条实测：', file=sys.stderr)
        for p in j.get('problems', []):
            print('  - ' + p, file=sys.stderr)
        print(j.get('hint', ''), file=sys.stderr)
        sys.exit(2)


def post(a, rows):
    data = {'date': a.date or date.today().isoformat(), 'submitter': a.submitter, 'engine': a.engine,
            'tool': a.tool or f'vllm bench serve (ardelva_bench.py, {a.dataset}{" + " + a.hf_coding if a.hf_coding else ""}, --max-concurrency C)'[:120],
            'points': [{k: v for k, v in r.items() if v != ''} for r in rows]}
    if a.spec:
        data['spec'] = a.spec
    if a.link:
        data['link'] = a.link
    skipped = getattr(a, '_skipped', [])
    if a.notes or skipped:
        data['notes'] = '; '.join(([a.notes] if a.notes else []) + skipped)
    if getattr(a, 'acc', None):
        data.update({k: v for k, v in a.acc.items() if k in ('acc_set', 'acc_n', 'acc_pct', 'acc_ref_pct')})
    payload = {'keys': {'modelKey': a.model_key, 'deployKey': a.deploy_key, 'hwKey': a.hw_key}, 'data': data}
    if a.new_record:
        payload['mode'] = 'new'
    body = json.dumps(payload).encode()
    req = urllib.request.Request(a.site.rstrip('/') + '/api/results', data=body, method='POST',
                                 headers={'content-type': 'application/json', 'user-agent': UA})
    if os.environ.get('ARDELVA_KEY'):
        req.add_header('x-ardelva-key', os.environ['ARDELVA_KEY'])
    tok = a.token or os.environ.get('ARDELVA_TOKEN')
    if tok:   # 个人令牌（ardelva.ai/me 生成）：记录归在你的账号下
        req.add_header('x-ardelva-token', tok)
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            j = json.loads(r.read().decode())
            if j.get('merged'):
                print(f"已合并进记录 #{j['id']}：覆盖 {j['replaced']} 个点，新增 {j['added']} 个点，现共 {j['points']} 个点")
            else:
                print(f"已提交：记录 #{j.get('id')}{'（已核验）' if j.get('verified') else '（读者提交）'}")
    except urllib.error.HTTPError as e:
        print('提交失败：', e.code, e.read().decode(), file=sys.stderr)
        if e.code == 401:
            print('提交需要登录：在 https://ardelva.ai/zh/me/ 生成个人令牌，用 --token ardl_… 或环境变量 ARDELVA_TOKEN 提交', file=sys.stderr)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--base-url', help='vLLM 服务地址，例如 http://10.0.0.10:8000')
    ap.add_argument('--model', help='请求里的模型名（与服务端 served-model-name 一致）')
    ap.add_argument('--tokenizer', help='本地 tokenizer 路径（默认用 --model 去 Hugging Face 取）')
    ap.add_argument('--served-model-name')
    ap.add_argument('--endpoint', default='/v1/completions')
    ap.add_argument('--dataset', default='random',
                    help="random（默认，按矩阵造数据）| hf:<repo>[:split[:subset]] | sharegpt:<json> | custom:<jsonl> | prefix:<前缀长>:<后缀长>:<前缀个数>")
    ap.add_argument('--conc', help='非 random 数据集时扫描的并发列表（默认 1,4,8,16,32,64；--hf-coding 默认 1,4,8,16,32）')
    ap.add_argument('--output-len', type=int, help='非 random 数据集时固定输出长度（配合 --ignore-eos）；不填则用数据集自带长度')
    ap.add_argument('--hf-name', help='--dataset hf 用本地路径时，对应的 Hugging Face 名称')
    ap.add_argument('--skip-chat-template', action='store_true', help='不套对话模板直接发原文：custom 数据已套好模板时，或 hf 数据集而 tokenizer 没有 chat_template 时（更好的做法是 --tokenizer 指向带模板的 tokenizer）')
    ap.add_argument('--backend', default='vllm', help='vllm（/v1/completions）或 openai-chat（配 --endpoint /v1/chat/completions）')
    ap.add_argument('--matrix', help="自定义扫描矩阵，例如 '3000:1000:50:1,4,8;60000:1000:95:8,16,32'（输入:输出:命中%%:并发；命中写 cold = 完全冷启动）")
    ap.add_argument('--req-per-conc', type=int, default=4, help='每个点的请求数 = 并发 × 这个值（默认 4，即每个点约 4 轮；吞吐收尾误差约 1/轮数）')
    ap.add_argument('--min-requests', type=int, default=32, help='每个点至少的请求数（默认 32，p50 足够稳；C32 起样本 ≥ 128，p99 也有意义）')
    ap.add_argument('--stop-ttft', type=float, default=3000, help='过载即停：某点首字 p50 超过这个毫秒数且达标率低于 --stop-slo，这组不再测更高并发（默认 3000；0 关闭）')
    ap.add_argument('--stop-slo', type=float, default=50, help='过载即停的达标率门槛 %%（默认 50）')
    ap.add_argument('--warmup', type=int, default=8, help='每个点预热请求数（0 关闭）')
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--extra', help='原样传给 vllm bench serve 的其他参数')
    ap.add_argument('--vllm', default='vllm', help='vllm 命令路径')
    ap.add_argument('--out', default='ardelva_bench_' + time.strftime('%Y%m%d_%H%M%S'), help='结果目录')
    # 提交
    ap.add_argument('--post', action='store_true', help='测完提交到 ardelva.ai')
    ap.add_argument('--token', help='个人令牌（在 ardelva.ai/me 生成；也可用环境变量 ARDELVA_TOKEN）：提交需要登录身份，记录归在你的账号下')
    ap.add_argument('--site', default='https://ardelva.ai')
    ap.add_argument('--model-key'); ap.add_argument('--deploy-key'); ap.add_argument('--hw-key')
    ap.add_argument('--submitter', default='Ardelva Lab'); ap.add_argument('--engine')
    ap.add_argument('--tool'); ap.add_argument('--spec', choices=['on', 'off'])
    ap.add_argument('--link'); ap.add_argument('--notes')
    ap.add_argument('--new-record', action='store_true', help='带核验密钥时不合并进已有记录，另起一条（默认：同配置、同引擎、同投机开关的已核验记录里同点覆盖、新点追加）')
    ap.add_argument('--date', help='测试日期 YYYY-MM-DD（提交已有结果时填当时的日期，默认今天）')
    # 不重测，提交已有结果
    ap.add_argument('--from-csv', help='直接提交已有的 ardelva_points.csv，不跑压测')
    ap.add_argument('--from-json', nargs='+', help='直接提交 vllm bench serve 保存的结果 JSON（目录或文件），不跑压测')
    ap.add_argument('--ds', help='--from-json 时这些结果用的数据集（默认 random），如 hf:likaixin/InstructCoder')
    ap.add_argument('--hf-coding', nargs='?', const=HF_CODING, help=f'random 跑完后再用真实编程数据集扫一遍并发（默认 {HF_CODING}，并发取 --conc，输出取 --output-len 或 1000）')
    ap.add_argument('--realtext', help='tools/build_realtext.py 生成的目录：random 跑完后再依次跑其中的 realtext_3k / realtext_10k（等）jsonl，并发取 --conc，输出取 --output-len 或 1000')
    ap.add_argument('--think', action='store_true', help='配合 --realtext：再用 realtext_10k 跑一组长输出（思考模式 / 代码生成），输出 --think-output，并发 --think-conc')
    ap.add_argument('--think-output', type=int, default=4000, help='思考模式组的输出长度（默认 4000）')
    ap.add_argument('--think-conc', default='1,8,32', help='思考模式组的并发（默认 1,8,32）')
    ap.add_argument('--multiturn', nargs='?', const=MULTITURN, help=f'random 跑完后再跑多轮 Agent 负载（默认 {MULTITURN}，并发取 --conc 或 1,4,8,16,32）')
    ap.add_argument('--api-key', help='推理服务的 API key（vLLM 用 --api-key 启动时；也可设环境变量 OPENAI_API_KEY）')
    ap.add_argument('--mt-dir', help='本地 vllm/benchmarks/multi_turn 目录（不填则下载固定版本到 ~/.cache/ardelva）')
    ap.add_argument('--hit', type=float, default=0, help='--from-json 时文件名里没有命中率，用这个值（%%）')
    # 按速率压测与服务目标
    ap.add_argument('--rate', help='按请求速率扫描（个/秒，泊松到达），如 0.5,1,2,4；对矩阵每一行（或数据集）再跑一遍，替代并发扫描时配合 --matrix')
    ap.add_argument('--rate-cap', type=int, default=256, help='按速率压测时的并发上限（默认 256）')
    ap.add_argument('--rate-seconds', type=int, default=120, help='按速率压测时每个点的目标时长（秒），请求数 = 速率 × 时长（默认 120）')
    ap.add_argument('--burstiness', type=float, default=1.0, help='到达的突发程度（1 = 泊松；越小越突发），传给 vllm bench serve')
    ap.add_argument('--slo', default='ttft:3000,tpot:50', help='服务目标（ms），所有点按它统计达标率与达标产能（默认 ttft:3000,tpot:50）')
    ap.add_argument('--only-rate', action='store_true', help='只做按速率扫描，不做并发扫描')
    # 精度检查
    ap.add_argument('--accuracy', action='store_true', help=f'压测前跑精度检查（{QC_SET}，40 题）并随结果提交')
    ap.add_argument('--accuracy-only', action='store_true', help='只跑精度检查，不压测')
    ap.add_argument('--accuracy-min', type=float, help='答对比例低于这个值（%%）就停止，不压测、不提交')
    ap.add_argument('--accuracy-ref', help='参考答案文件（--accuracy-save 生成），逐题比对答案是否一致')
    ap.add_argument('--accuracy-save', help='把本次答案存为参考答案文件')
    ap.add_argument('--accuracy-par', type=int, default=8, help='精度检查并行请求数（默认 8）')
    ap.add_argument('--accuracy-max-tokens', type=int, default=4096, help='每题最多生成 token（思考模型需要留足，默认 4096）')
    a = ap.parse_args()
    a._skipped = []   # 过载即停跳过的并发（写进提交的备注）
    a.slo = parse_slo(a.slo); a.acc = None
    if a.accuracy_only or a.accuracy_ref or a.accuracy_save or a.accuracy_min is not None:
        a.accuracy = True
    if a.api_key:
        os.environ['OPENAI_API_KEY'] = a.api_key   # vllm bench serve 和多轮工具都从这里取
    if a.from_csv or a.from_json:
        a.post = True
        if not a.dataset or a.dataset == 'random':
            a.dataset = 'existing results'
    elif not (a.base_url and a.model):
        ap.error('需要 --base-url 和 --model（或用 --from-csv / --from-json 提交已有结果）')
    if a.post and not (a.model_key and a.deploy_key and a.hw_key and a.engine):
        ap.error('--post 需要 --model-key --deploy-key --hw-key --engine')
    if a.post:
        check_keys(a)   # 先确认三把 key 是部署测算的格式，免得测完几小时才在提交时被拒

    if a.from_csv or a.from_json:
        rows = rows_from_csv(a.from_csv) if a.from_csv else rows_from_json(a.from_json, a)
        accf = Path(a.from_csv).with_name('ardelva_accuracy.json') if a.from_csv else None
        if accf and accf.exists():   # 同目录里有精度检查结果就一起提交
            a.acc = json.loads(accf.read_text()); print(f"附带精度检查：{a.acc.get('acc_set')} {a.acc.get('acc_pct')}%")
        print(f'读到 {len(rows)} 个点' + (f'（超过 {MAX_POINTS} 个，只提交前 {MAX_POINTS} 个）' if len(rows) > MAX_POINTS else ''))
        if not rows:
            sys.exit('没有可提交的测量点')
        for r in rows:
            print(f"  输入 {r['isl']} / 输出 {r['osl']} / 命中 {r['hit']}% / C{r['c']} → {r['out_tps']} tok/s")
        post(a, rows[:MAX_POINTS])
        return
    outdir = Path(a.out).resolve(); outdir.mkdir(parents=True, exist_ok=True)   # 绝对路径：多轮工具在自己的目录里运行
    if a.accuracy:
        a.acc = run_accuracy(a, outdir)
        if a.accuracy_min is not None and a.acc['acc_pct'] < a.accuracy_min:
            sys.exit(f"精度检查 {a.acc['acc_pct']}% 低于 --accuracy-min {a.accuracy_min}%：停止，不压测、不提交（明细见 ardelva_accuracy.json）")
        if a.accuracy_only:
            return
    if a.dataset == 'random':
        matrix = parse_matrix(a.matrix) if a.matrix else DEFAULT_MATRIX
    else:   # 真实数据集：输入长度由数据决定，只扫并发；命中率用服务端 /metrics 实测值
        matrix = [(0, a.output_len or 0, 0.0, [int(c) for c in (a.conc or '1,4,8,16,32,64').split(',')])]
    runs = [(a.dataset, matrix)]
    if a.multiturn and a.dataset == 'random':
        runs.append((a.multiturn, [(0, 0, 0.0, [int(c) for c in (a.conc or '1,4,8,16,32').split(',')])]))
    if a.realtext and a.dataset == 'random':
        for fn in sorted(Path(a.realtext).glob('realtext_*.jsonl'), key=lambda x: (len(x.stem), x.stem)):
            if fn.stem.endswith('60k') and a.multiturn:
                continue   # 60K 已由多轮负载覆盖
            cs = [int(c) for c in (a.conc or ('1,4,8,16,32' if fn.stem.endswith('32k') else '1,4,8,16,32,64')).split(',')]
            runs.append((f'custom:{fn.resolve()}', [(0, a.output_len or 1000, 0.0, cs)]))
        t10 = Path(a.realtext) / 'realtext_10k.jsonl'
        if a.think and t10.exists():   # 思考模式：10K 输入、长输出
            runs.append((f'custom:{t10.resolve()}', [(0, a.think_output, 0.0, [int(c) for c in a.think_conc.split(',')])]))
    if a.hf_coding and a.dataset == 'random':
        runs.append((a.hf_coding, [(0, a.output_len or 1000, 0.0, [int(c) for c in (a.conc or '1,4,8,16,32').split(',')])]))
    rows = []
    out0, skip0, failed = a.output_len, a.skip_chat_template, []
    for ds, mtx in runs:
        a.dataset = ds
        a.skip_chat_template = skip0 or 'realtext_' in ds   # 真实文本已按 tokenizer 截好长度，不再套对话模板；其余组按命令行
        n0, a._tmpl_err = len(rows), False
        if ds != 'random' and not ds.startswith('multiturn'):
            a.output_len = mtx[0][1] or out0   # 每组负载自己的输出长度（思考模式组为 --think-output）
        for isl, osl, hit, cs in mtx:
            print(f'== 输入 {isl} / 输出 {osl} / 目标命中 ' + ('完全冷启动' if hit is COLD else f'{hit:.0%}') if a.dataset == 'random' else f'== 数据集 {a.dataset}', flush=True)
            for c in ([] if a.only_rate and a.rate else cs):
                r = run_mt_point(a, c, outdir) if a.dataset.startswith('multiturn') else run_point(a, isl, osl, hit, c, outdir)
                if r:
                    rows.append(r)
                if a._tmpl_err:
                    break   # 缺对话模板时每个并发都会同样失败，不再空跑
                rest = [x for x in cs if x > c]
                if r and rest and a.stop_ttft and r.get('ttft_p50', 0) > a.stop_ttft and r.get('slo_pct', 0) < a.stop_slo:
                    msg = f"{ds_label(a.dataset) if a.dataset != 'random' else f'random {isl}/{osl}'} C{'/C'.join(map(str, rest))} 跳过（C{c} 已过载：首字 p50 {r['ttft_p50'] / 1000:.1f} s" + (f"、达标 {r['slo_pct']}%" if 'slo_pct' in r else '') + "）"
                    print('  ⏭ ' + msg, flush=True)
                    a._skipped.append(msg)
                    break
            if a.rate and not a.dataset.startswith('multiturn'):   # 按速率扫描（多轮负载的会话节奏由工具控制，不支持）
                for rt in [float(x) for x in a.rate.split(',')]:
                    r = run_point(a, isl, osl, hit, a.rate_cap, outdir, rate=rt)
                    if r:
                        rows.append(r)
            if a._tmpl_err:
                break
        if len(rows) == n0:
            failed.append(ds_label(ds) + ('（缺 chat_template）' if a._tmpl_err else ''))
            print(f'  ✗ {ds_label(ds)} 这一组 0 个点', file=sys.stderr, flush=True)
    a.dataset = runs[0][0]
    csv_path = outdir / 'ardelva_points.csv'
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLS, extrasaction='ignore'); w.writeheader(); w.writerows(rows)
    print(f'\n共 {len(rows)} 个点，CSV：{csv_path}（可直接粘贴到部署测算的「粘贴 CSV 导入」）')
    if a.post and rows:
        post(a, rows[:MAX_POINTS])
    if failed:   # 有整组没测出点：非零退出，免得外层脚本当成成功接着跑
        print(f'以下负载组 0 个点：{"、".join(failed)}', file=sys.stderr)
        sys.exit(3)


if __name__ == '__main__':
    main()
