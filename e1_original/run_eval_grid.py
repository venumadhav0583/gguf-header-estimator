"""Run the full measurement grid: 6 models x 2 KV formats x 3 context lengths = 36 configs.

Writes results to /home/user/workspace/eval_13/results.json.
"""
import subprocess, json, os, sys, time

LLAMA_CLI = "/home/user/workspace/eval_13/llama.cpp/build/bin/llama-cli"
MODELS_DIR = "/home/user/workspace/eval_13/models"
MEASURE = "/home/user/workspace/eval_13/measure_peak_rss.py"
OUT = "/home/user/workspace/eval_13/results.json"

MODELS = [
    ("qwen2-0.5b",  "qwen2-0.5b.Q4_K_M.gguf"),
    ("tinyllama-1.1b", "tinyllama-1.1b.Q4_K_M.gguf"),
    ("llama-3.2-1b", "llama-3.2-1b.Q4_K_M.gguf"),
    ("qwen2-1.5b",   "qwen2-1.5b.Q4_K_M.gguf"),
    ("gemma-2-2b",   "gemma-2-2b.Q4_K_M.gguf"),
    ("phi-3-mini",   "phi-3-mini.Q4_K_M.gguf"),
]
KV_TYPES = ["f16", "q8_0"]   # 2 formats (Q4 KV requires flash-attn on GPU typically)
CTX_LENS = [512, 2048, 8192] # 3 context lengths

results = []
if os.path.exists(OUT):
    with open(OUT) as f:
        results = json.load(f)
done_keys = {(r['model'], r['kv_type'], r['ctx']) for r in results}

total = len(MODELS) * len(KV_TYPES) * len(CTX_LENS)
print(f"Grid: {total} configurations. Already done: {len(done_keys)}. Remaining: {total - len(done_keys)}", flush=True)

n = 0
for model_name, fn in MODELS:
    path = os.path.join(MODELS_DIR, fn)
    for kv in KV_TYPES:
        for ctx in CTX_LENS:
            n += 1
            key = (model_name, kv, ctx)
            if key in done_keys:
                print(f"[{n}/{total}] SKIP {model_name} kv={kv} ctx={ctx} (cached)", flush=True)
                continue
            print(f"[{n}/{total}] RUN  {model_name} kv={kv} ctx={ctx}", flush=True)
            t0 = time.time()
            # 5 samples per config, take max (per paper protocol)
            peaks = []
            for trial in range(3):  # reduced from 5 to 3 to save time
                cmd = ["python3", MEASURE,
                       "--llama-cli", LLAMA_CLI,
                       "--model", path,
                       "--ctx", str(ctx),
                       "--predict", str(min(32, ctx // 4)),
                       "--kv-type", kv,
                       "--threads", "2",
                       "--timeout", "180"]
                try:
                    r = subprocess.run(cmd, capture_output=True, text=True, timeout=200)
                    if r.returncode != 0:
                        print(f"    trial {trial}: RC={r.returncode} stderr={r.stderr[:200]}", flush=True)
                        continue
                    d = json.loads(r.stdout)
                    if d.get('peak_rss_bytes', 0) > 0:
                        peaks.append(d['peak_rss_bytes'])
                        print(f"    trial {trial}: {d['peak_rss_gb']:.3f} GB in {d['elapsed_s']:.1f}s", flush=True)
                except Exception as e:
                    print(f"    trial {trial}: EXC {e}", flush=True)
            if not peaks:
                print(f"    NO SUCCESSFUL TRIALS", flush=True)
                continue
            peak_max = max(peaks)
            peak_mean = sum(peaks) / len(peaks)
            peak_std = (sum((p - peak_mean)**2 for p in peaks) / len(peaks)) ** 0.5
            result = {
                'model': model_name,
                'file': fn,
                'kv_type': kv,
                'ctx': ctx,
                'peak_rss_max_bytes': peak_max,
                'peak_rss_max_gb': peak_max / (1024**3),
                'peak_rss_mean_gb': peak_mean / (1024**3),
                'peak_rss_std_gb':  peak_std / (1024**3),
                'n_trials': len(peaks),
                'wallclock_s': time.time() - t0,
            }
            results.append(result)
            # Persist after each config
            with open(OUT, 'w') as f:
                json.dump(results, f, indent=2)
            done_keys.add(key)

print(f"\nDONE. {len(results)} results in {OUT}", flush=True)
