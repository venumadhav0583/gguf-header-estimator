"""Measure peak RSS of a llama.cpp inference run.

Spawns llama-cli, samples /proc/<pid>/status VmRSS at 100ms intervals, records max.
Feeds a short warm-up prompt to fully populate the KV cache to the requested context.
"""
import subprocess, time, sys, os, argparse, json, signal


def sample_rss(pid: int) -> int:
    """Return RSS in bytes for a live pid, or 0 if gone."""
    try:
        with open(f'/proc/{pid}/status', 'r') as f:
            for line in f:
                if line.startswith('VmRSS:'):
                    kb = int(line.split()[1])
                    return kb * 1024
    except (FileNotFoundError, ProcessLookupError):
        pass
    return 0


def sample_rss_recursive(pid: int) -> int:
    """RSS of pid + all descendants."""
    total = sample_rss(pid)
    try:
        # Find children via /proc/*/status PPid field
        for d in os.listdir('/proc'):
            if not d.isdigit(): continue
            try:
                with open(f'/proc/{d}/status', 'r') as f:
                    for line in f:
                        if line.startswith('PPid:'):
                            ppid = int(line.split()[1])
                            if ppid == pid:
                                total += sample_rss(int(d))
                            break
            except (FileNotFoundError, ProcessLookupError):
                continue
    except Exception:
        pass
    return total


def run(args):
    llama = args.llama_cli
    cmd = [
        llama,
        '-m', args.model,
        '-c', str(args.ctx),
        '-n', str(args.predict),
        '-ctk', args.kv_type,
        '-ctv', args.kv_type,
        '-t', str(args.threads),
        '-b', str(min(args.ctx, 512)),
        '-ub', str(min(args.ctx, 512)),
        '--single-turn',
        '--no-mmap',   # forces RSS to reflect actual allocations, not mmap
        '-p', args.prompt,
    ]
    if args.flash_attn:
        cmd += ['-fa', 'on']

    t0 = time.time()
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    peak = 0
    samples = []
    while proc.poll() is None:
        rss = sample_rss_recursive(proc.pid)
        if rss > 0:
            samples.append(rss)
            if rss > peak:
                peak = rss
        time.sleep(0.1)
        if time.time() - t0 > args.timeout:
            proc.send_signal(signal.SIGTERM)
            try: proc.wait(timeout=5)
            except: proc.kill()
            break

    stderr = proc.stderr.read() if proc.stderr else ''
    ret = proc.returncode
    elapsed = time.time() - t0
    return {
        'peak_rss_bytes': peak,
        'peak_rss_gb': peak / (1024**3),
        'num_samples': len(samples),
        'elapsed_s': elapsed,
        'returncode': ret,
        'stderr_tail': stderr[-800:] if stderr else '',
    }


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--llama-cli', required=True)
    ap.add_argument('--model', required=True)
    ap.add_argument('--ctx', type=int, default=2048)
    ap.add_argument('--predict', type=int, default=16)
    ap.add_argument('--kv-type', default='f16', choices=['f32','f16','q8_0','q5_0','q4_0'])
    ap.add_argument('--threads', type=int, default=2)
    ap.add_argument('--flash-attn', action='store_true')
    ap.add_argument('--timeout', type=int, default=300)
    ap.add_argument('--prompt', default='The quick brown fox jumps over the lazy dog. ' * 20)
    args = ap.parse_args()
    result = run(args)
    print(json.dumps(result, indent=2))
