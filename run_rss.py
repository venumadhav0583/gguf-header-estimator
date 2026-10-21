"""Run llama-completion once, sample peak RSS of process tree from /proc every 100 ms, parse llama.cpp buffer log lines."""
import subprocess, time, os, re, json, sys
def tree_rss(pid):
    tot=0; pids=[pid]
    try:
        for c in open(f'/proc/{pid}/task/{pid}/children').read().split(): pids.append(int(c))
    except Exception: pass
    for p in pids:
        try:
            for line in open(f'/proc/{p}/status'):
                if line.startswith('VmRSS:'): tot+=int(line.split()[1])*1024
        except Exception: pass
    return tot
def run(args, log_path):
    t0=time.time(); lf=open(log_path,'w')
    p=subprocess.Popen(args,stdout=lf,stderr=subprocess.STDOUT); peak=0
    while p.poll() is None:
        peak=max(peak,tree_rss(p.pid)); time.sleep(0.1)
    lf.close(); log=open(log_path).read()
    def grab(pat):
        return [float(x) for x in re.findall(pat,log)]
    return dict(peak_rss_bytes=peak, wall_s=time.time()-t0, rc=p.returncode,
        model_buf_MiB=grab(r'CPU(?:_Mapped|_REPACK)? model buffer size =\s*([\d.]+) MiB'),
        kv_MiB=grab(r'llama_kv_cache.*?size =\s*([\d.]+) MiB'),
        compute_buf_MiB=grab(r'CPU compute buffer size =\s*([\d.]+) MiB'),
        n_prompt_tokens=grab(r'prompt eval time =.*?/\s*(\d+) tokens'),
        prompt_tps=grab(r'prompt eval time =.*?([\d.]+) tokens per second'))
if __name__=='__main__':
    print(json.dumps(run(sys.argv[2:], sys.argv[1])))
