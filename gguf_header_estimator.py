"""Header-only GGUF memory estimator (camera-ready version, IEEE AIoT 2026 Paper 161).

Reads only the leading architecture fields of a GGUF file (local path or HTTP URL),
then estimates llama.cpp peak resident memory on a CPU host with --no-mmap:

    M = M_w + M_KV(C) + M_rt + M_att(C)

  M_w    file size (sum of shards)
  M_KV   b(q) * n_kv * (d_k + d_v) * sum_l cells_l(C)   (Gemma-2 windowed layers hold min(C, W+u) cells)
  M_rt   alpha * P + beta                               (calibrated per backend)
  M_att  gamma * 4 * u * n_heads * C                    (filled-context regime only; calibrated)

Range protocol: request 4 KB, parse keys in order, double the range if a required key
has not been reached, skip array bodies by their declared length, and stop as soon as
every required field (plus the vocabulary length prefix) has been read.
Standard library only.
"""
import struct, os, sys, json, urllib.request

T = {0: ('B', 1), 1: ('b', 1), 2: ('H', 2), 3: ('h', 2), 4: ('I', 4), 5: ('i', 4), 6: ('f', 4),
     7: ('?', 1), 10: ('Q', 8), 11: ('q', 8), 12: ('d', 8)}
STRING, ARRAY = 8, 9

# Storage bytes per cache element including block scales (ggml block sizes).
KV_BYTES = {'f32': 4.0, 'f16': 2.0, 'bf16': 2.0, 'q8_0': 34 / 32, 'q5_1': 24 / 32,
            'q5_0': 22 / 32, 'q4_1': 20 / 32, 'q4_0': 18 / 32}

# Coefficients calibrated on the paper's CPU host (2 vCPU Xeon, llama.cpp, --no-mmap, FA off).
RESERVED = dict(alpha=0.0044, beta=0.0778)            # Experiment 1, all 36 configurations
FILLED = dict(gamma=None, beta=None)                  # Experiment 2, filled by calibrate step
PHYS_BATCH = 512

REQUIRED = ['block_count', 'attention.head_count', 'attention.head_count_kv', 'embedding_length']
OPTIONAL = ['attention.key_length', 'attention.value_length', 'attention.sliding_window', 'context_length']


class NeedMore(Exception):
    pass


class Reader:
    def __init__(self, buf):
        self.b, self.p = buf, 0

    def take(self, n):
        if self.p + n > len(self.b):
            raise NeedMore()
        v = self.b[self.p:self.p + n]; self.p += n; return v

    def skip(self, n):
        if self.p + n > len(self.b):
            raise NeedMore()
        self.p += n

    def u32(self): return struct.unpack('<I', self.take(4))[0]
    def u64(self): return struct.unpack('<Q', self.take(8))[0]
    def s(self): return self.take(self.u64()).decode('utf-8', 'replace')


def parse(buf):
    """Parse as far as needed. Raises NeedMore if buf ends before the required fields."""
    r = Reader(buf)
    if r.take(4) != b'GGUF':
        raise ValueError('not a GGUF file')
    version = r.u32(); r.u64(); n_kv = r.u64()
    kv, arch, n_vocab = {}, None, None
    for _ in range(n_kv):
        key = r.s(); t = r.u32()
        if t in T:
            c, sz = T[t]; val = struct.unpack('<' + c, r.take(sz))[0]
        elif t == STRING:
            val = r.s()
        elif t == ARRAY:
            at = r.u32(); n = r.u64(); val = None
            if key == 'tokenizer.ggml.tokens':
                n_vocab = n
                if arch and all(f'{arch}.{k}' in kv for k in REQUIRED):
                    break                                  # count prefix is enough; body not needed
            if at in T:
                r.skip(T[at][1] * n)
            else:
                for _ in range(n):
                    if at == STRING:
                        r.skip(r.u64())
                    else:
                        raise ValueError('nested array')
        else:
            raise ValueError(f'unknown type {t}')
        kv[key] = val
        if key == 'general.architecture':
            arch = val
        if arch and all(f'{arch}.{k}' in kv for k in REQUIRED) and n_vocab is not None:
            break
    if not arch or not all(f'{arch}.{k}' in kv for k in REQUIRED):
        raise ValueError('required fields missing from metadata')
    f = {k: kv.get(f'{arch}.{k}') for k in REQUIRED + OPTIONAL}
    return dict(version=version, arch=arch, n_vocab=n_vocab, bytes_parsed=r.p, **f)


def _range(url, end):
    req = urllib.request.Request(url, headers={'Range': f'bytes=0-{end - 1}', 'User-Agent': 'gguf-header-estimator'})
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = resp.read(end)
        cr = resp.headers.get('Content-Range', '')
    size = int(cr.split('/')[-1]) if '/' in cr else None
    return data, size


def read_header(src, first=4096, cap=64 << 20):
    """Return (fields, file_size, bytes_transferred, requests)."""
    n, sent, reqs = first, 0, 0
    while True:
        if src.startswith('http'):
            buf, size = _range(src, n); reqs += 1
        else:
            with open(src, 'rb') as fh:
                buf = fh.read(n)
            size = os.path.getsize(src); reqs += 1
        sent += len(buf)
        try:
            return parse(buf), size, sent, reqs
        except NeedMore:
            if n >= cap:
                raise
            n *= 2


def head_dims(h):
    dk = h['attention.key_length'] or h['embedding_length'] // h['attention.head_count']
    dv = h['attention.value_length'] or h['embedding_length'] // h['attention.head_count']
    return dk, dv


def kv_bytes(h, C, cache='f16', u=PHYS_BATCH):
    dk, dv = head_dims(h); L = h['block_count']
    per_cell = KV_BYTES[cache] * h['attention.head_count_kv'] * (dk + dv)
    W = h['attention.sliding_window']
    if h['arch'] == 'gemma2' and W:                      # every second layer is windowed in llama.cpp
        return per_cell * (L // 2 * C + (L - L // 2) * min(C, W + u))
    return per_cell * L * C


def estimate(h, file_size, C, cache='f16', filled=False, P=None, u=PHYS_BATCH, coef=None):
    G = 2 ** 30
    P = P if P is not None else file_size * 8 / 4.65 / 1e9
    w, kv = file_size / G, kv_bytes(h, C, cache, u) / G
    if filled:
        c = coef or FILLED
        att = c['gamma'] * 4 * u * h['attention.head_count'] * C / G
        rt = c['beta']
    else:
        c = coef or RESERVED
        att, rt = 0.0, c['alpha'] * P + c['beta']
    return dict(weights=w, kv=kv, runtime=rt, attention=att, total=w + kv + rt + att)


_cal = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'coefficients_filled.json')
if os.path.exists(_cal):
    FILLED.update(json.load(open(_cal)))

if __name__ == '__main__':
    src = sys.argv[1]; C = int(sys.argv[2]) if len(sys.argv) > 2 else 4096
    cache = sys.argv[3] if len(sys.argv) > 3 else 'f16'
    h, size, sent, reqs = read_header(src)
    print(json.dumps(dict(header=h, file_size=size, bytes_transferred=sent, requests=reqs,
                          reserved=estimate(h, size, C, cache),
                          filled=estimate(h, size, C, cache, filled=True) if FILLED['gamma'] else None), indent=1))
