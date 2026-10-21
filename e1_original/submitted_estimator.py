"""Header-only GGUF memory estimator.

Implements Algorithm 1 from the paper:
  1. HTTP Range GET first 256KB
  2. Parse GGUF v3 KV block for {n_heads, n_kv, d_model, n_layers, n_params}
  3. HEAD requests to sum shard sizes
  4. Return M_weights + M_KV(C) + M_overhead(P)

No dependency on transformers/huggingface; reads GGUF binary directly.
"""

import struct
import requests
from urllib.parse import urlparse
from dataclasses import dataclass
from typing import Optional

# GGUF v3 metadata value types (from GGUF spec / llama.cpp gguf.h)
GGUF_TYPE_UINT8   = 0
GGUF_TYPE_INT8    = 1
GGUF_TYPE_UINT16  = 2
GGUF_TYPE_INT16   = 3
GGUF_TYPE_UINT32  = 4
GGUF_TYPE_INT32   = 5
GGUF_TYPE_FLOAT32 = 6
GGUF_TYPE_BOOL    = 7
GGUF_TYPE_STRING  = 8
GGUF_TYPE_ARRAY   = 9
GGUF_TYPE_UINT64  = 10
GGUF_TYPE_INT64   = 11
GGUF_TYPE_FLOAT64 = 12

_TYPE_STRUCT = {
    GGUF_TYPE_UINT8:   ('<B', 1),
    GGUF_TYPE_INT8:    ('<b', 1),
    GGUF_TYPE_UINT16:  ('<H', 2),
    GGUF_TYPE_INT16:   ('<h', 2),
    GGUF_TYPE_UINT32:  ('<I', 4),
    GGUF_TYPE_INT32:   ('<i', 4),
    GGUF_TYPE_FLOAT32: ('<f', 4),
    GGUF_TYPE_BOOL:    ('<B', 1),
    GGUF_TYPE_UINT64:  ('<Q', 8),
    GGUF_TYPE_INT64:   ('<q', 8),
    GGUF_TYPE_FLOAT64: ('<d', 8),
}

# KV-cache bytes-per-value per format (paper Table II)
KV_BYTES = {
    'f32':  4.0,
    'f16':  2.0,
    'q8_0': 1.0,
    'q5_0': 0.625,
    'q4_0': 0.5,
    'i8':   1.0,  # alias
    'q4':   0.5,  # alias
    'q8':   1.0,  # alias
}


class BufReader:
    def __init__(self, buf: bytes):
        self.buf = buf
        self.pos = 0
    def read(self, n: int) -> bytes:
        if self.pos + n > len(self.buf):
            raise EOFError(f"need {n} at {self.pos}, have {len(self.buf)}")
        b = self.buf[self.pos:self.pos+n]
        self.pos += n
        return b
    def read_scalar(self, gtype: int):
        fmt, sz = _TYPE_STRUCT[gtype]
        return struct.unpack(fmt, self.read(sz))[0]
    def read_string(self) -> str:
        length = struct.unpack('<Q', self.read(8))[0]
        return self.read(length).decode('utf-8', errors='replace')
    def read_value(self, gtype: int):
        if gtype == GGUF_TYPE_STRING:
            return self.read_string()
        if gtype == GGUF_TYPE_ARRAY:
            elem_type = struct.unpack('<I', self.read(4))[0]
            n = struct.unpack('<Q', self.read(8))[0]
            # For architecture/tokenizer arrays we don't need the full contents;
            # skip cheaply where possible
            if elem_type == GGUF_TYPE_STRING:
                out = []
                for _ in range(n):
                    out.append(self.read_string())
                return out
            if elem_type in _TYPE_STRUCT:
                fmt, sz = _TYPE_STRUCT[elem_type]
                data = self.read(sz * n)
                return list(struct.unpack('<' + fmt[1] * n, data))
            raise ValueError(f"unsupported array elem type {elem_type}")
        return self.read_scalar(gtype)


@dataclass
class GGUFHeader:
    arch: str
    n_layers: int
    n_heads: int
    n_kv_heads: int
    d_model: int
    n_params: Optional[int]  # from general.parameter_count if present
    header_bytes_read: int
    file_size: int
    all_metadata: dict


def parse_gguf_header(buf: bytes, file_size: int) -> GGUFHeader:
    """Parse GGUF header from a byte buffer. Reads only KV block, ignores tensor infos."""
    r = BufReader(buf)
    magic = r.read(4)
    if magic != b'GGUF':
        raise ValueError(f"not GGUF, magic={magic!r}")
    version = struct.unpack('<I', r.read(4))[0]
    if version not in (2, 3):
        raise ValueError(f"unsupported GGUF version {version}")
    tensor_count = struct.unpack('<Q', r.read(8))[0]
    kv_count = struct.unpack('<Q', r.read(8))[0]

    metadata = {}
    for _ in range(kv_count):
        key = r.read_string()
        vtype = struct.unpack('<I', r.read(4))[0]
        try:
            val = r.read_value(vtype)
        except Exception as e:
            metadata[key] = f"<parse_error: {e}>"
            break
        metadata[key] = val

    arch = metadata.get('general.architecture', 'unknown')
    # Standard GGUF keys use "{arch}.block_count", "{arch}.attention.head_count", etc.
    def _get(k):
        full = f"{arch}.{k}"
        return metadata.get(full, metadata.get(k))
    n_layers   = _get('block_count')
    n_heads    = _get('attention.head_count')
    n_kv_heads = _get('attention.head_count_kv')
    d_model    = _get('embedding_length')
    n_params   = metadata.get('general.parameter_count') or metadata.get('general.file_type')

    if n_kv_heads is None and n_heads is not None:
        n_kv_heads = n_heads  # standard MHA fallback

    return GGUFHeader(
        arch=arch,
        n_layers=int(n_layers) if n_layers is not None else -1,
        n_heads=int(n_heads) if n_heads is not None else -1,
        n_kv_heads=int(n_kv_heads) if n_kv_heads is not None else -1,
        d_model=int(d_model) if d_model is not None else -1,
        n_params=int(n_params) if isinstance(n_params, int) else None,
        header_bytes_read=r.pos,
        file_size=file_size,
        all_metadata=metadata,
    )


def fetch_gguf_header(url: str, first_bytes: int = 262144, timeout: int = 30) -> GGUFHeader:
    """Fetch first `first_bytes` of remote GGUF file via HTTP Range."""
    headers_req = {'Range': f'bytes=0-{first_bytes - 1}'}
    resp = requests.get(url, headers=headers_req, timeout=timeout, allow_redirects=True, stream=True)
    resp.raise_for_status()
    buf = resp.content
    # Determine full file size via HEAD (or Content-Range parsing)
    content_range = resp.headers.get('Content-Range')
    if content_range and '/' in content_range:
        file_size = int(content_range.split('/')[-1])
    else:
        head = requests.head(url, timeout=timeout, allow_redirects=True)
        file_size = int(head.headers.get('Content-Length', 0))
    hdr = parse_gguf_header(buf, file_size)
    return hdr


def read_local_gguf_header(path: str, first_bytes: int = 262144) -> GGUFHeader:
    """Read first `first_bytes` of a local GGUF file (equivalent to the browser fetch path)."""
    import os
    fsz = os.path.getsize(path)
    with open(path, 'rb') as f:
        buf = f.read(min(first_bytes, fsz))
    return parse_gguf_header(buf, fsz)


# Paper coefficients (fit on calibration set; for CPU/llama.cpp we will refit)
DEFAULT_ALPHA = 0.020  # GB per billion params
DEFAULT_BETA  = 0.15   # GB

def estimate_kv_bytes(n_kv_heads: int, d_model: int, n_heads: int, n_layers: int,
                       context_len: int, kv_format: str = 'f16') -> int:
    """M_KV(C) = 2 * b_KV * n_kv_heads * d_head * n_layers * C   [bytes]"""
    d_head = d_model // n_heads
    b_kv = KV_BYTES[kv_format]
    return int(2 * b_kv * n_kv_heads * d_head * n_layers * context_len)


def estimate_total_bytes(header: GGUFHeader, context_len: int,
                          kv_format: str = 'f16',
                          alpha: float = DEFAULT_ALPHA,
                          beta: float = DEFAULT_BETA,
                          n_params_b: Optional[float] = None) -> dict:
    """Return breakdown: weights, kv, overhead, total (all bytes)."""
    weights = header.file_size
    kv = estimate_kv_bytes(header.n_kv_heads, header.d_model, header.n_heads,
                           header.n_layers, context_len, kv_format)
    if n_params_b is None and header.n_params:
        n_params_b = header.n_params / 1e9
    overhead_gb = (alpha * (n_params_b or 0.0)) + beta
    overhead = int(overhead_gb * (1024**3))
    total = weights + kv + overhead
    return {
        'weights': weights,
        'kv': kv,
        'overhead': overhead,
        'total': total,
        'weights_gb': weights / (1024**3),
        'kv_gb':      kv / (1024**3),
        'overhead_gb': overhead / (1024**3),
        'total_gb':   total / (1024**3),
    }


if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print("usage: python gguf_header_estimator.py <local.gguf|https://url>")
        sys.exit(1)
    src = sys.argv[1]
    hdr = read_local_gguf_header(src) if src.endswith('.gguf') and not src.startswith('http') else fetch_gguf_header(src)
    print(f"arch={hdr.arch}")
    print(f"  n_layers={hdr.n_layers}")
    print(f"  n_heads={hdr.n_heads}")
    print(f"  n_kv_heads={hdr.n_kv_heads}")
    print(f"  d_model={hdr.d_model}")
    print(f"  n_params={hdr.n_params}")
    print(f"  file_size={hdr.file_size/1e9:.3f} GB")
    print(f"  header bytes read={hdr.header_bytes_read}")
    for c in [512, 2048, 8192]:
        e = estimate_total_bytes(hdr, c, 'f16')
        print(f"  ctx={c:5d}  W={e['weights_gb']:.3f}  KV={e['kv_gb']:.3f}  O={e['overhead_gb']:.3f}  total={e['total_gb']:.3f} GB")
