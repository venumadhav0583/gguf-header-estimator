"""Byte-level GGUF header audit: where do the fields the estimator needs end,
and how large is the full metadata + tensor-info block?"""
import struct, json, os, sys
T = {0:('B',1),1:('b',1),2:('H',2),3:('h',2),4:('I',4),5:('i',4),6:('f',4),7:('?',1),10:('Q',8),11:('q',8),12:('d',8)}
def rd_str(f):
    n = struct.unpack('<Q', f.read(8))[0]; return f.read(n).decode('utf-8','replace')
def rd_val(f, t):
    if t in T:
        c,s = T[t]; return struct.unpack('<'+c, f.read(s))[0]
    if t == 8: return rd_str(f)
    if t == 9:
        at = struct.unpack('<I', f.read(4))[0]; n = struct.unpack('<Q', f.read(8))[0]
        if at in T:
            f.seek(T[at][1]*n, 1); return f'<array {n}>'
        for _ in range(n): rd_val(f, at)
        return f'<array {n}>'
    raise ValueError(t)
def audit(path):
    out = {}
    with open(path,'rb') as f:
        assert f.read(4) == b'GGUF'
        ver = struct.unpack('<I', f.read(4))[0]
        nt, nkv = struct.unpack('<QQ', f.read(16))
        kv = {}; end_off = {}
        for _ in range(nkv):
            k = rd_str(f); t = struct.unpack('<I', f.read(4))[0]; v = rd_val(f, t)
            kv[k] = v; end_off[k] = f.tell()
        meta_end = f.tell()
        for _ in range(nt):
            rd_str(f); nd = struct.unpack('<I', f.read(4))[0]; f.seek(8*nd + 4 + 8, 1)
        ti_end = f.tell()
    arch = kv['general.architecture']
    need = [f'{arch}.{s}' for s in ['block_count','attention.head_count','attention.head_count_kv','embedding_length','attention.key_length','attention.value_length','attention.sliding_window','context_length']]
    present = [k for k in need if k in kv]
    out = dict(file=os.path.basename(path), version=ver, n_tensors=nt, n_kv=nkv, arch=arch,
               needed_fields_end_offset=max(end_off[k] for k in present if k in end_off),
               metadata_kv_end=meta_end, tensor_info_end=ti_end, file_size=os.path.getsize(path),
               fields={k.split('.',1)[1]: kv[k] for k in present},
               param_count=kv.get('general.parameter_count'),
               tokenizer_vocab=kv.get('tokenizer.ggml.tokens'),
               file_type=kv.get('general.file_type'))
    # order: is any needed field after tokenizer arrays?
    keys = list(kv.keys()); tok_first = min([i for i,k in enumerate(keys) if k.startswith('tokenizer.')] or [10**9])
    out['needed_fields_before_tokenizer'] = all(keys.index(k) < tok_first for k in present)
    return out
if __name__ == '__main__':
    d = sys.argv[1]; res = [audit(os.path.join(d,f)) for f in sorted(os.listdir(d)) if f.endswith('.gguf')]
    json.dump(res, open(sys.argv[2],'w'), indent=1)
    for r in res: print(r['file'], r['needed_fields_end_offset'], r['metadata_kv_end'], r['tensor_info_end'], r['needed_fields_before_tokenizer'], r['fields'], r['param_count'], r['tokenizer_vocab'])
