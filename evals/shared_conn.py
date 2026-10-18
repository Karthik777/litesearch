"""One shared connection against a connection per thread: read throughput, read latency under a writer, open cost, memory.

`database(shared=True)` runs every statement on one `Conn`, one statement at a time. A connection per thread on a WAL file runs reads in parallel, and each connection opens, caches pages and loads the HNSW index on its own. Both run against one seeded store: 60 Zipf-distributed pseudo-words and one random 256-dim float16 unit vector per row, with an HNSW index. Contention does not depend on what the text means, so no encoder runs.

SQLite keeps allocation statistics by default, behind one process-wide mutex that every allocation takes. Connections on different threads queue on it. `--memstatus 0` turns the statistics off before the first connection opens.

    python -m evals.shared_conn --rows 20000 --threads 1,2,4,8 --repeats 3
    python -m evals.shared_conn --rows 20000 --threads 1,2,4,8 --repeats 3 --memstatus 0
"""
from __future__ import annotations
import argparse, json, os, platform, shutil, statistics as st, subprocess, sys, tempfile, threading, time
from pathlib import Path

import apsw, numpy as np

from litesearch.core import database, process_content

NDIM, BUSY, STORE = 256, 30_000, 'store'
OPS = dict(
    fts=lambda s, q, v: s.fts_search(q, columns=['rowid'], order_by='rank', limit=10),
    vec=lambda s, q, v: s.vec_search(v, columns=['rowid'], limit=10),
    ann=lambda s, q, v: s.ann_search(v, columns=['rowid'], limit=10))

def _words(n=5000, seed=0):
    rng, abc = np.random.default_rng(seed), np.array(list('abcdefghijklmnopqrstuvwxyz'))
    return list(dict.fromkeys(''.join(rng.choice(abc, rng.integers(4, 10))) for _ in range(n)))

def _vecs(rng, n):
    v = rng.standard_normal((n, NDIM)).astype(np.float32)
    return [r.astype(np.float16).tobytes() for r in v / np.linalg.norm(v, axis=1, keepdims=True)]

def _rows(rng, words, n):
    p = 1 / np.arange(1, len(words) + 1); p /= p.sum()
    w, ix = np.array(words), rng.choice(len(words), size=(n, 60), p=p)
    return [dict(content=' '.join(w[r]), embedding=e) for r, e in zip(ix, _vecs(rng, n))]

def _queries(rng, words, n): return [' OR '.join(rng.choice(words[100:1000], 2)) for _ in range(n)]

def _open(path, shared=False):
    db = database(path, busy_timeout=BUSY, shared=shared)
    return db, db.get_store(STORE, hash=True, ann=True, ndim=NDIM)

def build(path, n, seed=0):
    'Write the store and its HNSW sidecar to `path`; return the vocabulary.'
    words = _words(seed=seed)
    db, s = _open(path)
    process_content(s, _rows(np.random.default_rng(seed), words, n), embed=False)
    s.rebuild_index(); db.close()
    return words

def _copy(path, dst):
    'A fresh copy of the store, so a run that writes leaves the next run its starting file.'
    for f in Path(path).parent.glob(Path(path).name + '*'): shutil.copy(f, Path(dst) / f.name)
    return str(Path(dst) / Path(path).name)

def _run(nt, setup, body):
    'Run `body(i, state)` on `nt` threads after each ran `setup(i)`; return wall seconds from a common start.'
    go, errs, t0 = threading.Barrier(nt + 1), [], [0.0]
    def work(i):
        try: state = setup(i)
        except Exception as e: errs.append(repr(e)); go.abort(); return
        go.wait()
        try: body(i, state)
        except Exception as e: errs.append(repr(e))
    ts = [threading.Thread(target=work, args=(i,)) for i in range(nt)]
    for t in ts: t.start()
    try: go.wait(); t0[0] = time.perf_counter()
    except threading.BrokenBarrierError: pass
    for t in ts: t.join()
    if errs: raise RuntimeError(errs[0])
    return time.perf_counter() - t0[0]

def throughput(path, words, op, nt, n, shared, seed=1):
    'Queries per second when `nt` threads each run `n` queries of kind `op`.'
    rng = np.random.default_rng(seed)
    qs, vs = _queries(rng, words, nt * n), _vecs(rng, nt * n)
    one = _open(path, shared=True) if shared else None
    if one: OPS[op](one[1], qs[0], vs[0])
    dbs = []
    def setup(i):
        db, s = one or _open(path)
        dbs.append(db)
        OPS[op](s, qs[0], vs[0])
        return s
    def body(i, s):
        for k in range(i * n, (i + 1) * n): OPS[op](s, qs[k], vs[k])
    dt = _run(nt, setup, body)
    for db in {id(d): d for d in dbs}.values(): db.close()
    return nt * n / dt

def latency(path, words, nr, secs, shared, writer, batch=200, seed=2):
    'Reader FTS latency over `secs` seconds with `nr` readers, and one writer committing `batch`-row transactions when `writer`.'
    rng = np.random.default_rng(seed)
    qs, batches = _queries(rng, words, 20_000), [_rows(rng, words, batch) for _ in range(400)] if writer else []
    one = _open(path, shared=True) if shared else None
    stop, lat, wrote, dbs = threading.Event(), [[] for _ in range(nr)], [0], []
    def setup(i):
        db, s = one or _open(path)
        dbs.append(db)
        OPS['fts'](s, qs[0], None)
        return s
    def body(i, s):
        if i == nr:
            for b in batches:
                if stop.is_set(): return
                process_content(s, b, embed=False); wrote[0] += len(b)
            return
        k = i
        while not stop.is_set():
            t = time.perf_counter(); OPS['fts'](s, qs[k % len(qs)], None)
            lat[i].append(time.perf_counter() - t); k += nr
    timer = threading.Timer(secs, stop.set); timer.start()
    _run(nr + bool(writer), setup, body)
    for db in {id(d): d for d in dbs}.values(): db.close()
    ms = sorted(x * 1000 for l in lat for x in l)
    return dict(qps=len(ms) / secs, p50=ms[len(ms) // 2], p99=ms[int(len(ms) * .99)], max=ms[-1], rows_s=wrote[0] / secs)

def memory(path, k=8):
    'Open `k` connections in this process; return median open and index-load ms, and resident MB each one adds.'
    import psutil
    proc, rng, words = psutil.Process(), np.random.default_rng(3), _words()
    q, v, dbs, opn, idx, rss = _queries(rng, words, 1)[0], _vecs(rng, 1)[0], [], [], [], []
    for _ in range(k):
        t0 = time.perf_counter(); db, s = _open(path); t1 = time.perf_counter(); db.get_index(STORE); t2 = time.perf_counter()
        OPS['fts'](s, q, v); OPS['vec'](s, q, v); OPS['ann'](s, q, v)
        dbs.append(db); opn.append(t1 - t0); idx.append(t2 - t1); rss.append(proc.memory_info().rss)
    return dict(open_ms=st.median(opn) * 1000, index_ms=st.median(idx) * 1000, mb_per_conn=(rss[-1] - rss[0]) / (k - 1) / 2**20,
                sidecar_mb=os.path.getsize(f'{path}.{STORE}.usearch') / 2**20)

def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--rows', type=int, default=20_000)
    p.add_argument('--threads', default='1,2,4,8')
    p.add_argument('--repeats', type=int, default=3)
    p.add_argument('--n', default='fts=400,vec=10,ann=400', help='queries per thread per kind')
    p.add_argument('--secs', type=float, default=5)
    p.add_argument('--readers', type=int, default=4)
    p.add_argument('--memstatus', type=int, default=1, help='0 turns off SQLite allocation statistics')
    p.add_argument('--memory', help=argparse.SUPPRESS)
    a = p.parse_args()
    if not a.memstatus: apsw.shutdown(); apsw.config(apsw.SQLITE_CONFIG_MEMSTATUS, 0); apsw.initialize()
    if a.memory: return print(json.dumps(memory(a.memory)))
    tmp = tempfile.mkdtemp(); base = str(Path(tmp) / 'base.db')
    t = time.perf_counter(); words = build(base, a.rows)
    print(f'{platform.platform()}, {os.cpu_count()} cores, Python {platform.python_version()}, SQLite {apsw.sqlite_lib_version()}, memstatus {a.memstatus}; {a.rows} rows built in {time.perf_counter() - t:.1f}s\n')
    ns, nts = dict(kv.split('=') for kv in a.n.split(',')), [int(x) for x in a.threads.split(',')]
    print('| reads | threads | shared q/s | per-thread q/s | per-thread / shared |\n|---|---|---|---|---|')
    for op in OPS:
        for nt in nts:
            r = {m: st.median(throughput(base, words, op, nt, int(ns[op]), m == 'shared', seed=s) for s in range(a.repeats)) for m in ('shared', 'own')}
            print(f"| {op} | {nt} | {r['shared']:.0f} | {r['own']:.0f} | {r['own'] / r['shared']:.2f}x |", flush=True)
    print(f'\n| {a.readers} FTS readers | writer | q/s | p50 ms | p99 ms | max ms | rows written/s |\n|---|---|---|---|---|---|---|')
    for m in ('shared', 'own'):
        for w in (False, True):
            runs = [latency(_copy(base, tempfile.mkdtemp(dir=tmp)), words, a.readers, a.secs, m == 'shared', w, seed=s) for s in range(a.repeats)]
            r = {k: st.median(x[k] for x in runs) for k in runs[0]}
            print(f"| {m} | {'yes' if w else 'no'} | {r['qps']:.0f} | {r['p50']:.2f} | {r['p99']:.2f} | {r['max']:.1f} | {r['rows_s']:.0f} |", flush=True)
    out = subprocess.run([sys.executable, '-m', 'evals.shared_conn', '--memory', base, '--memstatus', str(a.memstatus)], capture_output=True, text=True, check=True)
    m = json.loads(out.stdout)
    print(f"\nper connection: open {m['open_ms']:.1f} ms, HNSW load {m['index_ms']:.1f} ms, {m['mb_per_conn']:.1f} MB resident (sidecar {m['sidecar_mb']:.1f} MB)")
    shutil.rmtree(tmp, ignore_errors=True)

if __name__ == '__main__': main()
