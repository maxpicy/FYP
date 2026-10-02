# jsonl_to_parquet.py: Lossless corpus JSONL <-> parquet (to_row() rebuilds the exact training row).

import argparse
import hashlib
import json
import os
import sys
from multiprocessing import Pool

STR_COLS = ["prompt", "reasoning", "emotion_label", "pace", "pitch", "cot_form", "source", "speaker", "id", "file",
            "caption_source"]
CHUNK = 20000


def _schema():
    import pyarrow as pa
    return pa.schema([(c, pa.string()) for c in STR_COLS] + [
        ("speaker_id", pa.int32()),
        ("speech_tokens", pa.list_(pa.int16())),
        ("residual_codes", pa.list_(pa.list_(pa.int16()))),
        ("plan", pa.string()),
        ("meta", pa.string()),
    ])


def _is_int(x):
    return isinstance(x, int) and not isinstance(x, bool)


def from_row(r):
    out = {c: None for c in STR_COLS + ["speaker_id", "speech_tokens", "residual_codes", "plan", "meta"]}
    meta = {}
    for k, v in r.items():
        if k in STR_COLS and isinstance(v, str):
            out[k] = v
        elif k == "speaker_id" and _is_int(v) and -2 ** 31 <= v < 2 ** 31:
            out[k] = v
        elif k == "speech_tokens" and isinstance(v, list) and all(_is_int(x) and -32768 <= x < 32768 for x in v):
            out[k] = v
        elif (k == "residual_codes" and isinstance(v, list) and all(isinstance(lv, list) for lv in v)
              and all(_is_int(x) and -32768 <= x < 32768 for lv in v for x in lv)):
            out[k] = v
        elif k == "plan" and isinstance(v, dict):
            out[k] = json.dumps(v, sort_keys=True, separators=(",", ":"))
        else:
            meta[k] = v
    out["meta"] = json.dumps(meta, sort_keys=True, separators=(",", ":")) if meta else None
    return out


def to_row(c):
    r = json.loads(c["meta"]) if c.get("meta") else {}
    for k in STR_COLS + ["speaker_id", "speech_tokens", "residual_codes"]:
        if c.get(k) is not None:
            r[k] = c[k]
    if c.get("plan") is not None:
        r["plan"] = json.loads(c["plan"])
    return r


def row_hash(r):
    return hashlib.sha1(json.dumps(r, sort_keys=True, separators=(",", ":")).encode()).digest()


def _shard(task):
    import pyarrow as pa
    import pyarrow.parquet as pq
    src, offset, n_rows, path = task
    schema = _schema()
    hashes, buf = [], []
    tmp = path + ".tmp"
    w = pq.ParquetWriter(tmp, schema, compression="zstd")

    def flush():
        cols = {name: [] for name in schema.names}
        for r in buf:
            for k, v in from_row(r).items():
                cols[k].append(v)
        w.write_table(pa.table(cols, schema=schema))
        buf.clear()

    with open(src, "rb") as f:
        f.seek(offset)
        for _ in range(n_rows):
            line = f.readline()
            if not line:
                raise SystemExit(f"{src}: unexpected EOF in shard {path}")
            r = json.loads(line)
            hashes.append(row_hash(r))
            buf.append(r)
            if len(buf) >= CHUNK:
                flush()
        if buf:
            flush()
    w.close()
    i = 0
    for b in pq.ParquetFile(tmp).iter_batches(batch_size=CHUNK):
        for c in b.to_pylist():
            if i >= len(hashes) or row_hash(to_row(c)) != hashes[i]:
                raise SystemExit(f"{path}: row {i} does not round-trip")
            i += 1
    if i != len(hashes):
        raise SystemExit(f"{path}: {i} rows read back, {len(hashes)} written")
    os.replace(tmp, path)
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 24), b""):
            h.update(blk)
    return {"file": os.path.basename(path), "rows": n_rows, "bytes": os.path.getsize(path), "sha256": h.hexdigest()}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", default="train")
    ap.add_argument("--shard_rows", type=int, default=100000)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    offsets, n = [], 0
    with open(args.src, "rb") as f:
        pos = 0
        for line in f:
            if n % args.shard_rows == 0:
                offsets.append(pos)
            pos += len(line)
            n += 1
    k = len(offsets)
    tasks = [(args.src, off, min(args.shard_rows, n - i * args.shard_rows),
              os.path.join(args.out, f"{args.split}-{i:05d}-of-{k:05d}.parquet")) for i, off in enumerate(offsets)]
    print(f"[parquet] {args.src}: {n} rows -> {k} shards of <= {args.shard_rows}", flush=True)
    with Pool(args.workers) as pool:
        res = []
        for r in pool.imap(_shard, tasks):
            print(f"[parquet] ok {r['file']}: {r['rows']} rows, {r['bytes'] / 1e9:.2f} GB, round-trip verified",
                  flush=True)
            res.append(r)
    assert sum(r["rows"] for r in res) == n
    man = {"source": os.path.basename(args.src), "rows": n, "shards": res,
           "source_bytes": os.path.getsize(args.src), "parquet_bytes": sum(r["bytes"] for r in res),
           "lossless_round_trip_verified": True}
    with open(os.path.join(args.out, f"{args.split}_manifest.json"), "w") as f:
        json.dump(man, f, indent=1)
    print(f"[parquet] DONE {args.src}: {n} rows, {man['source_bytes'] / 1e9:.2f} GB JSONL -> "
          f"{man['parquet_bytes'] / 1e9:.2f} GB parquet", flush=True)

if __name__ == "__main__":
    sys.exit(main())
