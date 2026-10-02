# fetch_corpus.py: Downloads a corpus config from maxpicy/p2cot-tts (corpus/) and writes the JSONL train.py reads.

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from jsonl_to_parquet import to_row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, choices=["stage1p", "stage2_v2"])
    ap.add_argument("--out_dir", default="data")
    ap.add_argument("--repo", default="maxpicy/p2cot-tts")
    ap.add_argument("--cache_dir", default=None)
    args = ap.parse_args()
    import pyarrow.parquet as pq
    from huggingface_hub import snapshot_download

    root = os.path.join(snapshot_download(args.repo, cache_dir=args.cache_dir,
                                          allow_patterns=[f"corpus/{args.config}/*", "corpus/aux/*"]), "corpus")
    os.makedirs(args.out_dir, exist_ok=True)
    for split in ("train", "validation"):
        man = json.load(open(os.path.join(root, args.config, f"{split}_manifest.json")))
        out = os.path.join(args.out_dir, f"{args.config}_{split}.jsonl")
        n = 0
        with open(out, "w") as f:
            for shard in man["shards"]:
                for batch in pq.ParquetFile(os.path.join(root, args.config, shard["file"])).iter_batches(batch_size=20000):
                    for c in batch.to_pylist():
                        f.write(json.dumps(to_row(c)) + "\n")
                        n += 1
        if n != man["rows"]:
            sys.exit(f"{out}: wrote {n} rows, the manifest says {man['rows']}")
        print(f"{out}: {n} rows")
    for f in os.listdir(os.path.join(root, "aux")):
        dst = os.path.join(args.out_dir, f)
        if not os.path.exists(dst):
            with open(os.path.join(root, "aux", f), "rb") as a, open(dst, "wb") as b:
                b.write(a.read())

if __name__ == "__main__":
    main()
