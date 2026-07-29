#!/usr/bin/env python3
"""
ab_prefix.py — controlled A/B of two weight blobs on the same questions.

Scores answer quality the only way that is comparable across models
trained on different data: exact-prefix match against the corpus's own
reference answers, best over all valid answers for each question.

Loss is NOT usable for this — a model trained on a subset is scored
against a different distribution. This is.

Usage:
  ab_prefix.py <labelA> <blobA> <labelB> <blobB> [--shard rustchain]
"""
import json, subprocess, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
HARNESS = "/tmp/h_pe"          # built from the current engine

def truths():
    c = json.loads((ROOT / "train" / "corpus.json").read_text())
    d = {}
    for line in c["QA_PAIRS"]:
        if ":" in line:
            q, a = line.split(":", 1)
            d.setdefault(q.strip(), []).append(a.strip())
    return d

def prefix(a, b):
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n

def gen(blob, q):
    return subprocess.run([HARNESS, blob, q + ": ", "55"],
                          capture_output=True, text=True).stdout.strip()

def main():
    if len(sys.argv) < 5:
        print(__doc__)
        sys.exit(2)
    la, ba, lb, bb = sys.argv[1:5]
    shard = None
    if "--shard" in sys.argv:
        shard = sys.argv[sys.argv.index("--shard") + 1]

    t = truths()
    if shard:
        sys.path.insert(0, str(ROOT / "train"))
        import experts as X
        idx = X.NAMES.index(shard)
        qs = sorted({l.split(":")[0].strip()
                     for l in X.shard()[0][idx] if ":" in l} & set(t))
    else:
        qs = sorted(t)

    print(f"{len(qs)} questions" + (f" (shard: {shard})" if shard else ""))
    print(f"{'question':30s} {la[:8]:>8s} {lb[:8]:>8s}")
    ta = tb = 0
    wins = losses = 0
    for q in qs:
        ga, gb = gen(ba, q), gen(bb, q)
        pa = max(prefix(ga, x) for x in t[q])
        pb = max(prefix(gb, x) for x in t[q])
        ta += pa; tb += pb
        wins += pb > pa
        losses += pb < pa
        mark = "  <<" if pb > pa else ("  >>" if pb < pa else "")
        print(f"{q:30s} {pa:>8d} {pb:>8d}{mark}")
        if pb != pa:
            print(f"     {lb}: {gb[:56]!r}")
    n = len(qs)
    print(f"\nMEAN EXACT PREFIX   {la} {ta/n:.1f} ch   {lb} {tb/n:.1f} ch")
    if ta:
        print(f"ratio {tb/ta:.2f}x   ({wins} better, {losses} worse, "
              f"{n-wins-losses} tied)")

if __name__ == "__main__":
    main()
