#!/usr/bin/env python3
"""Compare the C integer engine against the QAT model's greedy outputs."""
import json, subprocess, sys
from pathlib import Path

here = Path(__file__).resolve().parent
blob = here.parent / "res" / "elya_genesis.bin"
vectors = json.loads((here.parent / "train" / "test_vectors.json").read_text())

total_match = total_tok = 0
for v in vectors:
    exp = v["expect"]
    out = subprocess.run([str(here / "harness"), str(blob), v["prompt"],
                          str(max(len(exp), 8))],
                         capture_output=True, text=True).stdout.rstrip("\n")
    n = max(len(exp), 1)
    # common prefix length: the honest metric for greedy autoregressive
    # divergence (one differing token shifts everything after it)
    p = 0
    for a, b in zip(out, exp):
        if a != b:
            break
        p += 1
    total_match += p
    total_tok += n
    flag = "OK " if p / n > 0.5 else "DIFF"
    print(f"[{flag}] {v['prompt']!r}  prefix {p}/{n}")
    print(f"   pytorch: {exp!r}")
    print(f"   c-int  : {out!r}")
print(f"\nMean common-prefix agreement: {total_match}/{total_tok} "
      f"({100*total_match/max(total_tok,1):.1f}%)")
sys.exit(0 if total_match / max(total_tok, 1) > 0.5 else 1)
