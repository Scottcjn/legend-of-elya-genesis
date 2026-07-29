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
    m = sum(1 for a, b in zip(out, exp) if a == b)
    total_match += m
    total_tok += n
    flag = "OK " if m / n > 0.85 else "DIFF"
    print(f"[{flag}] {v['prompt']!r}")
    print(f"   pytorch: {exp!r}")
    print(f"   c-int  : {out!r}")
print(f"\nToken agreement: {total_match}/{total_tok} "
      f"({100*total_match/max(total_tok,1):.1f}%)")
sys.exit(0 if total_match / max(total_tok, 1) > 0.7 else 1)
