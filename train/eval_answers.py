#!/usr/bin/env python3
"""
eval_answers.py — the honest MoE-vs-single comparison.

Loss is NOT comparable between the two: each expert trains on a different
corpus subset, so its cross-entropy is measured against a different
distribution. A lower per-expert loss could just mean "easier shard".

This measures the thing we actually care about instead: **given a question
from the game, does she answer it correctly?** Same questions, same
decoding, same metric, both models.

Metric. Each distinct question has several valid answers in the corpus
(e.g. "Who are you?" has 5). Scoring against one arbitrarily chosen answer
would be unfair, so for each generation we take the BEST match over all
valid answers for that question:

  exact      generated answer equals a valid answer exactly
  prefix     longest common prefix / answer length, best over valid answers
  charset    fraction of generated chars that are plausible English
             (catches the "aaaa aaaa" degenerate failure mode that can
             score deceptively well on prefix)

Usage:
  eval_answers.py single train/champions/genesis_best.pt
  eval_answers.py moe    train/elya_moe.pt
"""
import ast, json, sys, math
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from experts import load_corpus, classify, NAMES, N_EXPERTS   # noqa: E402

CTX = 64
VOCAB = 256
device = "cuda" if torch.cuda.is_available() else "cpu"

def valid_answers():
    """question -> set of valid answers, from the corpus itself."""
    qa, _bg = load_corpus()
    d = defaultdict(set)
    for line in qa:
        if ":" not in line:
            continue
        q, a = line.split(":", 1)
        d[q.strip() + ":"].add(a.strip())
    return d

def common_prefix(a, b):
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n

def score(gen, answers):
    gen = gen.strip()
    best_pfx, best_ans = 0.0, ""
    for a in answers:
        p = common_prefix(gen, a) / max(len(a), 1)
        if p > best_pfx:
            best_pfx, best_ans = p, a
    exact = 1.0 if gen in answers else 0.0
    return exact, best_pfx, best_ans

def degenerate(gen):
    """Detect the 'aaa aaa aaa' / 'the the the' failure mode."""
    words = gen.split()
    if len(words) < 3:
        return len(set(gen.replace(" ", ""))) <= 2
    return len(set(words)) <= max(1, len(words) // 3)

def run(kind, ckpt):
    va = valid_answers()
    questions = sorted(va.keys())

    if kind == "single":
        import train_elya_genesis as M            # noqa
        model = M.ElyaGPT().to(device) if hasattr(M, "ElyaGPT") else None
        if model is None:
            print("single-model module does not expose ElyaGPT; "
                  "run via the MoE path or refactor the trainer")
            return
        model.load_state_dict(torch.load(ckpt, map_location=device))
        model.eval()
        gen_fn = lambda p: (None, greedy(model, p))
    else:
        import train_elya_moe as M                # noqa
        model = M.ElyaMoE().to(device)
        model.load_state_dict(torch.load(ckpt, map_location=device))
        model.eval()
        gen_fn = lambda p: moe_greedy(model, p)

    n = len(questions)
    tot_exact = tot_pfx = 0.0
    n_degen = 0
    route_ok = 0
    rows = []
    for q in questions:
        e, gen = gen_fn(q + " ")
        ex, pfx, best = score(gen, va[q])
        deg = degenerate(gen)
        tot_exact += ex
        tot_pfx += pfx
        n_degen += deg
        if e is not None:
            truth = classify(q)
            if truth is not None and truth == e:
                route_ok += 1
        rows.append({"q": q, "gen": gen, "exact": ex, "prefix": pfx,
                     "expert": e, "degenerate": bool(deg)})

    print(f"\n=== {kind}  ({n} distinct questions) ===")
    print(f"  exact match      {tot_exact:5.1f}/{n}  ({100*tot_exact/n:5.1f}%)")
    print(f"  mean prefix      {100*tot_pfx/n:5.1f}%")
    print(f"  degenerate       {n_degen:5d}/{n}  ({100*n_degen/n:5.1f}%)")
    if kind == "moe":
        routed = sum(1 for r in rows if r["expert"] is not None)
        print(f"  router agreement {route_ok}/{routed} vs keyword labels")
    print("\n  worst 5 by prefix:")
    for r in sorted(rows, key=lambda r: r["prefix"])[:5]:
        tag = f"[{NAMES[r['expert']]}]" if r["expert"] is not None else ""
        print(f"    {tag} {r['q'][:34]:36s} -> {r['gen'][:40]!r}")
    print("\n  best 5 by prefix:")
    for r in sorted(rows, key=lambda r: -r["prefix"])[:5]:
        tag = f"[{NAMES[r['expert']]}]" if r["expert"] is not None else ""
        print(f"    {tag} {r['q'][:34]:36s} -> {r['gen'][:40]!r}")

    out = HERE / f"eval_{kind}.json"
    out.write_text(json.dumps(
        {"kind": kind, "n": n, "exact": tot_exact / n,
         "prefix": tot_pfx / n, "degenerate": n_degen / n,
         "rows": rows}, indent=2))
    print(f"\n  -> {out}")

@torch.no_grad()
def greedy(model, prompt, n=70):
    toks = list(prompt.encode("ascii", "replace"))[-CTX:]
    x = torch.tensor([toks], dtype=torch.long, device=device)
    out = []
    for _ in range(n):
        lg = model(x[:, -CTX:])[0, -1]
        nl = lg[10].item()
        lg[:32] = float("-inf"); lg[127:] = float("-inf"); lg[10] = nl
        t = int(lg.argmax())
        if t == 10:
            break
        out.append(t)
        x = torch.cat([x, torch.tensor([[t]], device=device)], 1)
    return bytes(out).decode("ascii", "replace")

@torch.no_grad()
def moe_greedy(model, prompt, n=70):
    b = list(prompt.encode("ascii", "replace"))[:CTX]
    pi = torch.zeros(1, CTX, dtype=torch.long, device=device)
    pm = torch.zeros(1, CTX, device=device)
    pi[0, :len(b)] = torch.tensor(b, device=device)
    pm[0, :len(b)] = 1.0
    e = int(model.route(pi, pm).argmax(-1))
    x = torch.tensor([b], dtype=torch.long, device=device)
    out = []
    for _ in range(n):
        lg = model(x[:, -CTX:], e)[0, -1]
        nl = lg[10].item()
        lg[:32] = float("-inf"); lg[127:] = float("-inf"); lg[10] = nl
        t = int(lg.argmax())
        if t == 10:
            break
        out.append(t)
        x = torch.cat([x, torch.tensor([[t]], device=device)], 1)
    return e, bytes(out).decode("ascii", "replace")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(2)
    run(sys.argv[1], sys.argv[2])
