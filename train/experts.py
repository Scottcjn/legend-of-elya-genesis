#!/usr/bin/env python3
"""
experts.py — split the Elya corpus into topic experts for the Lock-On MoE.

The split is derived from the corpus's own question distribution (see
docs/LOCKON_MOE.md), not invented: identity / quest / rustchain / hardware
are the four clusters the 122 QA pairs actually fall into.

Each expert is trained on its own shard, so a 114K-parameter model has to
memorize ~30 QA pairs instead of 122 — inside its capacity instead of well
past it.
"""
import ast, re
from pathlib import Path


EXPERTS = [
    ("identity", [
        r"who are you", r"your name", r"where are you from", r"your purpose",
        r"flameholder", r"are you wise", r"what do you love", r"a secret",
        r"who made you", r"are you alive", r"do you dream", r"your creator",
        r"sophia", r"elya", r"who is scott",
    ]),
    ("quest", [
        r"dungeon", r"proceed", r"lurks", r"need here", r"help me",
        r"encourage", r"quest", r"danger", r"treasure", r"path", r"door",
        r"monster", r"boss", r"realm", r"where do i", r"what should i",
        r"lost", r"afraid", r"tired",
    ]),
    ("rustchain", [
        r"rustchain", r"\brtc\b", r"earn", r"node", r"epoch", r"antiquity",
        r"mining", r"miner", r"token", r"wallet", r"block", r"chain",
        r"reward", r"attest", r"consensus",
    ]),
    ("hardware", [
        r"\bg4\b", r"\bg5\b", r"power8", r"altivec", r"vec_perm",
        r"big-endian", r"endian", r"vr4300", r"\brsp\b", r"console",
        r"cpu", r"powerpc", r"\bmips\b", r"silicon", r"runs this",
        r"processor", r"\brom\b", r"\bram\b", r"68000", r"genesis",
    ]),
]
N_EXPERTS = len(EXPERTS)
NAMES = [n for n, _ in EXPERTS]

def load_corpus():
    """Canonical corpus, vendored in-repo (train/corpus.json)."""
    import json
    d = json.loads((Path(__file__).resolve().parent / "corpus.json").read_text())
    return d["QA_PAIRS"], d["CORPUS_LINES"]

def classify(line):
    """Return expert index for a corpus line, or None if it matches none."""
    low = line.lower()
    q = low.split(":")[0] if ":" in low else low
    best, best_hits = None, 0
    for i, (_name, pats) in enumerate(EXPERTS):
        hits = sum(1 for p in pats if re.search(p, q))
        # a match in the answer counts less than one in the question
        hits += 0.4 * sum(1 for p in pats if re.search(p, low)) if hits == 0 else 0
        if hits > best_hits:
            best, best_hits = i, hits
    return best

def shard():
    """Return (shards, router_examples).

    shards[i]        : list of corpus lines for expert i
    router_examples  : list of (prompt_text, expert_index) for the router
    """
    qa, bg = load_corpus()
    shards = [[] for _ in range(N_EXPERTS)]
    router = []
    unmatched = []

    for line in qa:
        e = classify(line)
        if e is None:
            unmatched.append(line)
            continue
        shards[e].append(line)
        prompt = line.split(":")[0] + ":"
        router.append((prompt, e))

    # background lore lines: give them to whichever expert they match,
    # and to identity as a fallback so no expert is starved of prose
    for line in bg:
        e = classify(line)
        shards[e if e is not None else 0].append(line)

    # unmatched QA goes to every expert: shared connective tissue rather
    # than lost data (there are few of these and they are generic)
    for line in unmatched:
        for s in shards:
            s.append(line)

    return shards, router, unmatched

if __name__ == "__main__":
    shards, router, unmatched = shard()
    print(f"{N_EXPERTS} experts, {len(router)} routed prompts, "
          f"{len(unmatched)} generic QA shared across all\n")
    for i, name in enumerate(NAMES):
        n_qa = sum(1 for p, e in router if e == i)
        print(f"  {i} {name:10s} {len(shards[i]):4d} lines  ({n_qa} routed QA)")
        for line in shards[i][:2]:
            print(f"        {line[:66]}")
    if unmatched:
        print(f"\n  generic (all experts): {len(unmatched)}")
        for line in unmatched[:5]:
            print(f"        {line[:66]}")
