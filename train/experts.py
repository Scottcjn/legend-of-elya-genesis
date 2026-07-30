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
    # 8 narrow topics. Plain lowercase SUBSTRINGS, no regex: this is the
    # same matching the ROM router performs, so training labels and
    # runtime routing cannot disagree. Order matters - the expert with the
    # most hits wins, so specific topics precede general ones.
    ("origin", [
        "how old are you", "are you old", "are you vintage",
        "remember being born", "older than this",
        "when were you born", "what year were you born",
        "when is your birthday", "your origin", "born in 19",
    ]),
    ("identity", [
        "who are you", "your name", "where are you from", "your purpose",
        "flameholder", "are you wise", "what do you love", "a secret",
        "who made you", "are you alive", "do you dream", "your creator",
        "do you fear", "are you happy", "can you say no", "you love",
        "victorian study",
    ]),
    ("zelda", [
        "zelda", "link", "ganon", "navi", "saria", "malon", "impa",
        "epona", "triforce", "master sword", "hyrule", "kokiri",
        "death mountain", "lon lon", "goron", "zora", "ocarina",
        "temple", "medallion",
    ]),
    ("quest", [
        "dungeon", "proceed", "lurks", "need here", "help me",
        "encourage", "quest", "danger", "treasure", "monster",
        "boss", "realm", "where do i", "what should i",
        "i am lost", "i am afraid", "should i go", "waits at the end",
    ]),
    ("rustchain", [
        "rustchain", "rtc", "earn", "node", "epoch",
        "antiquity", "mining", "miner", "token", "wallet", "attest",
        "reward", "consensus", "cheat the chain", "much rtc",
    ]),
    ("retro", [
        "amiga", "c64", "commodore", "apple ii", "atari",
        "the nes", "snes", "ti-99", "trs-80", "zx spectrum",
        "6502", "spectrum", "2600", "retro machine", "vintage hardware",
    ]),
    ("hardware", [
        "g4", "g5", "power8", "altivec", "vec_perm",
        "big-endian", "endian", "vr4300", "rsp", "rdp",
        "console", "powerpc", "mips", "68000", "genesis",
        "runs this", "expansion pak", "n64", "render",
        "z80", "vdp", "much ram", "chip slow", "this console",
    ]),
    ("meta", [
        "quantization", "q4", "your model", "language runs you",
        "how big", "parameters", "neural", "transformer",
        "how do you think", "how fast do you think", "why ternary",
        "your experts", "ternary",
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

    # Background lore. Only give a line to an expert it actually matches;
    # round-robin the rest. The old fallback sent EVERY unclassifiable
    # line (219 of 303) to identity, making that shard 291 lines against
    # 83-92 for the others - which inverted the whole premise for expert
    # 0, giving it a harder job than the unsharded model had.
    # Lore is OPT-IN. The 50-line shard that reached 95% exact answers
    # trained on QA pairs ONLY (EG_SHARD mode zeroed CORPUS_LINES) - a
    # fact that turned out to be load-bearing, not incidental. Adding
    # ~140 lines of lore per expert pushed them out of that regime and
    # cost most of the gain. Keep shards small; lore is available behind
    # EG_SHARD_LORE=1 if fluency ever needs it.
    import os as _os
    if _os.environ.get("EG_SHARD_LORE") == "1":
        rr = 0
        for line in bg:
            e = classify(line)
            if e is None:
                e = rr % N_EXPERTS
                rr += 1
            shards[e].append(line)

    # Generic QA (no topic match) is round-robined, NOT copied into every
    # shard - duplicating 38 pairs four times was another 38 lines of
    # bulk per expert for no topical benefit.
    for i, line in enumerate(unmatched):
        shards[i % N_EXPERTS].append(line)

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
