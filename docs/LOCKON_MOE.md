# Lock-On MoE — sharded experts in cartridge ROM

*"One huge cart."* — the Sonic 3 & Knuckles idea, applied to a mixture of
experts.

## Why this is the right architecture for a Genesis (and not for a GPU)

On every modern machine, a mixture of experts saves **compute** but not
**memory**: all expert weights must sit in RAM or VRAM whether they fire or
not, and activating one means a copy or a DMA.

The Genesis inverts this. Cartridge ROM is **memory-mapped and directly
executable**. Activating an expert is not a load — it is one write to the
mapper register at `$A130F1`, after which those weights are simply
*addressable*. No copy. No decompression. No RAM cost at all.

| | GPU MoE | Genesis Lock-On MoE |
|---|---|---|
| Expert storage | VRAM (scarce) | ROM (abundant, 32MB) |
| Cost to activate | DMA / copy | **one register write** |
| Working RAM | all experts | **zero** — weights stay in ROM |
| Limit on total params | VRAM | cartridge size |

The 68000's scarce resource is cycles and RAM; its abundant resource is
address space. MoE spends the abundant one to save the scarce ones. This
is the same trade as SGT2 index streams (ROM for cycles), one level up.

## The capacity math

Verified against SGDK's `mapper.h`: the SEGA mapper views ROM as 512KB
banks, region 0 (`$000000-$07FFFF`) fixed, regions 1-7 remappable, bank
index 0-63.

⚠️ **This table was wrong in the first version of this document** and is
corrected here. The error: it used the size of a *single-model blob*
(83.7 KB, which includes the 16 KB shared embedding) as the size of an
*expert shard* (which does not). Measured from the actual exported
`res/elya_moe.bin`:

| | |
|---|---|
| Shared block (embedding + exp LUT + router) | **17,119 B**, resident |
| One expert **shard** | **~72.5 KB** (not 83.7) |
| Experts per 512KB bank | **7** |
| **Plain 4MB cart, no mapper** | **57 experts** |
| SEGA mapper ceiling (64 banks x 512KB) | 32MB ≈ 470 shards |
| ⚠️ **What the engine actually supports** | **16** (`EG_MAX_EXPERTS`) |

The last row matters: `src/elya_gpt.h` caps experts at 16 and `eg_init`
enforces it. The 32MB ceiling describes a machine that would need a code
change to reach, so quoting it as a current capability is not honest.
57-in-4MB is real once the cap is raised; 16 is what runs today.

⚠️ **Needs hardware verification before we commit to >4MB**: which
EverDrive models implement the SEGA/SSF2 mapper, and what maximum ROM they
accept. Emulator support (BlastEm) is not proof for the real 1601. Until
that is checked on the actual cart, target **4MB, no mapper** — which is
already 57 experts and needs no special hardware.

## Status: UNPROVEN

⚠️ The first MoE training run produced **60 spaces for every prompt**
(`train/moe_vectors.json`) and misrouted 2 of 6 probes. The blob was
exported and committed anyway; nothing in the pipeline gated on output
quality. Treat everything below as the *hypothesis under test*, not a
result. The decisive control — one 114K model trained on a single shard —
is what settles it.

## Why this MIGHT fix our actual problem (hypothesis)

The current model is 114,688 parameters trying to memorize 122 QA pairs
spanning identity, dungeon lore, RustChain economics, and vintage CPU
trivia. It reaches eval 0.9671 and produces `"I am Aof cepm of of 1979."`
— it has learned the *subject matter* but cannot hold a sentence. That is
a capacity problem, not a training problem: five hyperparameter sweeps in
autotrain moved it barely at all.

Sharding turns one hard task into four easy ones. Each expert sees ~30 QA
pairs instead of 122 — well inside what 114K parameters can memorize
cleanly.

## Expert map (derived from the corpus, not invented)

Measured by running `train/experts.py` — earlier figures in this table
were guessed, and were wrong:

| Expert | Topics | routed QA | total lines |
|---|---|---|---|
| 0 IDENTITY | who/name/from/purpose, Flameholder, wisdom, love | 21 | **279** |
| 1 QUEST | dungeon, proceed, what lurks, help, encouragement | 26 | 76 |
| 2 RUSTCHAIN | RustChain, RTC, earning, nodes, epochs, antiquity | 18 | 85 |
| 3 HARDWARE | G4/G5/POWER8, AltiVec, vec_perm, endianness, RSP | 26 | 78 |

⚠️ **Expert 0 is 3.5x oversized** because `experts.py` dumps every
unclassifiable background-lore line into identity as a fallback. That
breaks the central premise for that expert specifically — it has a
*harder* job than the single model did, not an easier one. Fix before
drawing conclusions from expert 0.

## Routing

**Shared and always resident** (fixed region 0, never banked out):
- embedding table, 16KB — every token needs it
- exp LUT
- the router itself

**Router**: a learned linear classifier over the mean-pooled embeddings of
the prompt — `EG_EMBED x N_EXPERTS` = 64x4 = **256 weights**, ternary like
everything else. Effectively free, and genuinely learned rather than a
hand-written keyword table.

**Granularity: per-prompt, not per-token.** She reads the question, picks
which part of her mind to think with, and stays there for the answer. This
is cheaper (one bank switch per answer, not per token), more stable (no
mid-sentence expert flapping), and it matches the fiction: the dream has
rooms, and she walks into one.

Per-token routing remains possible — a bank switch is only a register
write — but there is no reason to pay for the instability.

## Blob format: SGTM

```
'SGTM' u8 n_experts, u8 layers, u8 heads, u8 pad,
       u16 embed, u16 vocab, u16 ctx, u16 flags
shared:  emb int8[vocab*embed]        (tied logits)
         explut u16be[256]
         router: u16 M, u8 S, u8 pad, index streams [n_experts][embed]
experts: u32 offset[n_experts]        (byte offset from blob start)
         then each expert's 6 tensors in SGT2 layout
```

Expert offsets are absolute so an expert can live in any bank; the loader
converts an offset into (bank, in-bank address) when the mapper is used.

## Engine changes

- `eg_init` parses SGTM, keeps shared pointers, stores expert offsets.
- `eg_route(prompt)` mean-pools prompt embeddings, runs the 4-way ternary
  classifier, returns an expert id.
- `eg_select_expert(id)` maps the bank (when >4MB) and repoints the six
  `EgTensor` structs. **No data is moved.**
- Everything else is unchanged — the matvec does not care which bank its
  pointer lives in.

## Honest risks

1. **Expert collapse**: if the router is bad, a wrong expert answers
   confidently and the result is worse than the single model. Mitigation:
   measure router accuracy on held-out prompts *before* shipping, and fall
   back to a designated generalist expert when the router's margin is low.
2. **Bank switching during a forward pass** is safe for data but a bank
   switch that maps out *executing code* is fatal. Keep all code in the
   fixed region 0; only weights get banked.
3. ~~The N64 port cannot use this~~ — **WRONG, retracted.** The N64 can,
   and its version is richer: libdragon's `dma_read_async()` lets the next
   expert stream in *while the CPU generates from the current one*, which
   the Genesis cannot do. See `legend-of-elya-n64/docs/STREAMING_MOE.md`.
4. ROM size is not free on real hardware: a 4MB EverDrive load is fine; a
   32MB image needs mapper support confirmed on the actual device.

---

## RESULT: the capacity hypothesis is CONFIRMED (2026-07-29)

The control the rigor audit demanded: one 114K ternary model, unchanged in
every respect except that it trained on **one shard** (rustchain, 50
lines) instead of the full 122-pair corpus. Scored against the corpus's
own reference answers, best-match over all valid answers per question,
same greedy decoding, same C engine, same six questions.

| question | full-corpus | single-shard |
|---|---|---|
| What is RustChain? | 0 ch | **28 ch** |
| What is RTC? | 3 ch | 2 ch |
| How do I earn RTC? | 1 ch | **13 ch** |
| What is a node? | 2 ch | **13 ch** |
| What is epoch? | 0 ch | **28 ch** |
| What is proof of antiquity? | 0 ch | 3 ch |
| **mean exact prefix** | **1.0 ch** | **14.5 ch** |

**14.5x**, and eval loss halved (0.9671 → 0.5001). Qualitatively:

```
full-corpus   "What is epoch?" -> "Man A Vecttati SIP?: MIPS MD MIPS MIPS ian"
single-shard  "What is epoch?" -> "Epochs settle miner rewards minexpaintes."
                        truth:    "Epochs settle miner rewards each ten minutes."
```

She now *starts every answer correctly*. Capacity was a real limit, the
sharding premise holds, and the MoE is worth building properly.

### But the failure has moved, and its new shape is informative

Every answer is **correct then degrades**, typically after 13-28
characters. That is not what a pure capacity ceiling looks like — a model
short on parameters fails uniformly, not positionally.

Correct-then-degrade with distance is the signature of **missing
positional encoding**. This model has none (which is exactly what makes
the KV ring buffer valid, see SPEED_PLAN.md §2). Early in an answer there
is little context and the bag is unambiguous; as tokens accumulate the
order-free bag of context gets muddier and the model loses its place in
the sentence.

The same absent feature has now surfaced three times independently — as
the justification for the ring buffer, as the (retracted) explanation for
Top-K, and now as the shape of this failure. It is the most likely single
next win, and it is cheap to test.

### Standing hypotheses, in priority order

1. **Positional encoding** — predicted to fix the degradation tail. Also
   makes real Top-K worth re-measuring. Costs a retrain, ~64x64 weights,
   and invalidates the ring-buffer shortcut (attention stops being
   order-invariant), which is a real trade to weigh.
2. **Quinary weights** (`EG_LEVELS=2`, already implemented) — tests
   whether a ternary QAT noise floor contributes. Orthogonal to 1.
3. **MoE end-to-end** — now justified, but fix expert 0's 3.5x
   oversized shard first or that expert reproduces the full-corpus
   failure.
