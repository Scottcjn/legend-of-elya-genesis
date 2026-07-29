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

| | |
|---|---|
| One expert (SGT2, 2L/64d) | **83.7 KB** |
| Experts per 512KB bank | **6** |
| Max banks (SEGA mapper) | 64 |
| **Theoretical ceiling** | **32 MB ≈ 380 experts ≈ 43M parameters** |
| Plain 4MB cart, no mapper at all | **~45 experts** |

Note the last row: we do not even need a mapper to get a large gain. A
standard 4MB ROM holds ~45 experts today. The mapper is for later.

⚠️ **Needs hardware verification before we commit to >4MB**: which
EverDrive models implement the SEGA/SSF2 mapper, and what maximum ROM they
accept. Emulator support (BlastEm) is not proof for the real 1601. Until
that is checked on the actual cart, target **4MB, no mapper** — which is
already ~45 experts and needs no special hardware.

## Why this fixes our actual problem

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

| Expert | Topics | QA pairs |
|---|---|---|
| 0 IDENTITY | who/name/from/purpose, Flameholder, wisdom, love, secrets | ~30 |
| 1 QUEST | dungeon, how to proceed, what lurks, help, encouragement | ~35 |
| 2 RUSTCHAIN | RustChain, RTC, earning, nodes, epochs, proof of antiquity | ~30 |
| 3 HARDWARE | G4/G5/POWER8, AltiVec, vec_perm, big-endian, VR4300, RSP | ~27 |

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
3. **The N64 port cannot use this** — its cartridge is not memory-mapped
   the same way (weights are DMA'd from ROM to RDRAM), so there the MoE
   would cost real RAM. This technique is genuinely Genesis-specific.
4. ROM size is not free on real hardware: a 4MB EverDrive load is fine; a
   32MB image needs mapper support confirmed on the actual device.
