# Making a 7.67 MHz 68000 Think — Speed Plan & Measured Results

Each claim below states **which machine it was measured on**. That
distinction is load-bearing and an earlier version of this document
elided it: the headline speedup was measured on an x86 host, not on a
68000, and the honest 68000 figure is much smaller. Where a technique
failed, the failure is written down; the negative results are the most
useful part of this document.

⚠️ **Measurement debt (open):** no timing has yet been taken on a 68000,
cycle-accurate emulator or otherwise. Until that exists, every speed
number here is a host-x86 proxy or a static estimate, and is labelled as
such.

## Baseline economics

| Operation | 68000 cycles |
|---|---|
| `ADD.W Dn,Dm` | 4 |
| `MOVE.B (A0,D0.W),D1` | 14 |
| `MULS.W <ea>,Dn` | 70 |
| `__mulsi3` (32x32, GCC helper) | ~250 |
| Taken branch | 10 |
| Cycles per NTSC frame | 127,840 |

The 68000 has **no 32x32 multiply**. Every C expression GCC believes is
32-bit becomes a `__mulsi3` call. This single fact drove two of the four
wins below.

---

## 1. SGT2 index streams — 11.3x, the big one

**Before**: weights packed 2 bits each, 4 per byte. The inner loop shifted,
masked, and branched three ways per weight.

**After**: each output row carries an explicit list of activation indices
to ADD and a list to SUBTRACT. Zero weights are simply absent from both
lists — they cost nothing at all, not even a test.

```
row := u16 n_add, u16 n_sub, n_add index bytes, n_sub index bytes
```

| | µs/token (**host x86**, not 68000) |
|---|---|
| SGT1 2-bit packed | 403.8 |
| **SGT2 index streams** | **35.6** |

**11.3x on the x86 host harness.** The 68000 speedup is **UNMEASURED**;
static estimate **~1.7-3x**. The gap is not a rounding error, it is the
instrument: SGT1's inner loop has a data-dependent 3-way branch per
weight, which an out-of-order x86 punishes at ~15-20 cycles per
misprediction, while the 68000 has no branch predictor to mispredict (a
taken branch is a flat 10 cycles). The ratio is therefore dominated by
x86-specific effects that do not exist on the target.

Static estimate, from the cycle table above and the measured 58-65%
nonzero density (98,304 weights/token):
  SGT1 ~45-55 cy/weight  -> ~4.5-5.0M cycles/token
  SGT2 ~36-46 cy/nonzero -> ~2.3-2.9M cycles/token

The direction and the structural argument (trade ROM for decode cycles)
hold. The magnitude does not transfer.

Equivalence: `host/sgt1_to_sgt2.py` converts a blob between formats so
both engines run on the same weights. ⚠️ The comparison performed was
**greedy generations over 6 prompts**, which is a lossy observer — two
engines can differ in every logit and still emit identical text. The
claim this supports is "identical greedy output on 6 short prompts", NOT
"byte-identical output". A logit-exact suite over random prompts
(including length > CTX, the full byte range, and all-positive /
all-negative rows) is owed.

Cost: 2.02x ROM (41.5KB → 83.7KB). On a 4MB cartridge that is free. This
is the T2 doctrine from PORT_PLAN.md: **ROM is cheap, decode cycles are
not.**

Codex's follow-up (not yet implemented): the claimed inner loop
`MOVE.B (A1)+,D0 / ADD.W (A2,D0.W),D2` is not quite realizable because the
68000 has no scaled indexing — a byte index must be cleared, doubled, then
used. Generating straight-line `ADDA.W 2*index(Ain),Aacc` instructions into
ROM (one instruction per nonzero weight, ~250KB) removes the decode
entirely. That is the next big lever.

## 2. KV ring buffer — deletes a 16KB memmove per token

The model has **no positional encoding**, so attention is order-invariant:
the output is a weighted sum over a *set* of KV pairs. Overwriting the
oldest ring slot therefore produces bit-identical results to sliding the
whole window, minus the copy.

`EG_CTX` is 64, a power of two, so the wrap is a single `AND`.

Invisible on host x86 (memcpy is vectorized); on a real 68000 a 16KB
byte-copy is roughly 40ms — per token.

## 3. Attention collapse — a split verdict

### 3a. Lossless zero-prune: works, free

A position whose Q14 exp-LUT weight rounds to zero contributes *exactly*
nothing to the V mixdown. Skipping it is correct by construction, verified
byte-identical.

### 3b. Hard Top-K (PSE-style): FAILS on this model

The POWER8 build runs Top-K:8 vec_perm collapse successfully. The same
idea here degrades badly:

| EG_TOPK | Output |
|---|---|
| 4 | `is is is is is is is…` |
| 8 | *(empty — immediate stop token)* |
| 16 | `Old` then nothing |
| 32 | degraded |
| 48 | matches baseline |
| 64 (off) | baseline |

**Why**: with no positional encoding this is a *bag-of-context* model. Its
signal is spread across the whole window rather than concentrated in a few
sharp lookups, which is the opposite of the peaky attention that makes
Top-K work in large LLMs. Constraint-bound selection needs something to
select *between*; flat attention has no winners to keep.

Kept behind `EG_TOPK`, default off. **Revisit only after adding positional
encoding** — that change would likely make Top-K viable, and is the honest
prerequisite.

## 4. Killing accidental software multiplies

Found by the Codex audit, verified by counting call sites in generated
m68k assembly (`m68k-elf-gcc -S`).

| Site | Problem | Fix |
|---|---|---|
| RMSNorm scale | `r` tracked as 32-bit through a division | explicit `mul_ss()` |
| RMSNorm sum-of-squares | `(int32_t)h * h` cast forced 32-bit | `(h * h)` |
| Attention V mixdown | `s_selsc` was `int32_t` | `int16_t` (Q14 ≤ 16384) |
| Score dot product | `(int32_t)q * k` | `mul_ss()` |

`__mulsi3` call sites: **4 → 2**. The remaining two are genuinely 32-bit
(the requant `acc * M`, and a divide). Codex's suggestion for the requant:
since `M` is a per-tensor constant ≤ 127, generate a shift/add chain per
tensor — 2-4 terms instead of a helper call.

Host output stayed byte-identical throughout.

## 5. Presentation bug: a 4%-of-frame tax on inference

Found by the Opus VDP audit, in code from earlier this session. The floor
scroll computed `dsFrame * (r + 3)` with `dsFrame` as `vu32` — a 32x16
multiply calling the Sozobon `lmul` helper 12 times per frame, ~3000-4800
cycles (~4% of every frame) taken directly from the forward pass.

Replaced with incremental `s16` accumulators. **Background animation must
never do a 32-bit multiply.**

Related, also from that audit: `DMA_QUEUE` only flushes inside
`SYS_doVBlankProcess()`, which the main loop reaches once per *token*. Any
background animation queued that way freezes for a full second while she
thinks. Per-frame background work must use direct VDP writes or immediate
DMA from the V-int callback.

---

## Next levers, ranked

1. **Multi-byte tokens (gigatoken doctrine)** — the largest remaining win
   by far, and algorithmic rather than micro. We currently run one full
   forward pass **per character**. A BPE vocabulary emitting 3-4 characters
   per pass is a 3-4x effective speedup that no instruction-level work can
   match. Credit and inspiration: **[marcelroed/gigatoken](https://github.com/marcelroed/gigatoken)**
   (MIT) — Marcel Rød's Rust BPE tokenizer, whose core lesson is exactly
   this project's thesis: *replace computation with precomputed structure*.
   Its hand-compiled per-family pretokenizer DFAs, SWAR scans, and pretoken
   cache are the same trade we make with ROM index streams. Elyan Labs
   contributed the POWER8 VSX tier to it (fork `Scottcjn/gigatoken`,
   upstream issue #49): 3.37x single-thread, 1.62 GB/s on 7 cores.
   On the Genesis the equivalent is a ROM trie: byte-at-a-time descent
   through a table, no allocation, no hashing — free under the T1 doctrine.
2. **Generated straight-line matvec** (`ADDA.W`/`SUBA.W` per nonzero) —
   removes index decode; ~250KB ROM.
3. **Generated requant chains** — kills the last hot `__mulsi3`.
4. **Context 32 instead of 64** — halves score and V work, and halves the
   KV cache to 8KB. Costs memory of the conversation.
5. **Positional encoding, then revisit Top-K** — would make the collapse
   in §3b actually work.

## Not worth doing (Codex verdict, endorsed)

- 16-bit matvec accumulator — provably unsafe (overflow).
- Native word casts into the current SGT2 layout — odd rows would address-
  error on a real 68000.
- Computed-goto dispatch — the taken-jump cost remains; straight-line code
  is strictly better.
- Z80 as an inference coprocessor — bus arbitration and marshaling
  overwhelm the useful 8-bit work. Keep it on audio.
- Further weight compression (3^5 packing) — ROM is cheap, decode is not.
- Group LUTs before generated matvec is measured — nibble assembly and ROM
  lookups are not free on a cacheless 68000.
- HInt timeslicing *as a speed optimization* — it improves responsiveness,
  it does not add throughput.
