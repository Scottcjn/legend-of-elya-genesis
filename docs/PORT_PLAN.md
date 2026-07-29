# Legend of Elya — Sega Genesis Port Plan

**Target hardware**: Sega Genesis Model 1 (VA-series, 1601, 1989–1991) — the
discrete-YM2612 variant with the headphone jack. If it runs on BlastEm, it runs
on this board.

**Goal**: World's first LLM on a Sega Genesis. 16-bit Elya sprite whose mouth
animates as tokens stream into a dialog box, powered by real on-console
transformer inference. No canned responses in the final build.

---

## 1. The gap we're closing (N64 → Genesis)

| | N64 (shipped) | Genesis (target) |
|---|---|---|
| CPU | MIPS R4300i @ 93.75 MHz, FPU | Motorola 68000 @ 7.67 MHz, **no FPU** |
| RAM | 4MB RDRAM (256KB KV cache) | **64KB work RAM** total |
| Model | 819K params, 4L/128d, ctx 64 | ~64–128K params, 2L/48–64d, ctx 32 |
| Math | float32 on FPU | **ternary/tetranary, multiplication-free** |
| Weights | 458KB Q8 in ROM | ~40–120KB packed ternary in ROM |
| Speed | 60 tok/s emu, 1–3 tok/s real HW | target 0.5–2 tok/s real HW |

Real N64 hardware only achieved 1–3 tok/s. A Genesis hitting ~1 tok/s is the
same class of experience, on hardware from 1989.

## 2. Why multiplication-free is the whole ballgame

68000 instruction costs: `MULS.W` ≈ 70 cycles. `ADD.W` ≈ 4–8 cycles.
`MOVE.B (A0,D0.W),D1` (table lookup) ≈ 14 cycles.

- **Ternary weights** {-1, 0, +1} (BitNet b1.58 style): dot product = add,
  subtract, or skip. Zero multiplies in the hot loop.
- **Tetranary extension** (Scott's theory, from the neuromorphic coffers
  4-state confidence work): add a 4th state — e.g. {-2, -1, 0, +1} or
  {-1, 0, +1, +2} per-block — a "strong" weight costs ONE extra add
  (shift-free doubling via `ADD Dn,Dn` on the activation is 4 cycles).
  Doubles expressive range of a block for near-zero cost. 2 bits/weight
  packed 4-per-byte, or 3^5=243 pure-ternary packed 5-per-byte.

## 3. "Blast Processing for inference" — replace compute with memory

The Genesis demoscene doctrine: cartridge ROM is zero-wait-state and memory
mapped. Anything computable offline goes in ROM as a table.

- **T1 — Packed-weight decode LUT**: 256-entry ROM tables map a packed weight
  byte directly to unpacked sign/skip micro-op sequences (or jump offsets into
  unrolled add/sub code — computed-goto matmul, the 68000 `JMP (A0,D0.W)`
  trick).
- **T2 — Per-column add/sub index streams**: offline, convert each weight
  matrix column into a stream of "add activation[i] / sub activation[j]"
  offsets. Inner loop is `MOVE.W (A2,D0.W),D1` + `ADD.W D1,D2` — no decode at
  all. Trades ROM for speed (ROM is cheap: 4MB standard, more with SSF2-style
  mapper).
- **T3 — Activation-quantized group LUTs**: quantize activations to 4 bits;
  a (packed-5-ternary-weights, activation-nibble) pair indexes a precomputed
  partial-sum table. Evaluate feasibility: table size vs accuracy.
- **T4 — exp()/softmax as ROM tables**: 8-bit domain exp LUT (the N64 needed
  a Taylor series; we need one `MOVE.B`). Same for the sampling temperature
  curve.
- **T5 — VDP DMA for presentation**: mouth tiles + dialog glyphs DMA'd to VRAM
  during VBlank. The 68000 never draws; it only thinks.
- **T6 — Z80 as the "body"**: Z80 runs sound driver AND paces mouth-flap
  cadence from a shared mailbox in Z80 RAM, so Elya keeps moving even while
  the 68K is deep in a forward pass. (Same hemisphere-split as the NUMA
  coffers: 68K = cortex, Z80 = brainstem.)
- **T7 — Line-interrupt timeslicing**: HInt callback yields inference in
  bounded slices so music/animation never hitch. Inference is a coroutine,
  the display is the scheduler.

## 4. Memory budget (64KB work RAM)

| Item | Size |
|---|---|
| KV cache, int8: 2 layers × 32 ctx × 64 dim × 2 (K,V) | 8 KB |
| Activations/scratch (int16 vectors, 4×64d ffn) | ~2 KB |
| Logits (vocab 96–128, int16) | ≤256 B |
| Game state, sprites, dialog buffers | ~4 KB |
| SGDK runtime + stack | ~8 KB |
| **Headroom** | **~40 KB** |

Vocab: printable-ASCII subset (96) or byte-level 256 like N64 — decide after
tokenizer experiments. ROM trie tokenizer is free (T1 doctrine).

## 5. Phases

1. **P0 — Scaffold (this commit)**: SGDK project boots in BlastEm. Elya face
   sprite (procedural tiles, auburn hair per canon), dialog box, mouth flaps
   synced to a stubbed token streamer with tok/s counter.
2. **P1 — Integer inference core**: port nano_gpt.c to int-only fixed point
   (Q8 weights, int16 activations, int32 accum), host-side C harness first
   (compile on Linux, verify vs PyTorch), then 68K build.
3. **P2 — Ternary/tetranary retrain**: train the shrunk model with ternary
   QAT on the Victus/.136 V100, export packed weights + T2 index streams.
4. **P3 — Blast tricks**: computed-goto matmul, ROM LUT exp, Z80 mailbox,
   HInt timeslicing. Benchmark in BlastEm cycle counts.
5. **P4 — Real hardware**: Mega EverDrive flash cart on the 1601. The
   headphone jack gets the discrete YM2612 voice bleeps it deserves.
6. **Panel review**: fan the P1/P2 design to Grok + Codex + Opus
   (design-panel / tribrain) before committing the weight format.

## 6. Open questions for the multi-brain panel

- T2 index streams vs T3 group LUTs: ROM cost vs cycle cost crossover?
- Tetranary 4th state: ±2 tier vs a per-block scale (Q-block style)?
- Attention: full softmax vs argmax-ish "hard attention" (cheaper, and
  matches the constraint-bound-selection PSE doctrine)?
- Vocab 96 vs 256 vs ROM-trie subwords?
- Can the Z80 contribute real compute (8-bit partial sums) or is bus
  contention a net loss?

---

## P1 — Locked fixed-point spec (implemented in src/elya_gpt.c)

| Quantity | Format | Range | Notes |
|---|---|---|---|
| Activations | int16 Q3.12 | ±8 | fake-quantized in training |
| KV cache | int8 Q4.4 | ±8 | 16KB total (2L x 64ctx x 64d x 2) |
| Embedding (tied) | int8 Q2.6 | ±2 | doubles as logit projection |
| Matmul weights | 2-bit ternary | {-1,0,+1} | code 11 reserved for tetranary ±2 |
| Requant | (acc*M)>>S, M≤127 | — | int32-safe: 2^23 * 2^7 < 2^31 |
| RMSNorm | isqrt32 + Q10 reciprocal | gain ≤16x | parameter-free |
| Softmax | ROM LUT exp(-i/16) Q14 | 256 entries | T4 doctrine: no Taylor series |
| Attention mixdown | num/(sum>>8) | — | ±0.4% vs exact, avoids int64 |
| Sampling | greedy argmax 32..126 + '\n' | — | matches proven N64 path |

Game title: **ELYA INTO DREAMS** (ROM header + title screen).
