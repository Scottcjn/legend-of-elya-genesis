# Genesis (68000) measurement instrument — FINDINGS

Working copy: $S/genesis (copy of ~/legend-of-elya-genesis; upstream never modified).
Started 2026-08-04.

## Log

### [0] Environment survey
- MAME v0.277 at /usr/games/mame (genesis driver, no BIOS needed).
- BlastEm 0.6.2 flatpak com.retrodev.blastem.
- marsdev toolchain ~/marsdev/mars/m68k-elf (SGDK 1.81 makefile.gen).
- Engine src/elya_gpt.c (530 lines); src/main.c (1050 lines, VDP shell).
- Model res/elya_moe.bin 589474 bytes, SGTM (Lock-On MoE, index-stream weights).
- Prebuilt out/rom.bin = 1835008 bytes.
- host/harness.c compiles the SAME elya_gpt.c on x86 -> cheap token oracle,
  but says NOTHING about 68000 cycles.
- README "11.3x" is an x86 number. No 68000 measurement exists in-tree.

### [1] Bench ROM built
`$S/bench/` is a second SGDK project whose src/ *symlinks* the same
`src/elya_gpt.c` as the real ROM, so an engine edit is measured and shipped from
one file. It has its own `src/bench_main.c`:
 - VDP display disabled, interrupts disabled (removes VBlank IRQ + VDP refresh
   bus contention from the measured region -> reproducibility).
 - Writes `bench_marker = 1` before and `= 2` after the measured region.
 - Feeds the prompt "Who are you?: " (14 chars) then generates 24 tokens.
 - Stores tokens in `bench_tok[]`, count in `bench_ntok`, status 0x5A in
   `bench_status`, router expert in `bench_expert`.
Symbols land in work RAM: bench_marker 0xFF00E2, expert 0xFF00E6,
status 0xFF00E8, ntok 0xFF00E9, tok 0xFF00EA. Host reads them from
`out/symbol.txt` (never hard-coded).

Bench ROM = 1179648 bytes. Real ROM still builds: 1835008 bytes.

### [2] Host reference (x86, same engine source)
`host_bench.c` mirrors bench_main.c's loop exactly (fixed 24 tokens, no
early stop on newline).
  expert=1
  tokens=67,97,108,108,32,109,101,32,83,111,112,104,105,97,32,69,108,121,97,44,32,121,111,117
  text = "Call me Sophia Elya, you"
24 tokens. This is the correctness gate (not 3).

### [3] Instrument: MAME 0.277 + Lua
- `mame genesis -cart rom.bin` boots with NO BIOS. Confirmed.
- `cpu:total_cycles()` does NOT exist in this MAME's Lua API.
- `space:install_write_tap()` DOES exist -> tap `bench_marker`, and read
  `manager.machine.time` (attotime) inside the callback. cycles =
  dt_seconds * cpu.clock. This is the cycle-resolution path.
- `emu.add_machine_frame_notifier` also used, to report a frame count as a
  cross-check on the tap number.
Runner: `mamework/bench_run.py` (generates the Lua, parses the result file).

### [4] Static analysis of the 68000 asm (NOT a measurement — for targeting only)
`m68k-elf-gcc -O3 -S src/elya_gpt.c`:
- matvec index-stream inner loop, 7 insns / nonzero weight:
    moveq #0,d0 ; move.b (a1)+,d0 ; add.l d0,d0 ;
    move.w (a4,d0.l),a3 ; add.l a3,a0 ; cmp.l a1,a2 ; jne
  ~58 cycles per nonzero.
- per output row: `jsr __mulsi3` for `acc * M` (~250-280 cyc), 1152 rows/token.
- logits loop already narrowed by GCC to `muls.w (a1)+,d0` (good) but runs
  96 vocab x 64 embed = 6144 times/token.
- `num / sum8` = `jsr __divsi3`, 128 per token.
Weight blob (expert 1) nonzeros per forward pass: **66,297**.
  L0 wq2805 wk2743 wv2725 wo2750 wff1 11218 wff2 11046
  L1 wq2763 wk2819 wv2670 wo2724 wff1 11139 wff2 10895
All requant multipliers M are < 128 (65..121) -> a 32x16 multiply would do;
__mulsi3 is doing a full 32x32.

### [5] Instrument v1 was correct but pathologically slow — FIXED
First working run: 8m23s wall for ~33s of emulated Genesis time (~0.07x
realtime). Cause: the Lua runner polled `bench_status` from
`emu.register_periodic`, which MAME calls per scheduler timeslice.
Also `cpu.clock` does NOT exist in MAME 0.277's Lua device binding
(the run completed but crashed formatting the report — the number was lost).

Instrument v2 (current):
 - NO polling at all. The ROM writes bench_marker=1 (start), 2 (end), then
   fills bench_ntok/bench_status, then writes 4. The write tap on marker
   value 4 dumps everything and calls machine:exit().
 - Elapsed time is taken as MAME attotime deltas at the marker=1 and
   marker=2 taps, so the resolution is a bus cycle, not a frame.
 - 68000 clock is the NTSC constant 53693175/7 = 7670453.57 Hz (hard-coded,
   since MAME's Lua does not expose it). Raw attosecond deltas are reported
   too, so the cycle number can be re-derived from a different clock.
 - `-seconds_to_run` is the safety net, plus a machine_stop notifier.

Gotcha recorded: SGDK's makefile.gen globs `*.s` in the project ROOT, so an
`elya_gpt.s` left there from `gcc -S` gets assembled and breaks the build.

### [6] THE BUG THAT LOOKED LIKE A HUNG ROM — MAME Lua GC
Symptom: markers/frames stopped arriving ~3-6 emulated seconds in, every
time, at a different point depending on callback frequency. It looked exactly
like the 68000 hanging at token 8.

Cause: **MAME's Lua bindings garbage-collect notifier and memory-tap
subscriptions when you don't keep a reference to the returned handle.**
`sp:install_write_tap(...)` and `emu.add_machine_frame_notifier(...)` return
handles; dropping them on the floor silently unsubscribes at the next GC.

Fix: hold them in a global table (`KEEP`). Anyone driving MAME from Lua for
measurement needs this or their instrument lies by omission.

### [7] INSTRUMENT VALIDATED — baseline measured
Command: `mamework/bench_run.py ../bench --repeat 3`
```
[baseline run 1] cycles=220579814  tokens=Call me Sophia Elya, you
[baseline run 2] cycles=220579814  tokens=Call me Sophia Elya, you
[baseline run 3] cycles=220579814  tokens=Call me Sophia Elya, you
REPRODUCIBLE cycles: True  (1 distinct)
REPRODUCIBLE tokens: True
```
**Resolution: exact 68000 bus cycles**, derived from MAME attotime deltas at
the write taps (attotime is 1e-18 s; one 68000 cycle is 1.3e-7 s). Three
consecutive runs returned the *same integer*, so the instrument is
deterministic, not merely precise. Wall cost: ~1.6 s per run.

**BASELINE = 220,579,814 cycles** for 38 forward passes
(14 prompt chars + 24 generated), NTSC 68000 @ 7,670,453.6 Hz:
 - 5,804,732 cycles / token
 - 1725.4 frames @60Hz total, 45.4 frames/token
 - 0.757 s/token -> **1.32 tokens/s on real hardware**

Token gate (24 tokens, matches the x86 harness byte for byte):
`67,97,108,108,32,109,101,32,83,111,112,104,105,97,32,69,108,121,97,44,32,121,111,117`
= "Call me Sophia Elya, you", router expert = 1.

Per-pass profile (cycles), showing the context-length dependence:
 prompt passes 0..13: 5,257,220 -> 5,683,428
 gen passes 0..23:    5,719,590 -> 6,269,846

### [8] Optimizations, each measured on the validated instrument
All runs x2, both runs identical, tokens identical to the 24-token gate.

| # | change | cycles | vs prev | vs baseline |
|---|--------|--------|---------|-------------|
| - | baseline                       | 220,579,814 | -      | -      |
| 1 | V-mixdown loop reorder + row-pointer hoist | 215,317,208 | -2.39% | -2.39% |
| 2 | matvec index-stream 4x unroll  | 207,068,775 | -3.83% | -6.13% |
| 3 | requant `acc*M` via 2x MULU.W instead of __mulsi3 | 201,179,383 | -2.84% | -8.79% |

**Opt 1** is the GBC "recomputing row*cols per access" pattern, and it is
present here verbatim: the V mixdown looped depth-outer/survivor-inner, so
`s_selix[s] * EG_EMBED` (LSL.L #6 + two LEAs, ~90 cycles) was recomputed for
every (s,d) pair to read ONE byte. Reordering to survivor-outer computes the
row pointer once and walks it with a post-increment.

**Opt 2**: GCC will not unroll a loop whose trip count it cannot see, and
`na`/`ns` come out of the blob at runtime. 4x unroll amortises the
CMP.L+Bcc (16 cycles) over 4 weights.

**Opt 3**: all requant multipliers M in the shipped blob are < 128, but
`acc * (int32_t)M` is a 32x32 expression, so GCC calls __mulsi3 (~280 cyc)
once per output row, 1152 rows/pass. Two MULU.W give the identical low 32
bits (two's complement) for ~135.

Where the remaining 201M goes (static estimate, for targeting only):
 - matvec index-stream inner loop  ~116M  (58%)  <- 2.52M nonzero weights
 - tied-embedding logits argmax     ~26M  (13%)
 - attention scores                 ~13M   (6%)
 - V mixdown                        ~11M   (6%)
 - matvec per-row overhead          ~11M   (5%)

| 4 | matvec 8x unroll instead of 4x | 197,932,511 | -1.61% | -10.27% |
| 5 | **W16 index streams** (blob format: pre-doubled BE u16 byte offsets) | 178,030,617 | -10.05% | -19.29% |
| 6 | `(int16_t)` cast on the W16 index -> (An,Dn.W) index mode | 173,080,737 | -2.78% | -21.53% |
| 7 | hand-written 4x-unrolled inner loop + DBRA | 158,059,944 | -8.68% | -28.34% |

**Opt 5 — the biggest single win, and it is a ROM-layout change, not a code
change.** The byte-index stream forced four instructions to turn one byte
into an address:
```
moveq #0,d0 ; move.b (a2)+,d0 ; add.l d0,d0 ; move.w (a5,d0.l),a1
```
Three of them exist only because the index is a byte that must be
zero-extended and doubled. Storing the ALREADY-DOUBLED byte offset as a
big-endian u16 (rows are even-aligned by construction, so MOVE.W is legal)
makes it:
```
move.w (a2)+,d0 ; move.w (a5,d0.w),a1
```
It also halves the cart-ROM *bus* traffic: the Mega Drive cart is 16 bits
wide, so a byte read costs a whole word fetch anyway — the old format was
throwing away half of every fetch.
Converter: `host/sgtm_to_w16.py`. Flags bit3 (0x08) marks it; the engine
still reads byte-streams and 2-bit-packed blobs, so old ROMs keep working.
Cost: model blob 589,474 -> 1,120,624 bytes (+90%).

**Opt 7**: GCC's best was `move.w (a5),a3 / move.w (a0,a3.l),d0 / ext.l d0 /
add.l a1,d0` = 34 cycles/weight plus 6 for ADDQ/CMP/Bcc. Loading the offset
into a DATA register lets `(An,Dn.W)` read straight into an ADDRESS register
(sign-extends for free, no EXT.L) = 30, and DBRA replaces the 3-instruction
loop tail. The host build keeps a plain-C fallback, so both targets run the
same arithmetic.

Running total: **220,579,814 -> 158,059,944 cycles = 1.396x**, output
byte-identical on all 24 tokens at every step.

### [9] HARD RULE 3 — the REAL ROM, not just the bench ROM
`res/resources.res` switched to the W16 blob and the full game ROM rebuilt:
  before: 1,835,008 bytes   after: 2,359,296 bytes (flat cart limit is 4 MB)
End-to-end check on the REAL ROM in MAME (not the bench): boot, press P1 A
via the emulated pad, sample `lastTok` as `genCount` advances:
```
Call me Sophia Elya, your guide.
```
first 24 tokens == the gate exactly. `elya.ok == 1`, so eg_init accepted the
new blob on hardware-representative code, and the shipped ROM still answers.

| 8 | logits: 4x unroll + int8 in the MULS.W *source* operand | 157,249,092 | -0.51% | -28.71% |
| 9 | V-mixdown divide via DIVS.W + SVS instead of __divsi3 | 154,635,260 | -1.66% | -29.90% |
| 10 | W16 row headers read with aligned MOVE.W | 149,551,163 | -3.29% | -32.20% |
| 11 | requant fast path: single MULS.W when acc fits int16 | 148,714,347 | -0.56% | -32.58% |

Opt 8 was the honest disappointment of the set: it looked like ~3% on paper
(96x64 multiplies per token, plus a cheaper MULS source operand) and measured
0.51%. Opt 10 was the opposite - estimated ~1.2%, measured 3.29%. This is
exactly why static reasoning does not get to be the instrument.

**TOTAL: 220,579,814 -> 148,714,347 cycles = 1.483x**, output byte-identical
on all 24 tokens at every one of the 11 steps.
 - 5,804,732 -> 3,913,535 cycles/token
 - 1.32 -> 1.96 tokens/s on a real NTSC Genesis

### [10] A real sampling profiler (added to the instrument)
`mamework/profile.py` + `profile.lua`: one 68000 PC sample per emulated frame,
bucketed against `out/symbol.txt`. It stops sampling as soon as bench_status
flips, because the ROM's idle `for(;;)` otherwise swamps the histogram (first
attempt reported 70% in one 32-byte block — that block was the park loop).

Profile at the 148.7M-cycle point (1162 samples inside the measured region):
```
 49.15%  eg_matvec (out-of-line)
 34.74%  eg_layer          <- mostly an LTO-INLINED copy of the same matvec
 13.79%  eg_next_token     <- tied-embedding logits argmax
  2.24%  eg_rmsnorm
```
So matvec is ~68% once the inlined copy is counted, logits ~14%, real
attention work only ~7%. My static estimate had said attention was 12% and
matvec 52% — the profiler moved both.

### [11] Later steps
| 12 | 0..3 tail of the W16 run also as a DBRA loop (GCC had built a CMPI/Bcc ladder) | 144,556,919 | -2.80% | -34.46% |
| 13 | drop the dead EXT.L in `mul_ss` (MULS.W overwrites all 32 bits anyway) | 142,972,761 | -1.10% | -35.18% |
| 14 | logits loop with post-increment pointers | 142,971,697 | **-0.0007%** | REVERTED |

Step 14 is recorded because it is the honest outcome: it looked like a clean
4-cycles-per-element win on the 68000 addressing modes and measured 1,064
cycles out of 143 MILLION. GCC had already produced equivalent code. Reverted
so the diff stays honest.

### [12] FINAL
```
baseline   220,579,814 cycles
final      142,972,761 cycles      1.543x       (3/3 runs identical)
```
 - 5,804,732 -> 3,762,441 cycles per token
 - 45.4 -> 29.8 frames per token on a real NTSC Genesis
 - 1.32 -> 1.96 tokens/s
 - tokens byte-identical at EVERY step: "Call me Sophia Elya, you"

Clean A/B of the ROM LAYOUT alone (identical engine binary, identical
weights, two ROMs, only the index-stream layout differs):
```
byte index streams  192,177,580 cycles
W16 index streams   142,972,761 cycles      1.344x from the format alone
```
That legacy run is also the backward-compatibility check: the new engine
still reads the old blob on real 68000 and produces the same 24 tokens.

REAL ROM (not the bench): rebuilt at 2,359,296 bytes, booted in MAME, P1 A
pressed on the emulated pad, generated
`Call me Sophia Elya, your guide.` — first 24 tokens == the gate.

### Artifacts
```
mamework/bench_run.py      cycle-accurate runner (MAME + write taps)
mamework/profile.py        sampling profiler
mamework/profile.lua
bench/                     headless bench ROM project (symlinks the real engine)
bench_legacy/              same, with the byte-index blob, for the A/B
genesis/host/sgtm_to_w16.py  blob converter
genesis/res/elya_brain_w16.bin  converted model
host_bench.c               x86 mirror of the bench loop (token oracle)
```
