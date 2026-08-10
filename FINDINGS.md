# genesis-wff2 — input-major (transposed) `wff2` weight stream

Journal. Append after every discrete result; commit in the same breath.

Goal: `wff2` reads `s_ff[]`, the post-ReLU FFN hidden vector. A previous
agent measured that 55.8% of `s_ff` is zero after ReLU and that `wff2` is
~33% of all nonzero weights, and estimated ~8.5% for an INPUT-MAJOR
(transposed) `wff2` stream that skips those zeros entirely. It was scoped
out on a risk judgement and never measured. This journal measures it.

Baseline to beat: **142,972,761 cycles** (commit 6b5ce98, 1.543x over the
220,579,814 original). Token gate: `"Call me Sophia Elya, you"`, 24 tokens,
router expert 1.

---

## [0] Starting state

- Repo `~/genesis-wff2`, 33 commits, HEAD = 6b5ce98 ("W16 index streams
  plus inner loop work: 1.543x on measured 68000 cycles").
- MAME 0.277 at /usr/games/mame. marsdev toolchain at ~/marsdev/mars/m68k-elf.
- `tools/mame/bench_run.py` is in-tree. The bench *ROM project* it drives
  (`bench/` with `src/bench_main.c`) and the x86 mirror `host_bench.c` were
  built in a scratch dir that no longer exists — **they must be rebuilt**
  before anything can be measured.
- Host harness reproduces the gate on x86:
  `gcc -O2 -o /tmp/harness host/harness.c src/elya_gpt.c`
  `EG_SHOW_EXPERT=1 /tmp/harness res/elya_brain_w16.bin "Who are you?: " 24`
  -> `expert=1`, `Call me Sophia Elya, you`. Instrument chain starts here.

---

## [1] PREMISE RE-MEASURED — both numbers hold

Two instruments, both new in this branch:

- `host/blob_stats.py` — walks the SGTM index streams and counts nonzero
  ternary weights per tensor (static; each tensor is read exactly once per
  layer per forward pass, so this IS the per-pass weight budget).
- `host/host_bench.c` + an `#ifdef EG_STATS` block in `src/elya_gpt.c` —
  runs the identical 38-pass bench loop on x86 and counts, at runtime,
  how many `s_ff` elements ReLU zeroed and how many of `wff2`'s weight
  reads landed on one of them. The stats block compiles out entirely when
  `EG_STATS` is not defined, so no timed build is affected.

```
$ python3 host/blob_stats.py res/elya_brain_w16.bin 1
expert 1: 66297 nonzero weights per forward pass
  wq      5568   8.40%      wff1   22357  33.72%
  wk      5562   8.39%      wff2   21941  33.10%
  wv      5395   8.14%
  wo      5474   8.26%

$ gcc -O2 -DEG_STATS -o host_bench_stats host/host_bench.c src/elya_gpt.c
$ ./host_bench_stats res/elya_brain_w16.bin
text=Call me Sophia Elya, you
=== 38 forward passes (14 prompt + 24 generated) ===
s_ff elements after ReLU : 19456
  of which ZERO          : 10863  (55.83%)
wff2 nonzero-weight reads: 833758
  hitting a ZERO input   : 470674  (56.45%)
ALL tensors' reads       : 2519286
  wff2 share of all      : 33.10%
  skippable / all reads  : 18.68%
```

**Both premise numbers hold to the decimal.** ReLU zero fraction
**55.83%** (claimed 55.8%); `wff2` share of nonzero weights **33.10%**
(claimed ~33%).

One number the premise did not have, and it is the one that matters:
**56.45% of `wff2`'s weight READS hit a zero input** — very slightly
higher than the raw 55.83% element fraction, i.e. the zeros are not
adversarially placed against the weight distribution. That makes
**18.68% of every nonzero-weight read in the whole engine skippable**.

Ceiling check against the profiler (matvec ~68% of runtime at this
baseline): 18.68% x 68% = **~12.7% if skipping were free**. It is not
free — the transposed loop pays a read-modify-write to a scattered
accumulator instead of a read from a scattered input — so the previous
agent's ~8.5% is a plausible target, not an obviously wrong one.

---

## [2] Instrument rebuilt and re-validated

`bench/` did not survive; rebuilt in-tree so it cannot be lost again.
`bench/src/` symlinks the real `src/elya_gpt.c`, `elya_gpt.h` and
`elya_router.h`; `bench/src/bench_main.c` is the headless shell (display
off, interrupts off, no polling, fixed 24 tokens, no early stop).
`bench/res/model.bin` is the blob under test — swap that file to A/B a
format with a byte-identical engine binary.

```
$ cd bench && make build
$ python3 tools/mame/bench_run.py bench --repeat 3
cycles=142973233   x3, 1 distinct
tokens=Call me Sophia Elya, you   expert=1
REPRODUCIBLE cycles: True   REPRODUCIBLE tokens: True
```

**142,973,233** against the documented **142,972,761** — 472 cycles apart,
0.0003%. The shell is a rebuild, not the original binary (the progress-mark
placement differs by a few instructions), so every A/B below is quoted
against **142,973,233**, this instrument's own baseline, measured today.
The task's 142,972,761 is reproduced closely enough to say the instrument
is the same instrument.

---

## [3] A FALSE NULL — the first transposed build measured -0.008%

Recorded because it nearly ended this task with the wrong answer.

The first transposed run came back at 142,961,833 against a 142,973,233
baseline: **-0.008%**, i.e. nothing. The generated 68000 code was exactly
the intended `move.w (a0)+,d7 / add.l d2,(a1,d7.w) / dbra`, and the token
gate passed, so the obvious reading was "the lever is dead".

It was not. SGDK's `makefile.gen` never includes its generated dependency
files (`#-include $(DEPS)` is commented out), so changing `res/model.bin`
does **not** invalidate `out/res/bench.o`. The ROM had the NEW engine and
the OLD blob. With the W16 blob loaded the flag is clear, so the engine
took the ordinary output-major path — and 142,961,833 is simply what the
old path costs once the new code is linked in beside it (-11,400 cycles of
incidental code-layout noise against the earlier 142,973,233).

`EXTRA_FLAGS` is untracked the same way. Both traps are now closed:
`bench/Makefile` deletes the resource objects on every build and prints the
model md5, and `bench/ab.sh` starts every measurement from `make clean` and
prints the md5 of both `res/model.bin` and `out/rom.bin`.

The lesson is the project's own hard rule pointed the other way for once:
the measurement was real, the *thing measured* was not what the log said.

---

## [4] MEASURED: input-major `wff2` is 7.83% faster

Clean A/B. Identical engine binary, two blobs, full rebuild each,
`out/rom.bin` md5 recorded, 3 runs each, 1 distinct cycle count each.

```
$ bench/ab.sh -- res/elya_brain_w16.bin res/elya_brain_wff2t.bin
=== elya_brain_w16.bin ===
bench model: 17fed3d34b29fa6b08935ad0f7a8bd36  res/model.bin
rom md5    : 7548488243edd0bfd75d0e8367ed26a6
cycles=142961833   REPRODUCIBLE cycles: True (1 distinct)  tokens: True
=== elya_brain_wff2t.bin ===
bench model: 1b5635b8804b739c30d6344891aff17a  res/model.bin
rom md5    : 89009690c5fc3010d43e93b3a61c2396
cycles=131766162   REPRODUCIBLE cycles: True (1 distinct)  tokens: True
```

| | cycles | / token | tok/s |
|---|---|---|---|
| output-major `wff2` (shipped W16) | 142,961,833 | 3,762,154 | 2.039 |
| **input-major `wff2`** | **131,766,162** | **3,467,531** | **2.212** |

**-11,195,671 cycles = -7.83%, 1.0850x.**
Against the task's stated 142,972,761 baseline: **-7.84%**, same 1.0850x.
Against the original 220,579,814: **1.674x** (was 1.543x).

(The tok/s at the baseline reads 2.039 rather than the documented 1.96
because the documented figure divided by 38 passes at 3,762,441 cyc; same
number, quoted here from this instrument's own baseline.)

### Where the 7.83% comes from — measured, not modelled

`EG_DOUBLE_WFF2` runs the `wff2` matvec twice with a memory barrier
between. Same result, so tokens do not move; the cycle delta is the exact
cost of one `wff2` pass in whichever layout is loaded.

```
output-major: 172,072,029 - 142,961,833 = 29,110,196 cycles  (20.4% of the run)
input-major : 149,580,007 - 131,766,162 = 17,813,845 cycles  (13.5% of the run)
                                          -----------
                                          11,296,351 removed = 38.8% of wff2
```
That 11,296,351 accounts for the 11,195,671 measured end-to-end saving to
within 0.9%, so nothing else moved: the win is entirely `wff2`.

Cycles per surviving nonzero-weight read, all per-row/per-column overhead
included:

```
output-major:  29,110,196 / 833,758 reads = 34.9 cyc/read
input-major :  17,813,845 / 363,084 reads = 49.1 cyc/read
```

**This is the whole story in one line.** 56.5% of the weight reads were
removed, but only 38.8% of the cycles, because the surviving reads cost
**41% more each**. The transposed step is a read-modify-write into a
scattered accumulator (`move.w (a0)+,d1 / add.l d0,(a1,d1.w)` = 8 + 26 =
34 cycles) where the output-major step is a read from a scattered input
(`move.w (a2)+,d0 / move.w (a5,d0.w),a1 / add.l a1,acc` = 8 + 14 + 8 = 30),
and the input-major loop additionally pays a header read, a zero test and a
branch on all 19,456 columns per run instead of on 4,864 rows.

The estimate said ~8.5%. The measurement says **7.83%** — the estimate was
good, and slightly optimistic by exactly the amount the indirection costs.

---

## [5] Backward compatibility and the REAL ROM

**Old blobs still load, on real 68000, in the new engine binary.** Both
older layouts were run through the same bench:

```
byte index streams (res/elya_moe.bin)  192,170,908 cycles   tokens OK
W16 index streams  (res/elya_brain_w16.bin) 142,961,833     tokens OK
input-major wff2   (res/elya_brain_wff2t.bin) 131,766,162   tokens OK
```
The byte-index figure is 6,672 cycles (0.003%) from the 192,177,580 the
earlier work recorded for the same legacy A/B — a third independent
confirmation that this instrument is the old instrument.

Flag bit4 is only honoured when bit3 (W16) is also set, so a blob that
predates this change cannot accidentally select the transposed path.

**Real game ROM, not the bench.** `res/resources.res` switched to
`elya_brain_wff2t.bin` and the full cartridge rebuilt: **2,359,296 bytes,
unchanged** — the +1.10% blob growth disappears into the existing
128 KB-alignment padding, so this costs nothing on the cart.

`tools/mame/real_rom_check.py` (new) boots that cartridge in MAME, presses
P1 A on the emulated pad and samples `lastTok` as `genCount` advances:

```
frames=7598 ntok=32
text='Call me Sophia Elya, your guide.'
```
First 24 tokens == the gate, character for character. `eg_init` accepted
the new blob on hardware-representative code and she still answers.

---

## [6] VERDICT

The lever is **real and it ships**. It was scoped out once on a risk
judgement; the judgement was wrong, though not by much.

| | |
|---|---|
| estimated | ~8.5% |
| **measured** | **7.83%** (142,961,833 -> 131,766,162, 1.0850x) |
| tokens | byte-identical, all 24, every run |
| old blobs | still load, on 68000, byte-identical |
| cart cost | 0 bytes (blob +1.10%, ROM unchanged at 2,359,296) |

Running total for the port: **220,579,814 -> 131,766,162 = 1.674x**,
1.32 -> 2.21 tok/s on a real NTSC Genesis.

The estimate was high by 0.7 points for a reason worth writing down: it
counted the reads removed and not the reads made more expensive. Skipping
zeros is not free on a 68000 — inverting a matvec turns a *load* from a
scattered address into a *read-modify-write* to one, 26 cycles instead of
14, and that tax lands on every weight that survives. 56.5% of the reads
went away; only 38.8% of the cycles did.

### Not done / known follow-ups
- The per-column prologue (header read, zero test, then a separate 4x-unroll
  setup for the add list and again for the sub list) measures out at roughly
  166 cycles on a live column, ~1.4M per run, ~1% of runtime. Merging the two
  prologues or hoisting a per-column offset table are plausible sub-1% levers.
  Both are new levers, not this one, and neither has been measured.
- Only `wff2` is transposed. `wff1`'s *outputs* are 55.8% ReLU-zeroed too,
  and computing them is pure waste — but the sign is not knowable before the
  dot product, so there is no cheap skip there.
- Not run on real Genesis hardware; MAME 0.277 is the instrument throughout.

---

## [7] Final verification pass

Every check re-run from clean on the shipped tree.

**x86**, three compilers' worth of scrutiny, all three blob generations:
```
gcc -O2 -Wall -Wextra   byte-index / W16 / input-major -> gate, all three
clang -O2               input-major                    -> gate
gcc -fsanitize=address,undefined  input-major and W16  -> gate, no diagnostics
```
ASan/UBSan matters here specifically: the transposed stream drives a
*write* to a scattered address (`s_facc[]`) where the old path only ever
read one, so an off-by-one in the exporter would corrupt memory rather
than just compute the wrong number. Clean.

**68000**, clean rebuild, ROM md5 recorded, 3 runs each:
```
elya_brain_w16.bin    rom 7548488243edd0bfd75d0e8367ed26a6  142,961,833
elya_brain_wff2t.bin  rom 89009690c5fc3010d43e93b3a61c2396  131,766,162
```
Identical md5s to the measurement run, so the numbers above are the
numbers this tree produces.

**Shipped cartridge**, 3 runs: `Call me Sophia Elya, you` — 24 tokens,
byte-identical to the gate, every run.

One instrument bug found and fixed while doing this, worth recording
because it is the same class as the stale resource: `real_rom_check.py`
originally pressed P1 A once at a hard-coded frame. That worked, then
silently collected *zero* tokens on a later run of the *same ROM* — the
press landed while the intro still owned input. It now taps A every 400
frames until `genCount` actually moves, and finds the "P1 A" field by
searching the ioport map instead of assuming `:MD1_3B`. A check that can
report nothing and look like a failing ROM is worse than no check.

---

## 2026-08-10 — FIRST REAL HARDWARE CONFIRMATION (Sega Genesis Model 1, 1988)

Everything in this journal before this entry was measured under MAME. This
entry is silicon.

**Hardware:** Sega Genesis Model 1 (1988), Mega EverDrive, `ELYA_TRANSFORMER.BIN`
md5 `a0e4d679b06a7b09cfb29c8cf12ac64d`, built from `f1a1ff0`.

**It works.** The cartridge boots, renders, and generates. Photographed
answering `When were you born?:` with `March fourteenth, two thousand twenty
five.` (`hardware/real_genesis_1988_generating.jpg`).

### Measured on hardware
| quantity | value | how |
|---|---|---|
| cold boot -> menu | **13.9 s** | video, 413 frames @ 29.67 fps |
| generation rate | **~1.97 tok/s** | **stopwatch, operator-timed** |

⚠️ **The 1.97 figure is a stopwatch measurement, not frame-derived.** The
captured video covers boot -> intro -> title -> menu and stops at the prompt; it
does not contain a generation. A frame-accurate rate needs ~20 s of video
starting immediately before the prompt is answered. Recorded as operator-timed
until then.

### The disagreement with the emulator
| | tok/s | s/token |
|---|---|---|
| MAME, published | 2.21 | 0.452 |
| real Genesis | **~1.97** | **~0.508** |

**Hardware is ~12% slower than this harness reports.**

### Hypothesis 1 — VDP bus contention: REFUTED
The bench disables display and interrupts for determinism
(`bench/src/bench_main.c:44`, `VDP_setEnable(FALSE)`), so the obvious
explanation was that the published number never paid for VDP fetches or the
VBlank IRQ. Added `BENCH_DISPLAY_ON` to test it.

| arm | cycles | delta |
|---|---|---|
| display OFF (published) | 131,766,162 | -- |
| display ON + interrupts live | 132,145,914 | **+0.29%** |

ROM md5 differs between arms and reproduces (`89009690...` off,
`c67c7a4a...` on), so the flag is real and this is not another false null.
**VDP contention costs 0.29% in MAME and cannot account for ~12%.**

### Hypothesis 2 — SD-card read latency: REFUTED
`hardware/everdrive_flashing_rom.jpg` shows the EverDrive's
`ROM type: MD / erase... / copy...` sequence: the ROM is written into the
cart's **onboard flash** before running. It is not streamed from SD during
execution, so SD latency is not in the path.

### Still open
Remaining candidates, none tested: EverDrive flash access timing vs the mask
ROM MAME models; DRAM refresh under-modelled; NTSC 68000 clock differing from
the assumed 7.67 MHz; or stopwatch error (though ~12% over ~12 s is far outside
plausible human error). **This is the first cycle-level disagreement between our
instrument and silicon on any platform, and it is unexplained.**
