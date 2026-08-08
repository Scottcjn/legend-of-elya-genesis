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
