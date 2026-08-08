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
