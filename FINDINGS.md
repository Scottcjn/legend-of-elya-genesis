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
