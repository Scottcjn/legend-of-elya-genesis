# Segmented speech — an old-school trick for an old-school machine

Scott's idea: if the model can only stay coherent for ~12-28 characters,
**stop fighting it.** Generate a short segment, leave it on screen,
re-prime, and continue. The player sees a complete, coherent answer built
out of pieces that each stay inside the model's good zone.

This is exactly how 8- and 16-bit games streamed text they could not hold
in RAM — page it, show it, load the next page. Here the constraint is not
RAM, it is *coherence distance*, but the fix has the same shape.

## Why it works at all

The measured failure is **positional**, not random: answers start correct
and degrade after 13-28 characters (see LOCKON_MOE.md). Quality is a
function of distance from the prompt. So any scheme that keeps generation
*near* a prompt keeps quality high. Segmenting resets the distance.

## Three variants, cheapest first

### V1 — Position reset only (nearly free)

With absolute positional encoding, "distance from the prompt" is literally
the position index fed to the PE table. Reset **just the position counter**
every N tokens, keeping the KV cache intact:

```c
if (++seg_len >= SEG_TOKENS) { seg_len = 0; st->pos_pe = 0; }
```

Context is fully preserved (the KV cache is untouched); only the model's
sense of "how far in am I" restarts. **Cost: zero extra forward passes.**

⚠️ Risk: two cached entries then share a positional embedding, which may
confuse attention. Must be measured, not assumed. This is the variant to
test first precisely because it is free.

### V2 — Segment + re-prime with a short cue (the honest version)

Generate N tokens, display them, reset the KV cache, then re-prime with
the question plus a **short tail** of what was already said:

```
segment 1 prompt: "What is epoch?: "
segment 2 prompt: "What is epoch?: ...settle miner rewards"   <- last ~16 chars
```

Each segment starts within a few characters of a prompt, so every segment
runs in the good zone. Semantic continuity comes from the tail; positional
freshness comes from the reset.

**Cost**: re-priming re-runs the forward pass over the cue. With a bounded
tail (~16 chars) and 12-token segments the overhead is roughly 2x, not the
4x+ that re-priming the entire history would cost. At ~0.65 tok/s that is
real and must be shown on the tok/s counter honestly.

### V3 — Multi-answer stitching (free, uses what the corpus already has)

The corpus already stores **several complete short answers per question**
(e.g. "What is RustChain?" has four). Each is a self-contained sentence
well inside the coherence budget. Generate one per segment with a
different sampling seed and display them in sequence:

> Elyan Labs blockchain for vintage chips.
> Nodes attest real vintage hardware.
> Old silicon earns extra RTC rewards.

Coherent by construction, zero extra machinery, and it reads as her
elaborating rather than as a workaround. Weakest as *research*, strongest
as *product*.

## Presentation — the part that makes it feel intentional

Do not hide the seams; make them the character. She is a dream-being
thinking out loud on a 7.67 MHz CPU:

- Keep completed segments on screen; never clear mid-answer.
- Between segments, a brief pause with the mouth closed and the gem
  palette-cycling faster — she is *considering*, and the Dreamscape keeps
  moving because the animation runs from the VBlank callback.
- The tok/s counter keeps running across the pause, so the number stays
  honest rather than being reset to flatter it.

## Decision rule

Run the A/B first. If positional encoding lifts mean exact prefix well
past ~12 characters, segmentation becomes **optional polish** (V3 for
flavour). If it does not, segmentation is the **shipping path**, and V1
should be measured before V2 because it is free.

Either way V1 is worth measuring: a technique that costs nothing and
might extend coherence is worth an experiment regardless of whether we
need it.
