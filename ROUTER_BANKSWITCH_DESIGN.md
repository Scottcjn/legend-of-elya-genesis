# Elya-Genesis — Router + Bank-Switch Design (scaling 8 → N experts)

## The core principle
Sparse activation decouples **capacity** from **runtime cost**:
- Capacity scales with **ROM** (cartridge size): N experts × ~71 KB each.
- Runtime cost stays **constant**: always exactly `shared (21.9 KB) + 1 expert (~28 KB active-ternary)` regardless of N.
- The 68000 never runs more than one expert per token, so a 32 MB cart of 400 experts costs the same per-token as today's 8.

Today: 8 experts, top-1 routing, 114 K active params. This design grows N without growing the active footprint.

---

## 1. Memory map

**68000 RAM (`$FF0000–$FFFFFF`, 64 KB total):**
| Region | Size | Contents |
|---|---|---|
| shared block | ~22 KB | shared embedding + output projection + router weights (resident, never swapped) |
| active expert | ~28 KB | the currently-routed expert, ternary-packed (2 bits/weight) |
| scratch | ~10 KB | token buffer, hidden state, accumulator, router logits, gen state |
| stack/system | ~4 KB | 68000 stack, vectors |

The whole *active* model lives in RAM. Everything else lives in ROM until routed to.

**ROM layout:**
| Region | Contents |
|---|---|
| `$000000–$0000FF` | Genesis header (`rom_head.bin`) |
| low bank | engine (`elya_gpt`), router gate, tokenizer, shared block (mastered here, copied to RAM at boot), **bank-switch table** |
| expert region | N expert blobs, each ~71 KB, laid out contiguously |

---

## 2. The router

**Gate (resident in RAM):** a tiny linear projection `hidden_dim → N_logits`, argmax = `expert_id`. It's the only part that grows with N, and it grows linearly (one row per expert). Keep it int8; softmax via LUT (no FPU).

**Flat router works to ~58 experts.** Past that an N-way argmax gets mushy, so:

**Hierarchical router (for N > ~32) — the key to scaling honestly:**
- Level 1: route to 1 of **G domain-groups** (G-way, small & sharp).
- Level 2: route to 1 of **E experts within that group** (E-way).
- N = G × E. A 64-expert model = 8 groups × 8 experts = two 8-way decisions, each as sharp as today's single 8-way. A 512-expert model = 8×8×8, three cheap decisions.
- Each decision is `log_k(N)` small softmaxes instead of one giant one → the router stays confident as N explodes.

**Cheapest honest variant — centroid routing (recommended on the 68000):**
- Give each expert a small **domain centroid** vector (e.g. 32 int8 dims), stored in the resident table.
- Route = nearest centroid to the pooled prompt embedding (cosine ≈ dot product on normalized int8 → just adds, no matmul).
- This is O(N) cheap dot-products, degrades gracefully, and is trivially interpretable ("prompt landed nearest the `rustchain` centroid"). Combine with the hierarchical groups for large N.

---

## 3. Bank-switch table + mechanism

**Table (resident, low ROM):** `expert_id → (bank, offset)`. One entry per expert.

**Case A — ≤ 58 experts, flat 4 MB, NO mapper (simplest, ship this first):**
- All experts fit in the flat address space (`58 × 71 KB ≈ 4.1 MB`).
- No bank register. Address is pure arithmetic: `expert_addr = EXPERT_BASE + id * EXPERT_STRIDE`.
- Router picks id → copy `~28 KB` from `expert_addr` into the RAM active-expert region → run. Done.

**Case B — > 58 experts, mapper (SSF2-style or custom flash):**
- Reserve a fixed ROM **window** (e.g. `$300000–$3FFFFF`, 1 MB) as the "expert window."
- A mapper register selects which physical bank maps into the window.
- Route: `write MAPPER_REG = table[id].bank` → expert appears at the window → copy `~28 KB` to RAM → run.
- SSF2 mapper gives up to 8 × 512 KB banks; a custom flash mapper can go far higher.

**Caching (important):** the router switches ~34% of the time today, so **don't reload ROM→RAM every token.** Keep `current_expert_id`; only re-copy when the routed id changes. Stable routing → near-zero switch cost.

---

## 4. Per-token forward pass (pseudocode)

```
loop each generated token:
    h  = embed(token)                          # shared, resident
    # --- route ---
    gid = argmax(group_gate  · h)              # level-1 (or skip if flat)
    eid = argmax(expert_gate[gid] · h)         # level-2  → global expert_id
    if eid != current_expert_id:
        addr = table[eid]                      # flat arith, or set MAPPER_REG
        copy_28KB(ROM[addr] -> RAM.active_expert)
        current_expert_id = eid
    # --- run the one active expert (ternary) ---
    h  = expert_forward(RAM.active_expert, h)  # signed adds via LUT, no mul
    logits = out_proj · h                       # shared, resident
    token  = sample(logits)                     # LUT softmax + PRNG
```

Only step `copy_28KB` touches ROM, and only on a route change.

---

## 5. Ternary math — why this fits a 68000 at all

- Weights are ternary `{-1,0,+1}`, packed 2 bits/weight → 114 K weights ≈ 28 KB (that's why the active expert fits RAM).
- Matmul with ternary weights = **signed accumulation, no multiply:** for each weight, `acc += a` / `acc -= a` / skip. The 68000's `MULS` is ~70 cycles; `ADD` is ~8 — ternary turns every MAC into a cheap add. This is the enabling trick, not an optimization.
- Activations int8/int16 with a per-block scale; nonlinearities (softmax, the gate) via 256-entry LUTs.

---

## 6. Scaling path (concrete)

| Step | N | ROM | Router | Change from today |
|---|---|---|---|---|
| now | 8 | 1.3 MB | flat 8-way | — |
| **v1** | ~58 | 4 MB flat | flat argmax or centroid | grow gate to N; flat address table; NO mapper |
| **v2** | 64–256 | 8–32 MB | hierarchical G×E (+ centroids) | add mapper + bank register; 2-level route |
| **v3** | 256+ | 32 MB+ | 3-level + load-balance aux loss | custom flash mapper; expert-diversity regularizer |

Runtime per token is **identical** across all four — one shared + one expert.

---

## 7. Keeping the router honest as N grows (the real risk)

1. **N-way mush** → hierarchical routing (§2): every decision stays K-way and sharp.
2. **Dead / overloaded experts** → the load-balance aux loss you already log (`router%`); target near-uniform group usage, per-expert usage within tolerance.
3. **Domain collision** (two experts confusable) → train each expert on **disjoint domain data**, and add a **centroid-separation** regularizer so expert domain vectors stay far apart (also makes centroid routing sharper).
4. **Interpretability shield** → because routing is a nearest-centroid pick, every generation logs *which expert answered and why* ("nearest to `hardware` centroid, margin 0.31") — the same "prove the content" instinct as the cert work, applied to inference.

---

## 8. The bigger claim this proves
The identical routing doctrine runs from a **512 GB POWER8** (RAM-Coffers: route queries to NUMA-resident domain banks) down to a **64 KB Sega Genesis** (route tokens to ROM-resident domain experts). Same architecture, 7 orders of magnitude of substrate. That's the demo.

---

## 9. First buildable step
Extend `train/train_elya_moe.py` from 8 → 24 experts (still flat, no mapper needed), add centroid vectors + a load-balance term, and grow the bank table in `src/elya_gpt` to `EXPERT_BASE + id*STRIDE`. If 24 stays coherent and the router stays sharp, the path to 58 (flat) then hundreds (mapper) is mechanical.
