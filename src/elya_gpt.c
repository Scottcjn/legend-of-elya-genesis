/* SPDX-License-Identifier: Apache-2.0
 * elya_gpt.c — integer-only transformer inference for the Sega Genesis.
 * See elya_gpt.h for format. Mirrors train/train_elya_genesis.py QAT
 * semantics: any rounding done here was fake-quantized during training.
 */
#include "elya_gpt.h"      /* pulls genesis.h (SGDK) or stdint.h (host)   */
#ifndef SGDK_GCC
#include <string.h>
#endif

/* ---- fixed-point helpers ---------------------------------------------- */

static uint16_t rd16be(const uint8_t *p) {
    return (uint16_t)((p[0] << 8) | p[1]);
}

/* 16x16 -> 32 signed multiply.
 * The 68000 HAS MULS.W (70 cycles) but no 32x32 multiply, so any C
 * expression GCC believes is 32-bit becomes a __mulsi3 call: three
 * MULUs plus call overhead, ~250 cycles. GCC will not always narrow a
 * value it has tracked through a 32-bit division, so we say it
 * outright. The host path is plain C, keeping both builds identical. */
static inline int32_t mul_ss(int16_t a, int16_t b) {
#if defined(__m68k__) && !defined(__mcoldfire__)
    int32_t r = a;
    __asm__ ("muls.w %1,%0" : "+d"(r) : "dm"(b));
    return r;
#else
    return (int32_t)a * (int32_t)b;
#endif
}

static int16_t sat16(int32_t v) {
    if (v > 32767)  return 32767;
    if (v < -32768) return -32768;
    return (int16_t)v;
}

/* integer sqrt of a uint32, standard bit-by-bit */
static uint16_t isqrt32(uint32_t v) {
    uint32_t res = 0, bit = 1UL << 30;
    while (bit > v) bit >>= 2;
    while (bit) {
        if (v >= res + bit) { v -= res + bit; res = (res >> 1) + bit; }
        else res >>= 1;
        bit >>= 2;
    }
    return (uint16_t)res;
}

/* ---- ternary matvec: out = requant(W * in) ----------------------------
 * in: int16 Q3.12   out: int16 Q3.12   acc: int32 (safe: 256*32767*127 < 2^31)
 * Inner loop is add/sub only — this is the reason the Genesis can think. */
/* Layout is chosen at RUNTIME from the blob's flags bit1 so that one
 * engine binary reads both index-stream and 2-bit-packed weights. That is
 * what makes an honest 68000 A/B possible: two ROMs, same code, same
 * weights, only the layout differs. */
static uint8_t eg_streams = 1;

static void eg_matvec(const EgTensor *t, const int16_t *in, int16_t *out,
                      int16_t in_dim, int16_t out_dim)
{
    uint16_t M = t->M;
    uint8_t  S = t->S;

if (eg_streams) {
    /* T2: each row is (u16 n_add, u16 n_sub, add idx bytes, sub idx bytes).
     * No bit unpacking and no per-weight branch — zero weights are simply
     * absent. Compiles to MOVE.B (A1)+,D0 / ADD.W (A2,D0.W),D2 on 68K. */
    const uint8_t *p = t->packed;
    (void)in_dim;
    for (int16_t o = 0; o < out_dim; o++) {
        uint16_t na = rd16be(p); p += 2;
        uint16_t ns = rd16be(p); p += 2;
        int32_t acc = 0;
        for (uint16_t i = 0; i < na; i++) acc += in[*p++];
        for (uint16_t i = 0; i < ns; i++) acc -= in[*p++];
        out[o] = sat16((acc * (int32_t)M) >> S);
    }
} else {
    /* 2-bit packed, 4 weights per byte */
    const uint8_t *row = t->packed;
    int16_t ib = in_dim >> 2;
    for (int16_t o = 0; o < out_dim; o++) {
        int32_t acc = 0;
        const int16_t *ip = in;
        for (int16_t b = 0; b < ib; b++) {
            uint8_t w = *row++;
            uint8_t c;
            c = (uint8_t)(w >> 6) & 3;
            if (c == 1) acc += ip[0]; else if (c == 2) acc -= ip[0];
            c = (uint8_t)(w >> 4) & 3;
            if (c == 1) acc += ip[1]; else if (c == 2) acc -= ip[1];
            c = (uint8_t)(w >> 2) & 3;
            if (c == 1) acc += ip[2]; else if (c == 2) acc -= ip[2];
            c = w & 3;
            if (c == 1) acc += ip[3]; else if (c == 2) acc -= ip[3];
            ip += 4;
        }
        out[o] = sat16((acc * (int32_t)M) >> S);
    }
}
}

/* ---- RMS norm, parameter-free, all int -------------------------------- */
static void eg_rmsnorm(int16_t *x, int16_t n)
{
    uint32_t ss = 0;
    for (int16_t i = 0; i < n; i++) {
        int16_t h = (int16_t)(x[i] >> 4);          /* Q8 */
        ss += (uint32_t)(h * h);                   /* Q16, MULS.W */
    }
    uint32_t ms = ss / (uint16_t)n;                /* mean square, Q16 */
    uint16_t rms = isqrt32(ms);                    /* Q8 */
    if (rms == 0) rms = 1;
    int32_t r32 = (int32_t)(1UL << 18) / rms;      /* Q10 reciprocal */
    if (r32 > 16383) r32 = 16383;                  /* max gain 16x   */
    /* r fits 16 bits after the clamp: keeping it int16_t lets the 68000
     * use a single MULS.W instead of the __mulsi3 helper (3x MULU +
     * call overhead) on every one of the 320 products per token. */
    int16_t r = (int16_t)r32;
    for (int16_t i = 0; i < n; i++)
        x[i] = sat16(mul_ss(x[i], r) >> 10);
}

/* ---- static scratch (keep the 68K stack small) ------------------------ */
static int16_t s_q[EG_EMBED], s_k[EG_EMBED], s_v[EG_EMBED];
static int16_t s_attn[EG_EMBED], s_proj[EG_EMBED], s_res[EG_EMBED];
static int16_t s_ff[EG_FFN];
static int32_t s_score[EG_CTX];
/* survivor weights are Q14 exp-LUT values (<= 16384) so they fit int16_t;
 * that keeps the V mixdown a single MULS.W per term on 68000 rather than
 * a __mulsi3 call. */
static int16_t s_selsc[EG_CTX];   /* collapse survivors: weights */
static int16_t s_selix[EG_CTX];   /* collapse survivors: indices */

/* ---- one transformer layer --------------------------------------------
 * PSE collapse, 68000 edition (see docs/SPEED_PLAN.md):
 *  - KV is a RING buffer. The model has no positional encoding, so
 *    attention is order-invariant: overwriting the oldest slot is exactly
 *    equivalent to shifting the window, minus a 16KB memmove per token.
 *  - Attention is TOP-K hard: only the EG_TOPK strongest scores survive
 *    to the exp-LUT and the V mixdown. Prune the weak, keep the strong —
 *    the same constraint-bound selection the POWER8 vec_perm build does,
 *    here as a scalar selection pass. */
static void eg_layer(EgState *st, int16_t li, int16_t slot, int16_t n_ctx)
{
    int16_t *x = st->x;

    memcpy(s_res, x, sizeof(s_res));
    eg_rmsnorm(x, EG_EMBED);

    eg_matvec(&st->wq[li], x, s_q, EG_EMBED, EG_EMBED);
    eg_matvec(&st->wk[li], x, s_k, EG_EMBED, EG_EMBED);
    eg_matvec(&st->wv[li], x, s_v, EG_EMBED, EG_EMBED);

    /* store K,V into the ring slot (int8 Q4.4) */
    for (int16_t i = 0; i < EG_EMBED; i++) {
        int16_t k8 = (int16_t)(s_k[i] >> 8);
        int16_t v8 = (int16_t)(s_v[i] >> 8);
        st->kc[li][slot][i] = (int8_t)(k8 > 127 ? 127 : (k8 < -128 ? -128 : k8));
        st->vc[li][slot][i] = (int8_t)(v8 > 127 ? 127 : (v8 < -128 ? -128 : v8));
    }

    /* multi-head attention */
    for (int16_t h = 0; h < EG_HEADS; h++) {
        const int16_t *qh = s_q + h * EG_HD;
        int16_t off = (int16_t)(h * EG_HD);

        /* scores: q(Q12) . k(Q4.4) -> Q16, then * 1/sqrt(hd=16) = >>2 */
        int32_t mx = (int32_t)0x80000000;
        for (int16_t tt = 0; tt < n_ctx; tt++) {
            const int8_t *kh = st->kc[li][tt] + off;
            int32_t sc = 0;
            for (int16_t d = 0; d < EG_HD; d++)
                sc += mul_ss(qh[d], (int16_t)kh[d]);
            sc >>= 2;
            s_score[tt] = sc;
            if (sc > mx) mx = sc;
        }

        /* COLLAPSE (lossless): weight each position from the ROM exp LUT
         * and keep only the survivors — a position whose Q14 weight
         * rounds to 0 contributes exactly nothing to the V mixdown, so
         * dropping it changes no bit of the result. Typically prunes
         * most of the window once the model is confident.
         * EG_TOPK < EG_CTX additionally caps survivors (LOSSY on this
         * model — see docs/SPEED_PLAN.md; default off). */
        int32_t sum = 0;
        int16_t nsel = 0;
        for (int16_t tt = 0; tt < n_ctx; tt++) {
            uint32_t d = (uint32_t)(mx - s_score[tt]) >> 12;  /* 1/16 units */
            /* Bound at the LUT's size, not at a hand-picked index. An
             * earlier version cut at d>155 "because exp(-9.7) rounds to
             * 0", but the exported LUT stays nonzero (=1) through index
             * 166 — so that prune silently dropped 11 real weights and
             * was NOT lossless as documented. Only w == 0 may be pruned. */
            if (d > 255) continue;
            uint16_t w = rd16be(st->explut + (d << 1));
            if (w == 0) continue;
            s_selsc[nsel] = (int16_t)w;            /* Q14, <= 16384 */
            s_selix[nsel] = tt;
            sum += w;
            nsel++;
        }

        /* Optional LOSSY cap: keep the K STRONGEST survivors.
         * The previous code simply broke out of the scan after K, which
         * keeps the first K in ring order — position, not strength. Any
         * "Top-K" measurement taken against that code did not test Top-K. */
        if (EG_TOPK < EG_CTX && nsel > EG_TOPK) {
            for (int16_t i = 1; i < nsel; i++) {       /* insertion sort desc */
                int16_t ws = s_selsc[i], ix = s_selix[i], j = i - 1;
                while (j >= 0 && s_selsc[j] < ws) {
                    s_selsc[j + 1] = s_selsc[j];
                    s_selix[j + 1] = s_selix[j];
                    j--;
                }
                s_selsc[j + 1] = ws; s_selix[j + 1] = ix;
            }
            nsel = EG_TOPK;
            sum = 0;
            for (int16_t i = 0; i < nsel; i++) sum += s_selsc[i];
        }
        if (sum == 0) { s_selsc[0] = 1; s_selix[0] = 0; nsel = 1; sum = 1; }
        int32_t sum8 = sum >> 8;                   /* out = num/(sum>>8) */
        if (sum8 == 0) sum8 = 1;

        /* weighted V sum over survivors. int16 x int8 keeps this a
         * single MULS.W per term on 68000. */
        for (int16_t d = 0; d < EG_HD; d++) {
            int32_t num = 0;
            for (int16_t s = 0; s < nsel; s++)
                num += mul_ss(s_selsc[s], (int16_t)st->vc[li][s_selix[s]][off + d]);
            s_attn[off + d] = sat16(num / sum8);
        }
    }

    eg_matvec(&st->wo[li], s_attn, s_proj, EG_EMBED, EG_EMBED);
    for (int16_t i = 0; i < EG_EMBED; i++)
        x[i] = sat16((int32_t)s_res[i] + s_proj[i]);

    /* FFN */
    memcpy(s_res, x, sizeof(s_res));
    eg_rmsnorm(x, EG_EMBED);
    eg_matvec(&st->wff1[li], x, s_ff, EG_EMBED, EG_FFN);
    for (int16_t i = 0; i < EG_FFN; i++)
        if (s_ff[i] < 0) s_ff[i] = 0;              /* ReLU */
    eg_matvec(&st->wff2[li], s_ff, s_proj, EG_FFN, EG_EMBED);
    for (int16_t i = 0; i < EG_EMBED; i++)
        x[i] = sat16((int32_t)s_res[i] + s_proj[i]);
}

/* ---- public API -------------------------------------------------------- */

/* Walk one tensor's index-stream rows and return the byte after it. */
static const uint8_t *eg_scan_tensor(EgTensor *t, const uint8_t *p,
                                     uint16_t rows, uint32_t wcount)
{
    t->M = rd16be(p); p += 2;
    t->S = *p++;      p++;                    /* pad */
    t->packed = p;
    if (eg_streams) {
        for (uint16_t r = 0; r < rows; r++) {
            uint16_t na = rd16be(p);
            uint16_t ns = rd16be(p + 2);
            p += 4 + na + ns;
        }
    } else {
        p += wcount >> 2;
    }
    return p;
}

/* Point the six per-layer tensors at an expert's weight block. This is
 * the whole cost of "activating" an expert — ROM is memory-mapped, so
 * nothing is copied. */
void eg_select_expert(EgState *st, uint16_t expert)
{
    if (expert >= st->n_experts) expert = 0;
    st->expert = expert;
    const uint8_t *p = st->expert_base[expert];
    for (int16_t li = 0; li < EG_LAYERS; li++) {
        p = eg_scan_tensor(&st->wq[li],   p, EG_EMBED, EG_EMBED * EG_EMBED);
        p = eg_scan_tensor(&st->wk[li],   p, EG_EMBED, EG_EMBED * EG_EMBED);
        p = eg_scan_tensor(&st->wv[li],   p, EG_EMBED, EG_EMBED * EG_EMBED);
        p = eg_scan_tensor(&st->wo[li],   p, EG_EMBED, EG_EMBED * EG_EMBED);
        p = eg_scan_tensor(&st->wff1[li], p, EG_FFN,
                           (uint32_t)EG_FFN * EG_EMBED);
        p = eg_scan_tensor(&st->wff2[li], p, EG_EMBED,
                           (uint32_t)EG_EMBED * EG_FFN);
    }
}

#include "elya_router.h"

/* case-insensitive substring search, ASCII only */
static int eg_find(const char *hay, const char *needle)
{
    for (const char *h = hay; *h; h++) {
        const char *a = h, *b = needle;
        while (*b) {
            char ca = *a, cb = *b;
            if (ca >= 'A' && ca <= 'Z') ca = (char)(ca + 32);
            if (cb >= 'A' && cb <= 'Z') cb = (char)(cb + 32);
            if (ca != cb) break;
            a++; b++;
        }
        if (!*b) return 1;
    }
    return 0;
}

uint16_t eg_route(EgState *st, const char *prompt)
{
    if (st->n_experts <= 1) return 0;

    /* Deterministic ROM keyword router. Generated from the same patterns
     * that produced the training labels, so it agrees with the shard
     * split by construction. Runs once per ANSWER, not per token. */
    if (st->n_experts == ER_N_EXPERTS) {
        uint16_t best = 0, best_hits = 0;
        for (uint16_t e = 0; e < ER_N_EXPERTS; e++) {
            uint16_t hits = 0;
            for (const char *const *k = ER_KW[e]; *k; k++)
                if (eg_find(prompt, *k)) hits++;
            if (hits > best_hits) { best_hits = hits; best = e; }
        }
        if (best_hits) return best;
        /* no keyword hit: fall through to the learned router */
    }

    /* [mean-pool | max-pool] of the prompt's embeddings, Q2.6 -> Q3.12.
     * Mean alone is a bag-of-bytes and cannot separate prompts differing
     * by one distinctive word; max captures that word's PRESENCE. This
     * took an 8-way router from ~60% to usable. Cost: 64 compares per
     * character. */
    int16_t pooled[EG_EMBED * 2];
    int32_t acc[EG_EMBED];
    int8_t  mx[EG_EMBED];
    int32_t n = 0;
    for (int16_t i = 0; i < EG_EMBED; i++) { acc[i] = 0; mx[i] = -128; }
    for (const char *c = prompt; *c && n < EG_CTX; c++) {
        const int8_t *e = st->emb + (uint32_t)(uint8_t)*c * EG_EMBED;
        for (int16_t i = 0; i < EG_EMBED; i++) {
            acc[i] += e[i];
            if (e[i] > mx[i]) mx[i] = e[i];
        }
        n++;
    }
    if (n == 0) return 0;
    /* Scale BEFORE dividing. Dividing first threw away the fractional
     * part of the mean, which is most of the router's signal — it
     * changed 2 of 6 routing decisions versus the trained model. Also
     * avoids a UB left-shift of a negative value. Prompt is capped at
     * EG_CTX to match training. */
    for (int16_t i = 0; i < EG_EMBED; i++) {
        pooled[i]             = sat16((acc[i] * 64) / n);
        pooled[EG_EMBED + i]  = (int16_t)(mx[i] * 64);
    }

    int16_t logits[EG_MAX_EXPERTS];
    eg_matvec(&st->router, pooled, logits, EG_EMBED * 2,
              (int16_t)st->n_experts);

    uint16_t best = 0;
    for (uint16_t e = 1; e < st->n_experts; e++)
        if (logits[e] > logits[best]) best = e;
    return best;
}

int eg_init(EgState *st, const uint8_t *blob)
{
    memset(st, 0, sizeof(*st));
    if (blob[0] != 'S' || blob[1] != 'G' || blob[2] != 'T') return -1;

    /* ---- SGTM: Lock-On MoE (shared embedding + N banked experts) ---- */
    if (blob[3] == 'M') {
        eg_streams = (rd16be(blob + 14) & 2) ? 1 : 0;
        {
        uint16_t ne = blob[4];
        if (ne == 0 || ne > EG_MAX_EXPERTS) return -5;
        if (blob[5] != EG_LAYERS || blob[6] != EG_HEADS) return -2;
        if (rd16be(blob + 8) != EG_EMBED || rd16be(blob + 10) != EG_VOCAB)
            return -3;
        if (rd16be(blob + 12) != EG_CTX) return -4;

        const uint8_t *p = blob + 16;          /* after flags */
        st->emb = (const int8_t *)p;
        p += (uint32_t)EG_VOCAB * EG_EMBED;
        if (rd16be(blob + 14) & 4) {           /* flags bit2: PE table */
            st->pe = (const int8_t *)p;
            p += (uint32_t)EG_CTX * EG_EMBED;
        }
        st->explut = p;
        p += 512;
        p = eg_scan_tensor(&st->router, p, ne, (uint32_t)ne * EG_EMBED * 2);

        st->n_experts = ne;
        for (uint16_t e = 0; e < ne; e++) {
            uint32_t off = ((uint32_t)p[0] << 24) | ((uint32_t)p[1] << 16)
                         | ((uint32_t)p[2] << 8)  | p[3];
            p += 4;
            st->expert_base[e] = blob + off;
        }
        st->ok = 1;
        eg_select_expert(st, 0);
        return 0;
        }
    }

    /* ---- SGT1 / SGT2: single model, treated as one expert ---------- */
    if (blob[3] != '1' && blob[3] != '2' && blob[3] != '3') return -1;
    eg_streams = (blob[12] & 2) ? 1 : 0;
    if (blob[4] != EG_LAYERS || blob[5] != EG_HEADS) return -2;
    if (rd16be(blob + 6) != EG_EMBED || rd16be(blob + 8) != EG_VOCAB)
        return -3;
    if (rd16be(blob + 10) != EG_CTX) return -4;

    const uint8_t *p = blob + 12 + 2;  /* +flags,pad */
    st->emb = (const int8_t *)p;
    p += (uint32_t)EG_VOCAB * EG_EMBED;
    if (blob[12] & 4) {                /* flags bit2: positional encoding */
        st->pe = (const int8_t *)p;
        p += (uint32_t)EG_CTX * EG_EMBED;
    }
    st->n_experts = 1;
    st->expert_base[0] = p;

    for (int16_t li = 0; li < EG_LAYERS; li++) {
        EgTensor *order[6];
        uint32_t  wcount[6];
        uint16_t  rows[6];
        order[0] = &st->wq[li];   wcount[0] = EG_EMBED * EG_EMBED;
        order[1] = &st->wk[li];   wcount[1] = EG_EMBED * EG_EMBED;
        order[2] = &st->wv[li];   wcount[2] = EG_EMBED * EG_EMBED;
        order[3] = &st->wo[li];   wcount[3] = EG_EMBED * EG_EMBED;
        order[4] = &st->wff1[li]; wcount[4] = (uint32_t)EG_FFN * EG_EMBED;
        order[5] = &st->wff2[li]; wcount[5] = (uint32_t)EG_EMBED * EG_FFN;
        rows[0] = rows[1] = rows[2] = rows[3] = EG_EMBED;
        rows[4] = EG_FFN;                  /* wff1: FFN outputs */
        rows[5] = EG_EMBED;                /* wff2: EMBED outputs */
        for (int16_t ti = 0; ti < 6; ti++) {
            order[ti]->M = rd16be(p); p += 2;
            order[ti]->S = *p++;      p++;         /* pad */
            order[ti]->packed = p;
#if EG_FORMAT_INDEX_STREAMS
            /* rows are variable length: walk their headers to find the end */
            for (uint16_t r = 0; r < rows[ti]; r++) {
                uint16_t na = rd16be(p);
                uint16_t ns = rd16be(p + 2);
                p += 4 + na + ns;
            }
            (void)wcount;
#else
            (void)rows;
            p += wcount[ti] >> 2;                  /* 4 weights per byte */
#endif
        }
    }
    st->explut = p;
    st->ok = 1;
    return 0;
}

void eg_reset(EgState *st)
{
    memset(st->kc, 0, sizeof(st->kc));
    memset(st->vc, 0, sizeof(st->vc));
    memset(st->x, 0, sizeof(st->x));
    st->pos = 0;
}

uint8_t eg_next_token(EgState *st, uint8_t input)
{
    if (!st->ok) return 0;
    /* ring slot for this token; EG_CTX is a power of 2 so & wraps free */
    int16_t slot  = (int16_t)(st->pos & (EG_CTX - 1));
    int16_t n_ctx = (st->pos < EG_CTX) ? (int16_t)(st->pos + 1) : EG_CTX;

    /* 1. embedding lookup: int8 Q2.6 -> int16 Q3.12 */
    {
        const int8_t *e = st->emb + (uint32_t)input * EG_EMBED;
        if (st->pe) {
            /* Absolute position, clamped at the context end. Summing in
             * int16 first is safe: (127+127)*64 = 16256 < 32767. */
            int16_t pi = (st->pos < EG_CTX) ? st->pos : (EG_CTX - 1);
            const int8_t *pv = st->pe + (uint32_t)pi * EG_EMBED;
            for (int16_t i = 0; i < EG_EMBED; i++)
                st->x[i] = (int16_t)(((int16_t)e[i] + pv[i]) * 64);
        } else {
            for (int16_t i = 0; i < EG_EMBED; i++)
                st->x[i] = (int16_t)(e[i] * 64);   /* Q2.6 -> Q3.12 */
        }
    }

    /* 2. layers */
    for (int16_t li = 0; li < EG_LAYERS; li++)
        eg_layer(st, li, slot, n_ctx);

    /* 3. final norm */
    eg_rmsnorm(st->x, EG_EMBED);

    /* 4. tied-embedding logits, greedy argmax over printable ASCII */
    {
        int32_t best = (int32_t)0x80000000;
        int16_t bestv = ' ';
        for (int16_t v = 32; v <= 126; v++) {
            const int8_t *e = st->emb + (uint32_t)v * EG_EMBED;
            int32_t acc = 0;
            for (int16_t i = 0; i < EG_EMBED; i++)
                acc += (int32_t)e[i] * st->x[i];
            if (acc > best) { best = acc; bestv = v; }
        }
        /* newline is a legal stop token too: compare its logit */
        {
            const int8_t *e = st->emb + 10u * EG_EMBED;
            int32_t acc = 0;
            for (int16_t i = 0; i < EG_EMBED; i++)
                acc += (int32_t)e[i] * st->x[i];
            if (acc > best) bestv = '\n';
        }

        /* 5. advance the ring. No memmove: the oldest slot is simply
         * overwritten next token (attention is order-invariant here). */
        st->pos++;
        if (st->pos > 30000) st->pos = EG_CTX;   /* keep the counter sane */
        return (uint8_t)bestv;
    }
}
