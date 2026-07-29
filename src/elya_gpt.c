/* SPDX-License-Identifier: AGPL-3.0
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
static void eg_matvec(const EgTensor *t, const int16_t *in, int16_t *out,
                      int16_t in_dim, int16_t out_dim)
{
    uint16_t M = t->M;
    uint8_t  S = t->S;

#if EG_FORMAT_INDEX_STREAMS
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
#else
    /* SGT1: 2-bit packed, 4 weights per byte */
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
#endif
}

/* ---- RMS norm, parameter-free, all int -------------------------------- */
static void eg_rmsnorm(int16_t *x, int16_t n)
{
    uint32_t ss = 0;
    for (int16_t i = 0; i < n; i++) {
        int16_t h = (int16_t)(x[i] >> 4);          /* Q8 */
        ss += (uint32_t)((int32_t)h * h);          /* Q16 */
    }
    uint32_t ms = ss / (uint16_t)n;                /* mean square, Q16 */
    uint16_t rms = isqrt32(ms);                    /* Q8 */
    if (rms == 0) rms = 1;
    int32_t r = (int32_t)(1UL << 18) / rms;        /* Q10 reciprocal */
    if (r > 16383) r = 16383;                      /* max gain 16x   */
    for (int16_t i = 0; i < n; i++)
        x[i] = sat16(((int32_t)x[i] * r) >> 10);
}

/* ---- static scratch (keep the 68K stack small) ------------------------ */
static int16_t s_q[EG_EMBED], s_k[EG_EMBED], s_v[EG_EMBED];
static int16_t s_attn[EG_EMBED], s_proj[EG_EMBED], s_res[EG_EMBED];
static int16_t s_ff[EG_FFN];
static int32_t s_score[EG_CTX];
static int32_t s_selsc[EG_CTX];   /* collapse survivors: scores  */
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
                sc += (int32_t)qh[d] * kh[d];
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
            if (d > 155) continue;              /* exp(-9.7) -> 0 in Q14 */
            uint16_t w = rd16be(st->explut + (d << 1));
            if (w == 0) continue;
            s_selsc[nsel] = (int32_t)w;
            s_selix[nsel] = tt;
            sum += w;
            if (++nsel >= EG_TOPK) break;
        }
        if (sum == 0) { s_selsc[0] = 1; sum = 1; }
        int32_t sum8 = sum >> 8;                   /* out = num/(sum>>8) */
        if (sum8 == 0) sum8 = 1;

        /* weighted V sum over survivors: EG_TOPK terms, not n_ctx */
        for (int16_t d = 0; d < EG_HD; d++) {
            int32_t num = 0;
            for (int16_t s = 0; s < nsel; s++)
                num += s_selsc[s] * st->vc[li][s_selix[s]][off + d];
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

int eg_init(EgState *st, const uint8_t *blob)
{
    memset(st, 0, sizeof(*st));
    if (blob[0] != 'S' || blob[1] != 'G' || blob[2] != 'T') return -1;
#if EG_FORMAT_INDEX_STREAMS
    if (blob[3] != '2') return -1;   /* engine built for index streams */
#else
    if (blob[3] != '1') return -1;   /* engine built for packed 2-bit  */
#endif
    if (blob[4] != EG_LAYERS || blob[5] != EG_HEADS) return -2;
    if (rd16be(blob + 6) != EG_EMBED || rd16be(blob + 8) != EG_VOCAB)
        return -3;
    if (rd16be(blob + 10) != EG_CTX) return -4;

    const uint8_t *p = blob + 12 + 2;  /* +flags,pad */
    st->emb = (const int8_t *)p;
    p += (uint32_t)EG_VOCAB * EG_EMBED;

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
        for (int16_t i = 0; i < EG_EMBED; i++)
            st->x[i] = (int16_t)((int16_t)e[i] << 6);
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
