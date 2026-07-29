/* SPDX-License-Identifier: AGPL-3.0
 * elya_gpt.c — integer-only transformer inference for the Sega Genesis.
 * See elya_gpt.h for format. Mirrors train/train_elya_genesis.py QAT
 * semantics: any rounding done here was fake-quantized during training.
 */
#include "elya_gpt.h"
#ifdef SGDK_GCC
#include <genesis.h>       /* SGDK's string.h needs its types loaded first */
#else
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
    const uint8_t *row = t->packed;
    int16_t ib = in_dim >> 2;
    uint16_t M = t->M;
    uint8_t  S = t->S;

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

/* ---- one transformer layer -------------------------------------------- */
static void eg_layer(EgState *st, int16_t li, int16_t pos)
{
    int16_t *x = st->x;
    int16_t n_ctx = (int16_t)(pos + 1);
    if (n_ctx > EG_CTX) n_ctx = EG_CTX;

    memcpy(s_res, x, sizeof(s_res));
    eg_rmsnorm(x, EG_EMBED);

    eg_matvec(&st->wq[li], x, s_q, EG_EMBED, EG_EMBED);
    eg_matvec(&st->wk[li], x, s_k, EG_EMBED, EG_EMBED);
    eg_matvec(&st->wv[li], x, s_v, EG_EMBED, EG_EMBED);

    /* store K,V in int8 Q4.4 cache */
    for (int16_t i = 0; i < EG_EMBED; i++) {
        int16_t k8 = (int16_t)(s_k[i] >> 8);
        int16_t v8 = (int16_t)(s_v[i] >> 8);
        st->kc[li][pos][i] = (int8_t)(k8 > 127 ? 127 : (k8 < -128 ? -128 : k8));
        st->vc[li][pos][i] = (int8_t)(v8 > 127 ? 127 : (v8 < -128 ? -128 : v8));
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

        /* softmax numerators from ROM exp LUT: w = exp(-(mx-sc)) in Q14 */
        int32_t sum = 0;
        for (int16_t tt = 0; tt < n_ctx; tt++) {
            uint32_t d = (uint32_t)(mx - s_score[tt]) >> 12;  /* 1/16 units */
            uint16_t w = (d > 255) ? 0
                       : rd16be(st->explut + (d << 1));
            s_score[tt] = (int32_t)w;
            sum += w;
        }
        if (sum == 0) sum = 1;
        int32_t sum8 = sum >> 8;                   /* out = num/(sum>>8) */
        if (sum8 == 0) sum8 = 1;

        /* weighted V sum: num(Q14 x Q4.4) / sum -> Q12 out */
        for (int16_t d = 0; d < EG_HD; d++) {
            int32_t num = 0;
            for (int16_t tt = 0; tt < n_ctx; tt++)
                num += s_score[tt] * st->vc[li][tt][off + d];
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
    if (blob[0] != 'S' || blob[1] != 'G' || blob[2] != 'T' || blob[3] != '1')
        return -1;
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
        order[0] = &st->wq[li];   wcount[0] = EG_EMBED * EG_EMBED;
        order[1] = &st->wk[li];   wcount[1] = EG_EMBED * EG_EMBED;
        order[2] = &st->wv[li];   wcount[2] = EG_EMBED * EG_EMBED;
        order[3] = &st->wo[li];   wcount[3] = EG_EMBED * EG_EMBED;
        order[4] = &st->wff1[li]; wcount[4] = (uint32_t)EG_FFN * EG_EMBED;
        order[5] = &st->wff2[li]; wcount[5] = (uint32_t)EG_EMBED * EG_FFN;
        for (int16_t ti = 0; ti < 6; ti++) {
            order[ti]->M = rd16be(p); p += 2;
            order[ti]->S = *p++;      p++;         /* pad */
            order[ti]->packed = p;
            p += wcount[ti] >> 2;                  /* 4 weights per byte */
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
    int16_t pos = st->pos;

    /* 1. embedding lookup: int8 Q2.6 -> int16 Q3.12 */
    {
        const int8_t *e = st->emb + (uint32_t)input * EG_EMBED;
        for (int16_t i = 0; i < EG_EMBED; i++)
            st->x[i] = (int16_t)((int16_t)e[i] << 6);
    }

    /* 2. layers */
    for (int16_t li = 0; li < EG_LAYERS; li++)
        eg_layer(st, li, pos);

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

        /* 5. advance KV position (sliding window) */
        if (st->pos < EG_CTX - 1) {
            st->pos++;
        } else {
            /* slide window left; dst < src so a forward copy is safe
             * (SGDK has no memmove) */
            for (int16_t li = 0; li < EG_LAYERS; li++) {
                int8_t *kd = &st->kc[li][0][0];
                int8_t *vd = &st->vc[li][0][0];
                const uint32_t nb = (uint32_t)(EG_CTX - 1) * EG_EMBED;
                for (uint32_t i = 0; i < nb; i++) {
                    kd[i] = kd[i + EG_EMBED];
                    vd[i] = vd[i + EG_EMBED];
                }
            }
        }
        return (uint8_t)bestv;
    }
}
