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

/* Same value, but for a pointer the format GUARANTEES is 2-byte aligned.
 * On the (big-endian) 68000 that is one MOVE.W off the 16-bit cart bus
 * instead of two byte reads, a shift and an OR. */
static inline uint16_t rd16be_a(const uint8_t *p) {
#if defined(__m68k__) && !defined(__mcoldfire__)
    return *(const uint16_t *)p;
#else
    return (uint16_t)((p[0] << 8) | p[1]);
#endif
}

/* in[] element at a pre-doubled byte offset held in the W16 index stream. */
/* The cast to int16_t is load-bearing on 68000: it lets GCC use the
 * (An,Dn.W) index mode instead of zero-extending into a full 32-bit
 * register first. Offsets are 0..510 so the cast is exact. */
#define EG_IN_AT(inb, p) \
    (*(const int16_t *)((inb) + (int16_t)rd16be_a(p)))

/* Accumulate floor(N/4) groups of four W16 stream entries into ACC,
 * advancing P and leaving the 0..3 remainder in N.
 *
 * This is the innermost loop of the entire engine (2.5M iterations per
 * 38-token run), so it is worth saying exactly what we want. GCC's best
 * output was 34 cycles per weight:
 *     move.w (a5),a3 ; move.w (a0,a3.l),d0 ; ext.l d0 ; add.l a1,d0
 * plus 6 more for ADDQ/CMP/Bcc loop control. Loading the offset into a DATA
 * register lets the (An,Dn.W) index mode read straight into an ADDRESS
 * register, which sign-extends for free and needs no EXT.L (30 cycles), and
 * DBRA replaces the three-instruction loop tail (2.5 instead of 6). The C
 * fallback below is what the host harness builds, so both stay identical. */
#if defined(__m68k__) && !defined(__mcoldfire__)
#define EG_W16_STEP(OP)                     \
    "move.w (%[p])+,%[ix]\n\t"              \
    "move.w (%[in],%[ix].w),%[v]\n\t"       \
    OP " %[v],%[acc]\n\t"
#define EG_W16_RUN4(OP, ACC, P, INB, N)                                  \
    do {                                                                 \
        if ((N) >= 4) {                                                  \
            uint16_t eg__g = (uint16_t)(((N) >> 2) - 1);                 \
            const uint8_t *eg__v;                                        \
            int32_t eg__ix;                                              \
            __asm__ (                                                    \
                "1:\n\t"                                                 \
                EG_W16_STEP(OP) EG_W16_STEP(OP)                          \
                EG_W16_STEP(OP) EG_W16_STEP(OP)                          \
                "dbra %[g],1b"                                           \
                : [acc]"+d"(ACC), [p]"+a"(P), [v]"=&a"(eg__v),           \
                  [ix]"=&d"(eg__ix), [g]"+d"(eg__g)                      \
                : [in]"a"(INB)                                           \
                : "cc", "memory");                                       \
            (N) &= 3;                                                    \
        }                                                                \
    } while (0)
/* ...and the 0..3 tail. GCC turned `while (i--)` into a CMPI/Bcc ladder
 * that measured as ~10% of the whole run; one DBRA loop is smaller and
 * branch-free per element. */
#define EG_W16_RUN1(OP, ACC, P, INB, N)                                  \
    do {                                                                 \
        if ((N)) {                                                       \
            uint16_t eg__g = (uint16_t)((N) - 1);                        \
            const uint8_t *eg__v;                                        \
            int32_t eg__ix;                                              \
            __asm__ (                                                    \
                "1:\n\t"                                                 \
                EG_W16_STEP(OP)                                          \
                "dbra %[g],1b"                                           \
                : [acc]"+d"(ACC), [p]"+a"(P), [v]"=&a"(eg__v),           \
                  [ix]"=&d"(eg__ix), [g]"+d"(eg__g)                      \
                : [in]"a"(INB)                                           \
                : "cc", "memory");                                       \
            (N) = 0;                                                     \
        }                                                                \
    } while (0)
#else
#define EG_W16_RUN4(OP, ACC, P, INB, N)                                  \
    do {                                                                 \
        while ((N) >= 4) {                                               \
            if (OP[0] == 'a') {                                          \
                (ACC) += EG_IN_AT(INB, P);                               \
                (ACC) += EG_IN_AT(INB, (P) + 2);                         \
                (ACC) += EG_IN_AT(INB, (P) + 4);                         \
                (ACC) += EG_IN_AT(INB, (P) + 6);                         \
            } else {                                                     \
                (ACC) -= EG_IN_AT(INB, P);                               \
                (ACC) -= EG_IN_AT(INB, (P) + 2);                         \
                (ACC) -= EG_IN_AT(INB, (P) + 4);                         \
                (ACC) -= EG_IN_AT(INB, (P) + 6);                         \
            }                                                            \
            (P) += 8;                                                    \
            (N) -= 4;                                                    \
        }                                                                \
    } while (0)
#define EG_W16_RUN1(OP, ACC, P, INB, N)                                  \
    do {                                                                 \
        while ((N)) {                                                    \
            if (OP[0] == 'a') (ACC) += EG_IN_AT(INB, P);                 \
            else              (ACC) -= EG_IN_AT(INB, P);                 \
            (P) += 2;                                                    \
            (N)--;                                                       \
        }                                                                \
    } while (0)
#endif

/* ---- input-major (transposed) step ------------------------------------
 * The mirror image of EG_W16_STEP. There the stream held an offset into
 * the INPUT and the value came back into an accumulator register; here the
 * stream holds an offset into an accumulator ARRAY and the value is
 * already in a register, so the memory access becomes a read-modify-write:
 *
 *     move.w (a2)+,d1            8 cycles
 *     add.l  d0,(a3,d1.w)       12 + 14 EA = 26 cycles
 *
 * 34 cycles per weight against 30 for the output-major step — the
 * transposed loop is ~13% MORE expensive per weight. It only wins because
 * it never executes at all for an input ReLU zeroed, and 56.5% of wff2's
 * weights point at one of those. Offsets are pre-multiplied by 4 by the
 * exporter (o*4, range 0..252), so there is no shift and no zero-extend,
 * exactly as the W16 conversion pre-doubled its own indices. */
#if defined(__m68k__) && !defined(__mcoldfire__)
#define EG_T16_STEP(OP)                     \
    "move.w (%[p])+,%[ix]\n\t"              \
    OP " %[v],(%[ab],%[ix].w)\n\t"
#define EG_T16_RUN4(OP, V, P, AB, N)                                     \
    do {                                                                 \
        if ((N) >= 4) {                                                  \
            uint16_t eg__g = (uint16_t)(((N) >> 2) - 1);                 \
            int32_t eg__ix;                                              \
            __asm__ (                                                    \
                "1:\n\t"                                                 \
                EG_T16_STEP(OP) EG_T16_STEP(OP)                          \
                EG_T16_STEP(OP) EG_T16_STEP(OP)                          \
                "dbra %[g],1b"                                           \
                : [p]"+a"(P), [ix]"=&d"(eg__ix), [g]"+d"(eg__g)          \
                : [v]"d"(V), [ab]"a"(AB)                                 \
                : "cc", "memory");                                       \
            (N) &= 3;                                                    \
        }                                                                \
    } while (0)
#define EG_T16_RUN1(OP, V, P, AB, N)                                     \
    do {                                                                 \
        if ((N)) {                                                       \
            uint16_t eg__g = (uint16_t)((N) - 1);                        \
            int32_t eg__ix;                                              \
            __asm__ (                                                    \
                "1:\n\t"                                                 \
                EG_T16_STEP(OP)                                          \
                "dbra %[g],1b"                                           \
                : [p]"+a"(P), [ix]"=&d"(eg__ix), [g]"+d"(eg__g)          \
                : [v]"d"(V), [ab]"a"(AB)                                 \
                : "cc", "memory");                                       \
            (N) = 0;                                                     \
        }                                                                \
    } while (0)
#else
/* Accumulator slot at a pre-quadrupled byte offset held in the stream. */
#define EG_ACC_AT(ab, p) (*(int32_t *)((ab) + (int16_t)rd16be_a(p)))
#define EG_T16_RUN4(OP, V, P, AB, N)                                     \
    do {                                                                 \
        while ((N) >= 4) {                                               \
            if (OP[0] == 'a') {                                          \
                EG_ACC_AT(AB, P)       += (V);                           \
                EG_ACC_AT(AB, (P) + 2) += (V);                           \
                EG_ACC_AT(AB, (P) + 4) += (V);                           \
                EG_ACC_AT(AB, (P) + 6) += (V);                           \
            } else {                                                     \
                EG_ACC_AT(AB, P)       -= (V);                           \
                EG_ACC_AT(AB, (P) + 2) -= (V);                           \
                EG_ACC_AT(AB, (P) + 4) -= (V);                           \
                EG_ACC_AT(AB, (P) + 6) -= (V);                           \
            }                                                            \
            (P) += 8;                                                    \
            (N) -= 4;                                                    \
        }                                                                \
    } while (0)
#define EG_T16_RUN1(OP, V, P, AB, N)                                     \
    do {                                                                 \
        while ((N)) {                                                    \
            if (OP[0] == 'a') EG_ACC_AT(AB, P) += (V);                   \
            else              EG_ACC_AT(AB, P) -= (V);                   \
            (P) += 2;                                                    \
            (N)--;                                                       \
        }                                                                \
    } while (0)
#endif

/* 16x16 -> 32 signed multiply.
 * The 68000 HAS MULS.W (70 cycles) but no 32x32 multiply, so any C
 * expression GCC believes is 32-bit becomes a __mulsi3 call: three
 * MULUs plus call overhead, ~250 cycles. GCC will not always narrow a
 * value it has tracked through a 32-bit division, so we say it
 * outright. The host path is plain C, keeping both builds identical. */
static inline int32_t mul_ss(int16_t a, int16_t b) {
#if defined(__m68k__) && !defined(__mcoldfire__)
    /* `int32_t r = a;` made GCC emit an EXT.L that MULS.W then overwrites
     * anyway -- 4 dead cycles on every product. Only the low word has to be
     * set before the multiply. */
    int32_t r;
    __asm__ ("move.w %1,%0\n\tmuls.w %2,%0" : "=&d"(r) : "dmi"(a), "dm"(b));
    return r;
#else
    return (int32_t)a * (int32_t)b;
#endif
}

/* 32x16 -> low 32 signed multiply.
 * The requant step is `acc * M` with acc a 32-bit accumulator and M a
 * per-tensor multiplier that is always < 128 in the shipped blobs. GCC
 * therefore has to call __mulsi3 (three MULU.W plus call/return and the
 * argument push), ~280 cycles, once per output row — 1152 rows per forward
 * pass. Two MULU.W do the same job: the low 32 bits of a two's-complement
 * product only need the low half of the multiplier, and the high partial
 * product is shifted left 16 so its own overflow is discarded. Bit-identical
 * to `acc * (int32_t)M` including the wrap on overflow. */
static inline int32_t mul32x16(int32_t a, uint16_t m) {
#if defined(__m68k__) && !defined(__mcoldfire__)
    /* Most accumulators still fit in 16 bits, and m is < 128, so one
     * MULS.W (~50 cycles) covers them exactly. */
    if ((int32_t)(int16_t)a == a) {
        int32_t r = a;
        __asm__ ("muls.w %1,%0" : "+d"(r) : "dm"(m));
        return r;
    }
    uint32_t lo = (uint32_t)a & 0xFFFFu;
    uint32_t hi = (uint32_t)a >> 16;
    __asm__ ("mulu.w %1,%0" : "+d"(lo) : "dm"(m));
    __asm__ ("mulu.w %1,%0" : "+d"(hi) : "dm"(m));
    return (int32_t)(lo + (hi << 16));
#else
    return a * (int32_t)m;
#endif
}

/* sat16(num / den) for a POSITIVE den that fits in 16 bits.
 * GCC has to call __divsi3 for a 32/32 divide (~600 cycles); the 68000's
 * own DIVS.W does 32/16 in <=158. It traps on den==0 (callers guarantee
 * den>0) and sets V when the quotient will not fit in 16 bits -- which is
 * exactly the case sat16 was going to clamp anyway, and since den>0 the
 * quotient's sign is num's sign. So this is bit-identical to
 * sat16(num/den), including C's truncate-toward-zero. */
static inline int16_t div_sat16(int32_t num, int16_t den) {
#if defined(__m68k__) && !defined(__mcoldfire__)
    int32_t q = num;
    uint8_t ovf;
    __asm__ ("divs.w %2,%0\n\tsvs %1"
             : "+d"(q), "=d"(ovf) : "dm"(den) : "cc");
    if (ovf) return (num < 0) ? (int16_t)-32768 : (int16_t)32767;
    return (int16_t)q;
#else
    int32_t v = num / den;
    if (v > 32767)  return 32767;
    if (v < -32768) return -32768;
    return (int16_t)v;
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
/* 0 = SGT1 2-bit packed, 1 = byte index streams, 2 = W16 index streams
 * (pre-doubled big-endian u16 byte offsets, flags bit3). */
static uint8_t eg_streams = 1;
/* flags bit4: wff2 is stored INPUT-MAJOR (transposed). Requires bit3. */
static uint8_t eg_wff2_t = 0;

/* Rows in a wff2 tensor's stream: EG_FFN columns when transposed, else
 * EG_EMBED output rows. eg_scan_tensor needs this to find the tensor end. */
#define EG_WFF2_ROWS (eg_wff2_t ? (uint16_t)EG_FFN : (uint16_t)EG_EMBED)

/* ---- input-major matvec: out = requant(W * in), skipping zero inputs ---
 * Same result as eg_matvec, reached by walking the INPUT instead of the
 * output. Each accumulator receives the identical set of +/- terms, just in
 * a different order; int32 addition is exact and cannot overflow here
 * (256 terms x 32767 = 8,388,352, well inside 2^31), so the output is
 * bit-identical.
 *
 * The point is the `if (v == 0)` skip: one pointer add jumps the whole
 * weight list for an input ReLU zeroed, without reading one index from
 * cart ROM. Only wff2 is stored this way — every other tensor's input is
 * dense, so they would pay the more expensive step for no skips. */
static int32_t s_facc[EG_EMBED];

static void eg_matvec_t(const EgTensor *t, const int16_t *in, int16_t *out,
                        int16_t in_dim, int16_t out_dim)
{
    uint16_t M = t->M;
    uint8_t  S = t->S;
    const uint8_t *p = t->packed;
    uint8_t *ab = (uint8_t *)s_facc;

    for (int16_t o = 0; o < out_dim; o++) s_facc[o] = 0;

    for (int16_t i = 0; i < in_dim; i++) {
        uint16_t na = rd16be_a(p);
        uint16_t ns = rd16be_a(p + 2);
        p += 4;
        int32_t v = in[i];
        if (v == 0) {                       /* ReLU zero: skip the column */
            p += ((uint32_t)na + ns) << 1;
            continue;
        }
        uint16_t n = na;
        EG_T16_RUN4("add.l", v, p, ab, n);
        EG_T16_RUN1("add.l", v, p, ab, n);
        n = ns;
        EG_T16_RUN4("sub.l", v, p, ab, n);
        EG_T16_RUN1("sub.l", v, p, ab, n);
    }

    for (int16_t o = 0; o < out_dim; o++)
        out[o] = sat16(mul32x16(s_facc[o], M) >> S);
}

static void eg_matvec(const EgTensor *t, const int16_t *in, int16_t *out,
                      int16_t in_dim, int16_t out_dim)
{
    uint16_t M = t->M;
    uint8_t  S = t->S;

if (eg_streams == 2) {
    /* W16: same rows as T2, but each index is a 2-byte big-endian BYTE
     * OFFSET into `in` (i.e. the index already doubled by the exporter).
     * That removes the zero-extend and the doubling from the innermost
     * loop in the whole engine, and halves its cart-ROM bus traffic. The
     * rows are aligned by construction, so MOVE.W is legal. */
    const uint8_t *p   = t->packed;
    const uint8_t *inb = (const uint8_t *)in;
    (void)in_dim;
    for (int16_t o = 0; o < out_dim; o++) {
        /* aligned reads: the row header is 2 MOVE.W, not 2x(byte,shift,or) */
        uint16_t na = rd16be_a(p); p += 2;
        uint16_t ns = rd16be_a(p); p += 2;
        int32_t acc = 0;
        uint16_t i = na;
        EG_W16_RUN4("add.l", acc, p, inb, i);
        EG_W16_RUN1("add.l", acc, p, inb, i);
        i = ns;
        EG_W16_RUN4("sub.l", acc, p, inb, i);
        EG_W16_RUN1("sub.l", acc, p, inb, i);
        out[o] = sat16(mul32x16(acc, M) >> S);
    }
} else if (eg_streams) {
    /* T2: each row is (u16 n_add, u16 n_sub, add idx bytes, sub idx bytes).
     * No bit unpacking and no per-weight branch — zero weights are simply
     * absent. Compiles to MOVE.B (A1)+,D0 / ADD.W (A2,D0.W),D2 on 68K. */
    const uint8_t *p = t->packed;
    (void)in_dim;
    for (int16_t o = 0; o < out_dim; o++) {
        uint16_t na = rd16be(p); p += 2;
        uint16_t ns = rd16be(p); p += 2;
        int32_t acc = 0;
        uint16_t i;
        /* Unrolled 4x. The body is 4 instructions per weight; the loop
         * control (CMP.L + Bcc) was another 16 cycles on top of that, i.e.
         * ~28% of the inner loop, and GCC will not unroll a loop whose trip
         * count it cannot see. Same terms, same order -> same int32 sum. */
        for (i = na; i >= 8; i -= 8) {
            acc += in[p[0]]; acc += in[p[1]];
            acc += in[p[2]]; acc += in[p[3]];
            acc += in[p[4]]; acc += in[p[5]];
            acc += in[p[6]]; acc += in[p[7]];
            p += 8;
        }
        while (i--) acc += in[*p++];
        for (i = ns; i >= 8; i -= 8) {
            acc -= in[p[0]]; acc -= in[p[1]];
            acc -= in[p[2]]; acc -= in[p[3]];
            acc -= in[p[4]]; acc -= in[p[5]];
            acc -= in[p[6]]; acc -= in[p[7]];
            p += 8;
        }
        while (i--) acc -= in[*p++];
        out[o] = sat16(mul32x16(acc, M) >> S);
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
        out[o] = sat16(mul32x16(acc, M) >> S);
    }
}
}

/* ---- measurement-only instrumentation ---------------------------------
 * Compiled out entirely unless EG_STATS is defined, so the ROM and every
 * timed build are bit-identical with or without this block present. It
 * exists to answer one question with data instead of a guess: how much of
 * wff2's weight stream reads an input that ReLU already zeroed? */
#ifdef EG_STATS
#include <stdio.h>
uint32_t eg_st_ff_total, eg_st_ff_zero;       /* s_ff elements after ReLU */
uint32_t eg_st_w_reads, eg_st_w_zeroin;       /* wff2 weight reads        */
uint32_t eg_st_rows, eg_st_rows_all;          /* wff2 rows / all rows     */
uint32_t eg_st_all_reads;                     /* every tensor's reads     */

/* Walk an index-stream tensor without doing arithmetic; count how many of
 * its nonzero-weight reads land on a zero element of `in`. */
static void eg_stat_scan(const EgTensor *t, const int16_t *in,
                         int16_t out_dim, int is_wff2)
{
    const uint8_t *p = t->packed;
    for (int16_t o = 0; o < out_dim; o++) {
        uint16_t na = rd16be(p); p += 2;
        uint16_t ns = rd16be(p); p += 2;
        uint32_t n = (uint32_t)na + ns;
        eg_st_rows_all++;
        if (is_wff2) eg_st_rows++;
        for (uint32_t i = 0; i < n; i++) {
            uint16_t ix;
            if (eg_streams == 2) { ix = (uint16_t)(rd16be(p) >> 1); p += 2; }
            else                 { ix = *p++; }
            eg_st_all_reads++;
            if (is_wff2) {
                eg_st_w_reads++;
                if (in[ix] == 0) eg_st_w_zeroin++;
            }
        }
    }
}

void eg_stats_report(const char *what)
{
    printf("=== %s ===\n", what);
    printf("s_ff elements after ReLU : %u\n", eg_st_ff_total);
    printf("  of which ZERO          : %u  (%.2f%%)\n", eg_st_ff_zero,
           100.0 * eg_st_ff_zero / (eg_st_ff_total ? eg_st_ff_total : 1));
    printf("wff2 nonzero-weight reads: %u\n", eg_st_w_reads);
    printf("  hitting a ZERO input   : %u  (%.2f%%)\n", eg_st_w_zeroin,
           100.0 * eg_st_w_zeroin / (eg_st_w_reads ? eg_st_w_reads : 1));
    printf("ALL tensors' reads       : %u\n", eg_st_all_reads);
    printf("  wff2 share of all      : %.2f%%\n",
           100.0 * eg_st_w_reads / (eg_st_all_reads ? eg_st_all_reads : 1));
    printf("  skippable / all reads  : %.2f%%\n",
           100.0 * eg_st_w_zeroin / (eg_st_all_reads ? eg_st_all_reads : 1));
    printf("wff2 output rows         : %u  (all tensors: %u)\n",
           eg_st_rows, eg_st_rows_all);
}
#endif

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

#ifdef EG_STATS
    eg_stat_scan(&st->wq[li], x, EG_EMBED, 0);
    eg_stat_scan(&st->wk[li], x, EG_EMBED, 0);
    eg_stat_scan(&st->wv[li], x, EG_EMBED, 0);
#endif
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
         * single MULS.W per term on 68000.
         *
         * Loop order is survivor-OUTER / depth-INNER. With depth outer the
         * 68000 recomputed `s_selix[s] * EG_EMBED` (an LSL.L #6 plus two
         * LEAs) for every (s,d) pair — ~90 of the ~180 cycles per term went
         * into address arithmetic for a byte it then read once. Hoisting the
         * row pointer out of the depth loop computes it once per survivor
         * and walks the row with a post-increment.
         *
         * Byte-identical: each num[d] still accumulates the same terms in
         * the same s order, and int32 addition is exact
         * (64 * 16384 * 127 = 133M < 2^31). */
        {
            int32_t num[EG_HD];
            for (int16_t d = 0; d < EG_HD; d++) num[d] = 0;
            for (int16_t s = 0; s < nsel; s++) {
                int16_t w = s_selsc[s];
                const int8_t *vp = st->vc[li][s_selix[s]] + off;
                for (int16_t d = 0; d < EG_HD; d++)
                    num[d] += mul_ss(w, (int16_t)*vp++);
            }
            for (int16_t d = 0; d < EG_HD; d++)
                s_attn[off + d] = div_sat16(num[d], (int16_t)sum8);
        }
    }

#ifdef EG_STATS
    eg_stat_scan(&st->wo[li], s_attn, EG_EMBED, 0);
#endif
    eg_matvec(&st->wo[li], s_attn, s_proj, EG_EMBED, EG_EMBED);
    for (int16_t i = 0; i < EG_EMBED; i++)
        x[i] = sat16((int32_t)s_res[i] + s_proj[i]);

    /* FFN */
    memcpy(s_res, x, sizeof(s_res));
    eg_rmsnorm(x, EG_EMBED);
#ifdef EG_STATS
    eg_stat_scan(&st->wff1[li], x, EG_FFN, 0);
#endif
    eg_matvec(&st->wff1[li], x, s_ff, EG_EMBED, EG_FFN);
    for (int16_t i = 0; i < EG_FFN; i++)
        if (s_ff[i] < 0) s_ff[i] = 0;              /* ReLU */
#ifdef EG_STATS
    for (int16_t i = 0; i < EG_FFN; i++) {
        eg_st_ff_total++;
        if (s_ff[i] == 0) eg_st_ff_zero++;
    }
    if (!eg_wff2_t) eg_stat_scan(&st->wff2[li], s_ff, EG_EMBED, 1);
#endif
    if (eg_wff2_t)
        eg_matvec_t(&st->wff2[li], s_ff, s_proj, EG_FFN, EG_EMBED);
    else
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
        uint16_t w = (uint16_t)eg_streams;      /* 1 = bytes, 2 = words */
        for (uint16_t r = 0; r < rows; r++) {
            uint16_t na = rd16be(p);
            uint16_t ns = rd16be(p + 2);
            p += 4 + (uint32_t)(na + ns) * w;
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
        p = eg_scan_tensor(&st->wff2[li], p, EG_WFF2_ROWS,
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
        {
            uint16_t fl = rd16be(blob + 14);
            eg_streams = (fl & 8) ? 2 : ((fl & 2) ? 1 : 0);
            /* bit4 only means anything on top of bit3 (W16); the
             * transposed stream is defined in terms of W16 offsets. */
            eg_wff2_t  = (eg_streams == 2 && (fl & 0x10)) ? 1 : 0;
        }
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
    eg_streams = (blob[12] & 8) ? 2 : ((blob[12] & 2) ? 1 : 0);
    eg_wff2_t  = (eg_streams == 2 && (blob[12] & 0x10)) ? 1 : 0;
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
        rows[5] = EG_WFF2_ROWS;            /* wff2: EMBED rows, or FFN
                                            * columns when transposed */
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
        const int16_t *xv = st->x;
        /* 96 vocab entries x 64 dims per token. Two things matter on 68000:
         *  - unroll, so the CMP/Bcc pair is amortised 4 ways;
         *  - put the int8 embedding weight in the MULS.W *source* operand.
         *    MULS.W costs 38 + 2n cycles where n counts bit transitions in
         *    the SOURCE; a sign-extended int8 has far fewer than a full
         *    Q3.12 activation. Multiplication is commutative, so the
         *    products - and therefore the argmax - are unchanged. */
        for (int16_t v = 32; v <= 126; v++) {
            const int8_t *e = st->emb + (uint32_t)v * EG_EMBED;
            int32_t acc = 0;
            for (int16_t i = 0; i < EG_EMBED; i += 4) {
                acc += mul_ss(xv[i],     (int16_t)e[i]);
                acc += mul_ss(xv[i + 1], (int16_t)e[i + 1]);
                acc += mul_ss(xv[i + 2], (int16_t)e[i + 2]);
                acc += mul_ss(xv[i + 3], (int16_t)e[i + 3]);
            }
            if (acc > best) { best = acc; bestv = v; }
        }
        /* newline is a legal stop token too: compare its logit */
        {
            const int8_t *e = st->emb + 10u * EG_EMBED;
            int32_t acc = 0;
            for (int16_t i = 0; i < EG_EMBED; i += 4) {
                acc += mul_ss(xv[i],     (int16_t)e[i]);
                acc += mul_ss(xv[i + 1], (int16_t)e[i + 1]);
                acc += mul_ss(xv[i + 2], (int16_t)e[i + 2]);
                acc += mul_ss(xv[i + 3], (int16_t)e[i + 3]);
            }
            if (acc > best) bestv = '\n';
        }

        /* 5. advance the ring. No memmove: the oldest slot is simply
         * overwritten next token (attention is order-invariant here). */
        st->pos++;
        if (st->pos > 30000) st->pos = EG_CTX;   /* keep the counter sane */
        return (uint8_t)bestv;
    }
}
