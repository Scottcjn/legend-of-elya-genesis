/* SPDX-License-Identifier: AGPL-3.0
 * elya_gpt.h — Elya into Dreams: integer-only nano-GPT for the 68000.
 *
 * Ternary weights (2-bit packed, ROM), int16 Q3.12 activations,
 * int8 Q4.4 KV cache, exp() via ROM LUT. No floats anywhere.
 * Portable C99: same source builds for x86 host harness and m68k.
 *
 * Weight blob format "SGT1" (all multi-byte fields BIG-endian, 68K-native):
 *   header(12): 'SGT1' u8 layers u8 heads u16 embed u16 vocab u16 ctx
 *               u8 flags u8 pad
 *   emb: int8[vocab*embed]                 (Q2.6, tied logits)
 *   per layer, order wq wk wv wo wff1 wff2:
 *     u16 M, u8 S, u8 pad, then 2-bit packed weights [out][in] MSB-first
 *     codes: 00=0 01=+1 10=-1 11=reserved (tetranary +/-2 extension)
 *   explut: u16[256]  Q14 of exp(-i/16)
 */
#ifndef ELYA_GPT_H
#define ELYA_GPT_H

#ifdef SGDK_GCC
#include <genesis.h>   /* SGDK maps int8_t & co. onto its own s8/u8 types */
#else
#include <stdint.h>
#endif

#define EG_LAYERS 2
#define EG_EMBED  64
#define EG_HEADS  4
#define EG_HD     (EG_EMBED / EG_HEADS)
#define EG_VOCAB  256
#define EG_CTX    64
#define EG_FFN    (EG_EMBED * 4)

typedef struct {
    const uint8_t *packed;   /* 2-bit ternary weights, row-major [out][in] */
    uint16_t M;              /* requant multiplier (<=127)                  */
    uint8_t  S;              /* requant shift                               */
} EgTensor;

typedef struct {
    const int8_t  *emb;      /* vocab x embed, Q2.6 */
    const uint8_t *explut;   /* u16be[256], Q14     */
    EgTensor wq[EG_LAYERS], wk[EG_LAYERS], wv[EG_LAYERS];
    EgTensor wo[EG_LAYERS], wff1[EG_LAYERS], wff2[EG_LAYERS];

    int8_t  kc[EG_LAYERS][EG_CTX][EG_EMBED];  /* K cache, Q4.4 */
    int8_t  vc[EG_LAYERS][EG_CTX][EG_EMBED];  /* V cache, Q4.4 */
    int16_t x[EG_EMBED];                      /* hidden state, Q3.12 */
    int16_t pos;
    int16_t ok;                               /* header validated */
} EgState;

/* Returns 0 on success, negative on bad blob. */
int  eg_init(EgState *st, const uint8_t *blob);
void eg_reset(EgState *st);
/* Feed one input byte, get greedy next-token prediction (printable ASCII). */
uint8_t eg_next_token(EgState *st, uint8_t input);

#endif
