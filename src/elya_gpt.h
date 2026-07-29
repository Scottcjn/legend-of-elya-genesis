/* SPDX-License-Identifier: Apache-2.0
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

/* PSE-style attention collapse: only the K strongest context positions
 * survive to the softmax + V mixdown. EG_TOPK == EG_CTX disables it. */
#ifndef EG_TOPK
#define EG_TOPK   EG_CTX
#endif

/* Weight layout: 1 = SGT2 index streams (fast, ~2.6x ROM),
 *                0 = SGT1 2-bit packed (compact, slower).
 * The blob's magic must match: SGT2 vs SGT1. */
#ifndef EG_FORMAT_INDEX_STREAMS
#define EG_FORMAT_INDEX_STREAMS 1
#endif

typedef struct {
    const uint8_t *packed;   /* 2-bit ternary weights, row-major [out][in] */
    uint16_t M;              /* requant multiplier (<=127)                  */
    uint8_t  S;              /* requant shift                               */
} EgTensor;

/* Lock-On MoE: experts live in cartridge ROM and are "activated" by
 * repointing these tensors (plus a mapper bank write past 4MB). Nothing
 * is copied — ROM is memory-mapped, so activation is free.
 * See docs/LOCKON_MOE.md. */
#define EG_MAX_EXPERTS 16

typedef struct {
    const int8_t  *emb;      /* vocab x embed, Q2.6 */
    const int8_t  *pe;       /* ctx x embed, Q2.6 — NULL if the blob has
                              * no positional encoding (SGT2)          */
    const uint8_t *explut;   /* u16be[256], Q14     */
    EgTensor router;         /* embed -> n_experts, ternary          */
    uint16_t n_experts;
    uint16_t expert;         /* currently activated expert           */
    const uint8_t *expert_base[EG_MAX_EXPERTS];
    EgTensor wq[EG_LAYERS], wk[EG_LAYERS], wv[EG_LAYERS];
    EgTensor wo[EG_LAYERS], wff1[EG_LAYERS], wff2[EG_LAYERS];

    int8_t  kc[EG_LAYERS][EG_CTX][EG_EMBED];  /* K cache, Q4.4 */
    int8_t  vc[EG_LAYERS][EG_CTX][EG_EMBED];  /* V cache, Q4.4 */
    int16_t x[EG_EMBED];                      /* hidden state, Q3.12 */
    int16_t pos;
    int16_t ok;                               /* header validated */
} EgState;

/* Returns 0 on success, negative on bad blob. Accepts SGT2 (single
 * model) and SGTM (Lock-On MoE); a single model is treated as 1 expert. */
int  eg_init(EgState *st, const uint8_t *blob);
void eg_reset(EgState *st);
/* Feed one input byte, get greedy next-token prediction (printable ASCII). */
uint8_t eg_next_token(EgState *st, uint8_t input);

/* Pick the expert for a prompt (mean-pooled embeddings -> ternary
 * classifier). Returns the expert id; does not activate it. */
uint16_t eg_route(EgState *st, const char *prompt);
/* Activate an expert: repoints the weight tensors. No data is copied. */
void eg_select_expert(EgState *st, uint16_t expert);

#endif
