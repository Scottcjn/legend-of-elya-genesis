/* SPDX-License-Identifier: Apache-2.0
 * bench_main.c — headless cycle bench for the Elya Genesis engine.
 *
 * This project's src/ SYMLINKS the real src/elya_gpt.c, so one engine edit
 * is measured here and shipped from the same file. Nothing about the model
 * or the arithmetic differs from the game ROM; only the shell does.
 *
 * Determinism, which is the whole point:
 *  - the VDP display is switched OFF and interrupts are disabled, so no
 *    VBlank IRQ and no VDP refresh contend for the 68000 bus inside the
 *    measured region;
 *  - nothing is drawn, nothing is polled, nothing waits on a frame.
 *
 * Protocol with tools/mame/bench_run.py (a MAME write tap on bench_marker):
 *   bench_marker = 1          start of the measured region
 *   bench_marker = 0x100 + n  progress mark before forward pass n
 *   bench_marker = 2          end of the measured region
 *   then bench_expert / bench_tok[] / bench_ntok / bench_status are filled
 *   bench_marker = 4          "results are readable, dump and exit"
 * The 68000 splits the move.l into two word writes; only the low word
 * (marker+2) carries the value, which is what the Lua tap keys on.
 */
#include <genesis.h>
#include "elya_gpt.h"
#include "bench.h"

#define BENCH_PROMPT "Who are you?: "
#define BENCH_NTOK   24

/* Named, in work RAM, so the runner reads them out of out/symbol.txt and
 * never hard-codes an address. */
volatile u32 bench_marker;
volatile u16 bench_expert;
volatile u8  bench_status;
volatile u8  bench_ntok;
volatile u8  bench_tok[64];

static EgState elya;

int main(void)
{
    /* Kill every source of bus contention and timing jitter. */
    SYS_disableInts();
    VDP_setEnable(FALSE);

    bench_status = 0;
    bench_ntok   = 0;
    bench_expert = 0xFFFF;

    if (eg_init(&elya, (const u8 *)elya_weights) != 0) {
        bench_status = 0xE1;
        bench_marker = 4;
        for (;;) ;
    }

    u16 e = eg_route(&elya, BENCH_PROMPT);
    eg_select_expert(&elya, e);
    eg_reset(&elya);

    u8 tok = 0;
    u16 pass = 0;

    bench_marker = 1;                       /* ---- START ---- */

    for (const char *p = BENCH_PROMPT; *p; p++) {
        bench_marker = 0x100u + pass++;
        tok = eg_next_token(&elya, (u8)*p);
    }
    for (u16 i = 0; i < BENCH_NTOK; i++) {
        bench_marker = 0x100u + pass++;
        bench_tok[i] = tok;
        tok = eg_next_token(&elya, tok);
    }

    bench_marker = 2;                       /* ---- END ---- */

    bench_expert = e;
    bench_ntok   = BENCH_NTOK;
    bench_status = 0x5A;
    bench_marker = 4;                       /* results readable */

    for (;;) ;
    return 0;
}
