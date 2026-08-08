/* SPDX-License-Identifier: Apache-2.0
 * host_bench.c — x86 mirror of the Genesis bench ROM's loop.
 *
 * Runs EXACTLY what src/bench_main.c runs on the 68000: route the prompt,
 * feed all 14 prompt characters, then generate a FIXED 24 tokens with no
 * early stop on newline. That makes it the token oracle for the cycle
 * bench: if this prints the gate string, the ROM must too.
 *
 * Build:  gcc -O2 -o host_bench host/host_bench.c src/elya_gpt.c
 * Stats:  gcc -O2 -DEG_STATS -o host_bench_stats host/host_bench.c \
 *              src/elya_gpt.c
 */
#include <stdio.h>
#include <stdlib.h>
#include "../src/elya_gpt.h"

#define BENCH_PROMPT "Who are you?: "
#define BENCH_NTOK   24

static EgState st;

#ifdef EG_STATS
void eg_stats_report(const char *what);
#endif

int main(int argc, char **argv)
{
    const char *path = (argc > 1) ? argv[1] : "res/elya_brain_w16.bin";
    FILE *f = fopen(path, "rb");
    if (!f) { perror(path); return 1; }
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    uint8_t *blob = malloc((size_t)sz);
    if (!blob || fread(blob, 1, (size_t)sz, f) != (size_t)sz) {
        perror("read"); return 1;
    }
    fclose(f);

    int rc = eg_init(&st, blob);
    if (rc) { fprintf(stderr, "eg_init failed: %d\n", rc); return 1; }

    uint16_t e = eg_route(&st, BENCH_PROMPT);
    eg_select_expert(&st, e);
    eg_reset(&st);

    uint8_t tok = 0;
    for (const char *p = BENCH_PROMPT; *p; p++)
        tok = eg_next_token(&st, (uint8_t)*p);

    uint8_t out[BENCH_NTOK];
    for (int i = 0; i < BENCH_NTOK; i++) {
        out[i] = tok;
        tok = eg_next_token(&st, tok);
    }

    printf("expert=%u\n", e);
    printf("ntok=%d\n", BENCH_NTOK);
    printf("tokens=");
    for (int i = 0; i < BENCH_NTOK; i++)
        printf("%s%d", i ? "," : "", out[i]);
    printf("\ntext=");
    for (int i = 0; i < BENCH_NTOK; i++) putchar(out[i]);
    printf("\n");

#ifdef EG_STATS
    eg_stats_report("38 forward passes (14 prompt + 24 generated)");
#endif
    return 0;
}
