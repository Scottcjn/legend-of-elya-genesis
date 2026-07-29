/* Host-side harness for elya_gpt.c — same engine source the 68K runs.
 * Usage: harness <blob.bin> "<prompt>" [max_tokens]
 * Prints the greedy generation to stdout (raw, ends at newline token).
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../src/elya_gpt.h"

static EgState st;

int main(int argc, char **argv)
{
    if (argc < 3) { fprintf(stderr, "usage: %s blob prompt [n]\n", argv[0]); return 2; }
    int maxn = (argc > 3) ? atoi(argv[3]) : 60;

    FILE *f = fopen(argv[1], "rb");
    if (!f) { perror("blob"); return 1; }
    fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
    uint8_t *blob = malloc(sz);
    if (fread(blob, 1, sz, f) != (size_t)sz) { perror("read"); return 1; }
    fclose(f);

    int rc = eg_init(&st, blob);
    if (rc) { fprintf(stderr, "eg_init failed: %d\n", rc); return 1; }
    eg_reset(&st);

    const char *prompt = argv[2];
    uint8_t tok = 0;
    for (const char *p = prompt; *p; p++)
        tok = eg_next_token(&st, (uint8_t)*p);

    for (int i = 0; i < maxn; i++) {
        putchar(tok);
        if (tok == '\n') break;
        tok = eg_next_token(&st, tok);
    }
    putchar('\n');
    return 0;
}
