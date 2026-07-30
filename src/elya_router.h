/* SPDX-License-Identifier: Apache-2.0
 * elya_router.h — GENERATED from train/experts.py. Do not edit.
 *
 * Deterministic keyword router in ROM. The learned linear router over
 * mean-pooled BYTE embeddings tops out near 60% at 8 experts, and got
 * WORSE (34%) with max-pooling added: max over byte embeddings is nearly
 * prompt-invariant because every English question shares the same common
 * letters, so each dimension saturates on spaces and vowels.
 *
 * These are the SAME plain substrings that produce the training labels,
 * so routing agrees with the shard split by construction. Runs once per
 * ANSWER, not per token.
 */
#ifndef ELYA_ROUTER_H
#define ELYA_ROUTER_H

#define ER_N_EXPERTS 8

static const char *const ER_KW0[] = {
    "how old are you", "are you old", "are you vintage", "when were you born", "what year were you born", "when is your birthday", "your origin", "born in 19", 0
};
static const char *const ER_KW1[] = {
    "who are you", "your name", "where are you from", "your purpose", "flameholder", "are you wise", "what do you love", "a secret", "who made you", "are you alive", "do you dream", "your creator", "victorian study", 0
};
static const char *const ER_KW2[] = {
    "zelda", "link", "ganon", "navi", "saria", "malon", "epona", "triforce", "master sword", "hyrule", "kokiri", "death mountain", "lon lon", "goron", "zora", "ocarina", "temple", "medallion", 0
};
static const char *const ER_KW3[] = {
    "dungeon", "proceed", "lurks", "need here", "help me", "encourage", "quest", "danger", "treasure", "monster", "boss", "realm", "where do i", "what should i", 0
};
static const char *const ER_KW4[] = {
    "rustchain", "rtc", "earn", "node", "epoch", "antiquity", "mining", "miner", "token", "wallet", "attest", "reward", "consensus", 0
};
static const char *const ER_KW5[] = {
    "amiga", "c64", "commodore", "apple ii", "atari", "the nes", "snes", "ti-99", "trs-80", "zx spectrum", "6502", "spectrum", "2600", 0
};
static const char *const ER_KW6[] = {
    "g4", "g5", "power8", "altivec", "vec_perm", "big-endian", "endian", "vr4300", "rsp", "rdp", "console", "powerpc", "mips", "68000", "genesis", "runs this", "expansion pak", "n64", "render", 0
};
static const char *const ER_KW7[] = {
    "quantization", "q4", "your model", "language runs you", "how big", "parameters", "neural", "transformer", 0
};

static const char *const *const ER_KW[ER_N_EXPERTS] = {
    ER_KW0, ER_KW1, ER_KW2, ER_KW3, ER_KW4, ER_KW5, ER_KW6, ER_KW7
};

static const char *const ER_NAMES[ER_N_EXPERTS] = {
    "origin", "identity", "zelda", "quest", "rustchain", "retro", "hardware", "meta"
};

#endif
