/*
 * ELYA INTO DREAMS — Sega Genesis / Mega Drive
 * World-first: on-console transformer inference on the Motorola 68000.
 *
 * P1: real integer-only nano-GPT (src/elya_gpt.c) replaces the P0 stub.
 * Ternary weights live in cartridge ROM (res/elya_genesis.bin, SGT1).
 * Elya's mouth animates per generated token; tok/s counter is genuine.
 *
 * Target: Model 1 Genesis (1601). Built with marsdev / SGDK 1.81.
 */

#include <genesis.h>
#include "elya_gpt.h"
#include "resources.h"

/* ------------------------------------------------------------------ */
/* Palette — PAL1 is Elya's face palette                              */
/* ------------------------------------------------------------------ */
#define CI_BG      0
#define CI_SKIN    1
#define CI_HAIR    2   /* auburn — canon: long auburn hair */
#define CI_HAIR_D  3
#define CI_EYE     4
#define CI_MOUTH   5
#define CI_OUTLINE 6
#define CI_BLUSH   7

/* ------------------------------------------------------------------ */
/* Elya face: 32x32 px = 4x4 tiles, procedural (art pass comes later) */
/* ------------------------------------------------------------------ */
#define FACE_TILE_INDEX  TILE_USER_INDEX
#define FACE_W  32
#define FACE_H  32
#define FACE_TILES ((FACE_W / 8) * (FACE_H / 8))

static u8  facePix[FACE_H][FACE_W];
static u32 faceTiles[FACE_TILES * 8];

enum { MOUTH_CLOSED = 0, MOUTH_OPEN = 1, MOUTH_WIDE = 2 };
static u16 mouthState = MOUTH_CLOSED;

static void drawFacePixels(u16 mouth)
{
    memset(facePix, CI_BG, sizeof(facePix));

    for (s16 y = 2; y < 32; y++) {
        for (s16 x = 2; x < 30; x++) {
            if (x < 7 || x > 24) {
                if (y > 4) facePix[y][x] = (x < 4 || x > 27) ? CI_HAIR_D : CI_HAIR;
                continue;
            }
            if (y < 9) facePix[y][x] = (y < 4) ? CI_HAIR_D : CI_HAIR;
        }
    }
    for (s16 y = 8; y < 28; y++) {
        for (s16 x = 7; x < 25; x++) {
            s16 dx = x - 15, dy = y - 17;
            if (dx < 0) dx = -dx;
            if (dy < 0) dy = -dy;
            if (dx + dy < 17) facePix[y][x] = CI_SKIN;
        }
    }
    for (s16 x = 7; x < 25; x++) {
        facePix[8][x] = CI_HAIR;
        if ((x & 3) != 1) facePix[9][x] = CI_HAIR;
        if ((x & 3) == 2) facePix[10][x] = CI_HAIR;
    }
    for (s16 x = 10; x < 13; x++) { facePix[14][x] = CI_EYE; facePix[15][x] = CI_EYE; }
    for (s16 x = 19; x < 22; x++) { facePix[14][x] = CI_EYE; facePix[15][x] = CI_EYE; }
    facePix[13][10] = CI_OUTLINE; facePix[13][11] = CI_OUTLINE; facePix[13][12] = CI_OUTLINE;
    facePix[13][19] = CI_OUTLINE; facePix[13][20] = CI_OUTLINE; facePix[13][21] = CI_OUTLINE;
    facePix[18][9]  = CI_BLUSH; facePix[18][10] = CI_BLUSH;
    facePix[18][21] = CI_BLUSH; facePix[18][22] = CI_BLUSH;
    facePix[18][15] = CI_OUTLINE; facePix[19][15] = CI_OUTLINE;

    switch (mouth) {
        case MOUTH_CLOSED:
            for (s16 x = 13; x < 19; x++) facePix[23][x] = CI_MOUTH;
            break;
        case MOUTH_OPEN:
            for (s16 x = 13; x < 19; x++) {
                facePix[22][x] = CI_MOUTH;
                facePix[23][x] = CI_MOUTH;
            }
            facePix[22][13] = CI_OUTLINE; facePix[22][18] = CI_OUTLINE;
            break;
        case MOUTH_WIDE:
            for (s16 x = 12; x < 20; x++) {
                facePix[22][x] = CI_MOUTH;
                facePix[23][x] = CI_MOUTH;
                facePix[24][x] = CI_MOUTH;
            }
            facePix[22][12] = CI_OUTLINE; facePix[22][19] = CI_OUTLINE;
            facePix[24][12] = CI_OUTLINE; facePix[24][19] = CI_OUTLINE;
            break;
    }
}

static void uploadFace(void)
{
    u16 t = 0;
    for (u16 ty = 0; ty < FACE_H / 8; ty++) {
        for (u16 tx = 0; tx < FACE_W / 8; tx++) {
            for (u16 row = 0; row < 8; row++) {
                u32 packed = 0;
                for (u16 col = 0; col < 8; col++)
                    packed = (packed << 4) | (facePix[ty * 8 + row][tx * 8 + col] & 0xF);
                faceTiles[t * 8 + row] = packed;
            }
            t++;
        }
    }
    VDP_loadTileData(faceTiles, FACE_TILE_INDEX, FACE_TILES, DMA_QUEUE);
}

static void placeFaceTilemap(u16 px, u16 py)
{
    u16 t = 0;
    for (u16 ty = 0; ty < FACE_H / 8; ty++)
        for (u16 tx = 0; tx < FACE_W / 8; tx++)
            VDP_setTileMapXY(BG_A,
                TILE_ATTR_FULL(PAL1, TRUE, FALSE, FALSE, FACE_TILE_INDEX + t++),
                px + tx, py + ty);
}

static void setMouth(u16 m)
{
    if (m != mouthState) {
        mouthState = m;
        drawFacePixels(m);
        uploadFace();
    }
}

/* ------------------------------------------------------------------ */
/* Inference-driven dialog                                            */
/* ------------------------------------------------------------------ */
static const char *prompts[] = {
    "Who are you?: ",
    "What is your name?: ",
    "Where are you from?: ",
    "What is your purpose?: ",
    "What is RustChain?: ",
    "Tell me of the realm.: ",
};
#define NUM_PROMPTS (sizeof(prompts) / sizeof(prompts[0]))

#define DIALOG_X 2
#define DIALOG_Y 20
#define DIALOG_W 36
#define DIALOG_H 5
#define MAX_GEN  100

static EgState elya;                 /* ~17KB static (KV cache inside) */

enum { ST_IDLE, ST_FEED, ST_GEN };
static u16 mode = ST_IDLE;
static u16 promptIdx = 0;
static const char *curPrompt = NULL;
static u16 feedPos = 0;
static u8  lastTok = 0;
static u16 genCount = 0;
static u16 col = 0, rowY = 0;
static u32 genStart = 0, genTokens = 0;
static u16 engineOk = FALSE;

static void clearDialog(void)
{
    VDP_clearTextArea(DIALOG_X, DIALOG_Y, DIALOG_W, DIALOG_H);
    col = 0; rowY = 0;
}

static void putGlyph(char c)
{
    if (c == '\n') { col = 0; rowY++; return; }
    char s[2] = { c, 0 };
    if (rowY < DIALOG_H) VDP_drawText(s, DIALOG_X + col, DIALOG_Y + rowY);
    col++;
    if (col >= DIALOG_W) { col = 0; rowY++; }
}

static void drawTokSpeed(void)
{
    char buf[24];
    /* x100 fixed point: vblank frames @60Hz -> tok/s (honest wall-clock) */
    u32 elapsed = vtimer - genStart;
    u32 t100 = elapsed ? (genTokens * 6000) / elapsed : 0;
    sprintf(buf, "%lu.%02lu tok/s", (u32)(t100 / 100), (u32)(t100 % 100));
    VDP_drawText(buf, 26, 27);
}

static void startPrompt(u16 idx)
{
    eg_reset(&elya);
    curPrompt = prompts[idx];
    feedPos = 0;
    genCount = 0;
    genStart = vtimer;
    genTokens = 0;
    mode = ST_FEED;
    clearDialog();
    VDP_drawText("Elya dreams", 2, 27);
    /* echo the question in the dialog box */
    for (const char *p = curPrompt; *p; p++) putGlyph(*p);
    putGlyph('\n');
}

int main(bool hardReset)
{
    (void)hardReset;

    VDP_setScreenWidth320();
    VDP_setTextPalette(PAL0);

    PAL_setColor(16 + CI_SKIN,    RGB24_TO_VDPCOLOR(0xF0C8A0));
    PAL_setColor(16 + CI_HAIR,    RGB24_TO_VDPCOLOR(0xA04020));
    PAL_setColor(16 + CI_HAIR_D,  RGB24_TO_VDPCOLOR(0x702010));
    PAL_setColor(16 + CI_EYE,     RGB24_TO_VDPCOLOR(0x30A050));
    PAL_setColor(16 + CI_MOUTH,   RGB24_TO_VDPCOLOR(0x902030));
    PAL_setColor(16 + CI_OUTLINE, RGB24_TO_VDPCOLOR(0x402020));
    PAL_setColor(16 + CI_BLUSH,   RGB24_TO_VDPCOLOR(0xE09080));

    VDP_drawText("ELYA INTO DREAMS", 12, 1);
    VDP_drawText("A transformer dreams on the 68000", 3, 2);

    if (eg_init(&elya, (const uint8_t *)elya_weights) == 0) {
        engineOk = TRUE;
        VDP_drawText("PRESS A: ask Elya", 11, 26);
    } else {
        VDP_drawText("WEIGHTS MISSING - STUB ROM", 7, 26);
    }

    drawFacePixels(MOUTH_CLOSED);
    uploadFace();
    placeFaceTilemap(4, 6);

    for (u16 x = DIALOG_X - 1; x <= DIALOG_X + DIALOG_W; x++) {
        VDP_drawText("-", x, DIALOG_Y - 1);
        VDP_drawText("-", x, DIALOG_Y + DIALOG_H);
    }

    u16 prevJoy = 0;
    u16 blink = 0;

    while (TRUE) {
        u16 joy = JOY_readJoypad(JOY_1);

        if ((joy & BUTTON_A) && !(prevJoy & BUTTON_A)
            && mode == ST_IDLE && engineOk) {
            startPrompt(promptIdx);
            promptIdx = (promptIdx + 1) % NUM_PROMPTS;
        }
        prevJoy = joy;

        switch (mode) {
        case ST_FEED:
            /* one forward pass per loop — display stays alive between */
            lastTok = eg_next_token(&elya, (u8)curPrompt[feedPos]);
            feedPos++;
            genTokens++;
            setMouth((++blink & 1) ? MOUTH_OPEN : MOUTH_CLOSED); /* pondering */
            if (curPrompt[feedPos] == 0) {
                mode = ST_GEN;
                genCount = 0;
            }
            drawTokSpeed();
            break;

        case ST_GEN:
            if (lastTok == '\n' || genCount >= MAX_GEN
                || rowY >= DIALOG_H) {
                mode = ST_IDLE;
                setMouth(MOUTH_CLOSED);
                VDP_drawText("PRESS A: ask Elya", 2, 27);
                break;
            }
            putGlyph((char)lastTok);
            genCount++;
            genTokens++;
            /* vowels open wide — she speaks as she thinks */
            {
                char c = (char)lastTok;
                if (c == ' ')      setMouth(MOUTH_CLOSED);
                else if (c=='a'||c=='e'||c=='i'||c=='o'||c=='u'||
                         c=='A'||c=='E'||c=='I'||c=='O'||c=='U')
                                   setMouth(MOUTH_WIDE);
                else               setMouth(MOUTH_OPEN);
            }
            drawTokSpeed();
            lastTok = eg_next_token(&elya, lastTok);
            break;

        default:
            break;
        }

        SYS_doVBlankProcess();
    }
    return 0;
}
