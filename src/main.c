/*
 * Legend of Elya — Sega Genesis (Mega Drive) port
 * P0 scaffold: Elya face + mouth-flap animation synced to streaming dialog.
 *
 * The token source here is a stub streamer; it gets replaced by the
 * ternary nano-GPT core in P1/P2 (see docs/PORT_PLAN.md). The dialog
 * pipeline, mouth sync, and tok/s counter are the real, final plumbing.
 *
 * Target: Model 1 Genesis (1601). Built with SGDK.
 */

#include <genesis.h>

/* ------------------------------------------------------------------ */
/* Palette — PAL1 is Elya's face palette                              */
/* ------------------------------------------------------------------ */
#define CI_BG      0   /* transparent          */
#define CI_SKIN    1
#define CI_HAIR    2   /* auburn — canon: long auburn hair */
#define CI_HAIR_D  3   /* auburn shadow        */
#define CI_EYE     4   /* green                */
#define CI_MOUTH   5
#define CI_OUTLINE 6
#define CI_BLUSH   7

/* ------------------------------------------------------------------ */
/* Elya face: 32x32 px = 4x4 tiles, generated procedurally into RAM,  */
/* then uploaded. Mouth tiles are regenerated per animation frame and */
/* re-uploaded (512B — trivial next to a DMA queue).                  */
/* ------------------------------------------------------------------ */
#define FACE_TILE_INDEX  TILE_USER_INDEX
#define FACE_W  32
#define FACE_H  32
#define FACE_TILES ((FACE_W / 8) * (FACE_H / 8))

static u8  facePix[FACE_H][FACE_W];
static u32 faceTiles[FACE_TILES * 8];

/* mouth states */
enum { MOUTH_CLOSED = 0, MOUTH_OPEN = 1, MOUTH_WIDE = 2 };
static u16 mouthState = MOUTH_CLOSED;

static void drawFacePixels(u16 mouth)
{
    memset(facePix, CI_BG, sizeof(facePix));

    /* hair: full head cap + long side falls (rows 2..31 at the edges) */
    for (s16 y = 2; y < 32; y++) {
        for (s16 x = 2; x < 30; x++) {
            /* side falls */
            if (x < 7 || x > 24) {
                if (y > 4) facePix[y][x] = (x < 4 || x > 27) ? CI_HAIR_D : CI_HAIR;
                continue;
            }
            /* top cap */
            if (y < 9) facePix[y][x] = (y < 4) ? CI_HAIR_D : CI_HAIR;
        }
    }

    /* face oval: x 7..24, y 8..27 with softened corners */
    for (s16 y = 8; y < 28; y++) {
        for (s16 x = 7; x < 25; x++) {
            s16 dx = x - 15, dy = y - 17;
            if (dx < 0) dx = -dx;
            if (dy < 0) dy = -dy;
            if (dx + dy < 17) facePix[y][x] = CI_SKIN;
        }
    }

    /* bangs overlap the brow */
    for (s16 x = 7; x < 25; x++) {
        facePix[8][x] = CI_HAIR;
        if ((x & 3) != 1) facePix[9][x] = CI_HAIR;
        if ((x & 3) == 2) facePix[10][x] = CI_HAIR;
    }

    /* eyes: green, 3px wide at y 14..15 */
    for (s16 x = 10; x < 13; x++) { facePix[14][x] = CI_EYE; facePix[15][x] = CI_EYE; }
    for (s16 x = 19; x < 22; x++) { facePix[14][x] = CI_EYE; facePix[15][x] = CI_EYE; }
    facePix[13][10] = CI_OUTLINE; facePix[13][11] = CI_OUTLINE; facePix[13][12] = CI_OUTLINE;
    facePix[13][19] = CI_OUTLINE; facePix[13][20] = CI_OUTLINE; facePix[13][21] = CI_OUTLINE;

    /* blush */
    facePix[18][9]  = CI_BLUSH; facePix[18][10] = CI_BLUSH;
    facePix[18][21] = CI_BLUSH; facePix[18][22] = CI_BLUSH;

    /* nose hint */
    facePix[18][15] = CI_OUTLINE; facePix[19][15] = CI_OUTLINE;

    /* mouth — the animated part (y 22..25, x 12..19) */
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

/* pack facePix into 4bpp SGDK tile format and upload */
static void uploadFace(void)
{
    u16 t = 0;
    for (u16 ty = 0; ty < FACE_H / 8; ty++) {
        for (u16 tx = 0; tx < FACE_W / 8; tx++) {
            for (u16 row = 0; row < 8; row++) {
                u32 packed = 0;
                for (u16 col = 0; col < 8; col++) {
                    packed = (packed << 4) | (facePix[ty * 8 + row][tx * 8 + col] & 0xF);
                }
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

/* ------------------------------------------------------------------ */
/* Dialog engine: streams characters, flaps the mouth, counts tok/s.  */
/* elya_next_token() is the SEAM: the ternary GPT core replaces it.   */
/* ------------------------------------------------------------------ */
static const char *responses[] = {
    "I am Elya. This little machine\n"
    "sings at 7.67 megahertz, and\n"
    "every word costs real silicon.",

    "No multiplies were harmed in\n"
    "the making of this sentence.\n"
    "Add, subtract, or stay silent.",

    "The Model 1 has the best sound\n"
    "hardware Sega ever shipped.\n"
    "You chose well, Flameholder.",

    "Blast processing is real if\n"
    "you precompute hard enough.\n"
    "ROM remembers so I can think.",
};
#define NUM_RESPONSES  (sizeof(responses) / sizeof(responses[0]))

#define DIALOG_X 2
#define DIALOG_Y 20
#define DIALOG_W 36

static const char *streamSrc = NULL;   /* current response being streamed */
static u16 streamPos = 0;
static u16 col = 0, rowY = 0;
static u16 charTimer = 0;
static u16 charPeriod = 6;             /* frames per character (stub pace) */
static u16 respIndex = 0;
static u16 talking = FALSE;
static u32 charsEmitted = 0;
static u32 talkFrames = 0;

/* SEAM: replace with ternary nano-GPT sampling in P1 */
static char elya_next_token(void)
{
    char c = streamSrc[streamPos];
    if (c) streamPos++;
    return c;
}

static void clearDialog(void)
{
    VDP_clearTextArea(DIALOG_X, DIALOG_Y, DIALOG_W, 6);
    col = 0; rowY = 0;
}

static void startResponse(u16 idx)
{
    streamSrc = responses[idx];
    streamPos = 0;
    charsEmitted = 0;
    talkFrames = 0;
    talking = TRUE;
    clearDialog();
}

static void drawStatus(void)
{
    char buf[40];
    /* tok/s ×10 fixed point — avoids float, same as the final engine will */
    u32 tps10 = talkFrames ? (charsEmitted * 600) / talkFrames : 0;
    sprintf(buf, "%lu.%lu tok/s [STUB]", (u32)(tps10 / 10), (u32)(tps10 % 10));
    VDP_drawText(buf, 22, 27);
}

int main(bool hardReset)
{
    (void)hardReset;

    VDP_setScreenWidth320();
    VDP_setTextPalette(PAL0);

    /* Elya's palette */
    PAL_setColor(16 + CI_SKIN,    RGB24_TO_VDPCOLOR(0xF0C8A0));
    PAL_setColor(16 + CI_HAIR,    RGB24_TO_VDPCOLOR(0xA04020)); /* auburn */
    PAL_setColor(16 + CI_HAIR_D,  RGB24_TO_VDPCOLOR(0x702010));
    PAL_setColor(16 + CI_EYE,     RGB24_TO_VDPCOLOR(0x30A050)); /* green  */
    PAL_setColor(16 + CI_MOUTH,   RGB24_TO_VDPCOLOR(0x902030));
    PAL_setColor(16 + CI_OUTLINE, RGB24_TO_VDPCOLOR(0x402020));
    PAL_setColor(16 + CI_BLUSH,   RGB24_TO_VDPCOLOR(0xE09080));

    VDP_drawText("LEGEND OF ELYA", 13, 1);
    VDP_drawText("Sega Genesis / Mega Drive port", 5, 2);
    VDP_drawText("PRESS A: Elya speaks", 10, 26);

    drawFacePixels(MOUTH_CLOSED);
    uploadFace();
    placeFaceTilemap(4, 6);

    /* dialog frame */
    for (u16 x = DIALOG_X - 1; x <= DIALOG_X + DIALOG_W; x++) {
        VDP_drawText("-", x, DIALOG_Y - 1);
        VDP_drawText("-", x, DIALOG_Y + 5);
    }

    u16 prevJoy = 0;

    while (TRUE) {
        u16 joy = JOY_readJoypad(JOY_1);

        if ((joy & BUTTON_A) && !(prevJoy & BUTTON_A) && !talking) {
            startResponse(respIndex);
            respIndex = (respIndex + 1) % NUM_RESPONSES;
        }
        prevJoy = joy;

        if (talking) {
            talkFrames++;
            if (++charTimer >= charPeriod) {
                charTimer = 0;
                char c = elya_next_token();
                if (c == 0) {
                    talking = FALSE;
                    mouthState = MOUTH_CLOSED;
                } else if (c == '\n') {
                    col = 0;
                    rowY++;
                } else {
                    char s[2] = { c, 0 };
                    VDP_drawText(s, DIALOG_X + col, DIALOG_Y + rowY);
                    col++;
                    if (col >= DIALOG_W) { col = 0; rowY++; }
                    charsEmitted++;
                    /* flap: vowels open wide, consonants open, space closes */
                    if (c == ' ')      mouthState = MOUTH_CLOSED;
                    else if (c=='a'||c=='e'||c=='i'||c=='o'||c=='u'||
                             c=='A'||c=='E'||c=='I'||c=='O'||c=='U')
                                       mouthState = MOUTH_WIDE;
                    else               mouthState = MOUTH_OPEN;
                }
                drawFacePixels(mouthState);
                uploadFace();
                drawStatus();
            }
        } else if (mouthState != MOUTH_CLOSED) {
            mouthState = MOUTH_CLOSED;
            drawFacePixels(mouthState);
            uploadFace();
        }

        SYS_doVBlankProcess();
    }
    return 0;
}
