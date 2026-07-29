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
    SYS_disableInts();
    VDP_loadTileData(faceTiles, FACE_TILE_INDEX, FACE_TILES, DMA_QUEUE);
    SYS_enableInts();
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
/* The Dreamscape — fake-2D on BG_B (PAL2)                            */
/* Starfield + crescent moon above, perspective checkerboard floor    */
/* below via per-scanline hscroll (the Space Harrier trick). Animated */
/* from the VBlank interrupt so the dream keeps moving even while the */
/* 68000 is deep inside a forward pass (T7 doctrine).                 */
/* ------------------------------------------------------------------ */
#define DS_TILE_INDEX  (FACE_TILE_INDEX + FACE_TILES)
/* tile roles (offset from DS_TILE_INDEX) */
#define DT_STAR1   0
#define DT_STAR2   1
#define DT_CHECK_A 2
#define DT_CHECK_B 3
#define DT_HORIZ1  4
#define DT_HORIZ2  5
#define DT_MOON    6   /* 6..9: 2x2 crescent */
#define DS_NTILES  10

#define HORIZON_ROW 14          /* tile row where the floor begins    */
#define FLOOR_ROW   16
#define FLOOR_ROWS  12

/* dreamscape colors, PAL2 */
#define DC_WHITE  1
#define DC_CYAN   2
#define DC_GOLD   3
#define DC_CHK_A  4
#define DC_CHK_B  5
#define DC_GLOW1  6
#define DC_GLOW2  7

/* the Mind Window lives further down but is driven from the VBlank
 * callback and from init, both of which appear before it */
static void panelWindow(u16 x, u16 y, u16 w, u16 h, const char *title);
static void panelClear(u16 x, u16 y, u16 w, u16 h);
static void runInit(void);
static void runAnimate(u16 frame);
static void mwInit(void);
static void mwAnimate(u16 frame);
static void mwSetExpert(u16 e);

static u32 dsTiles[DS_NTILES * 8];
static vu16 dsFrame = 0;
static s16  dsRowPos[FLOOR_ROWS];   /* incremental scroll accumulators */
static s16 dsScroll[FLOOR_ROWS * 8];

static void dsSetPix(u16 tile, u16 x, u16 y, u16 c)
{
    u32 *row = &dsTiles[tile * 8 + y];
    u16 shift = (7 - x) * 4;
    *row = (*row & ~((u32)0xF << shift)) | ((u32)c << shift);
}

static void buildDreamTiles(void)
{
    memset(dsTiles, 0, sizeof(dsTiles));

    /* star 1: single twinkle dot */
    dsSetPix(DT_STAR1, 3, 3, DC_WHITE);
    /* star 2: small plus */
    dsSetPix(DT_STAR2, 5, 4, DC_CYAN);
    dsSetPix(DT_STAR2, 5, 6, DC_CYAN);
    dsSetPix(DT_STAR2, 4, 5, DC_CYAN);
    dsSetPix(DT_STAR2, 6, 5, DC_CYAN);
    dsSetPix(DT_STAR2, 5, 5, DC_WHITE);

    /* solid checker tiles */
    for (u16 y = 0; y < 8; y++)
        for (u16 x = 0; x < 8; x++) {
            dsSetPix(DT_CHECK_A, x, y, DC_CHK_A);
            dsSetPix(DT_CHECK_B, x, y, DC_CHK_B);
        }

    /* horizon glow bands */
    for (u16 y = 0; y < 8; y++)
        for (u16 x = 0; x < 8; x++) {
            dsSetPix(DT_HORIZ1, x, y, (y < 5) ? 0 : DC_GLOW2);
            dsSetPix(DT_HORIZ2, x, y, (y < 3) ? DC_GLOW2 : DC_GLOW1);
        }

    /* crescent moon on a 16x16 grid across 4 tiles (2x2) */
    for (s16 y = 0; y < 16; y++) {
        for (s16 x = 0; x < 16; x++) {
            s16 dx = x - 8, dy = y - 8;
            s16 ex = x - 11, ey = y - 7;   /* bite offset */
            bool in  = (dx * dx + dy * dy) < 49;
            bool out = (ex * ex + ey * ey) < 36;
            if (in && !out) {
                u16 t = DT_MOON + ((y >> 3) << 1) + (x >> 3);
                dsSetPix(t, x & 7, y & 7, DC_GOLD);
            }
        }
    }

    VDP_loadTileData(dsTiles, DS_TILE_INDEX, DS_NTILES, DMA);
}

static void buildDreamMap(void)
{
    /* sky: sparse pseudo-random stars (deterministic hash scatter) */
    for (u16 y = 0; y < HORIZON_ROW; y++) {
        for (u16 x = 0; x < 64; x++) {
            u16 h = (u16)(x * 31 + y * 17 + ((x * y) >> 2));
            u16 t = 0;
            if ((h & 31) == 0)      t = DS_TILE_INDEX + DT_STAR1;
            else if ((h & 63) == 9) t = DS_TILE_INDEX + DT_STAR2;
            if (t)
                VDP_setTileMapXY(BG_B,
                    TILE_ATTR_FULL(PAL2, FALSE, FALSE, FALSE, t), x, y);
        }
    }

    /* moon, top right */
    for (u16 dy = 0; dy < 2; dy++)
        for (u16 dx = 0; dx < 2; dx++)
            VDP_setTileMapXY(BG_B,
                TILE_ATTR_FULL(PAL2, FALSE, FALSE, FALSE,
                               DS_TILE_INDEX + DT_MOON + dy * 2 + dx),
                33 + dx, 2 + dy);

    /* horizon glow */
    for (u16 x = 0; x < 64; x++) {
        VDP_setTileMapXY(BG_B, TILE_ATTR_FULL(PAL2, FALSE, FALSE, FALSE,
                         DS_TILE_INDEX + DT_HORIZ1), x, HORIZON_ROW);
        VDP_setTileMapXY(BG_B, TILE_ATTR_FULL(PAL2, FALSE, FALSE, FALSE,
                         DS_TILE_INDEX + DT_HORIZ2), x, HORIZON_ROW + 1);
    }

    /* checkerboard floor: checks widen toward the viewer (fake depth) */
    for (u16 r = 0; r < FLOOR_ROWS; r++) {
        u16 w = (r < 4) ? 0 : (r < 8) ? 1 : 2;   /* 1-, 2-, 4-tile checks */
        u16 g = r >> 2;
        for (u16 x = 0; x < 64; x++) {
            u16 a = (((x >> w) + g) & 1);
            VDP_setTileMapXY(BG_B,
                TILE_ATTR_FULL(PAL2, FALSE, FALSE, FALSE,
                    DS_TILE_INDEX + (a ? DT_CHECK_A : DT_CHECK_B)),
                x, FLOOR_ROW + r);
        }
    }
}

/* ------------------------------------------------------------------ */
/* TOKEN RAIN — her thoughts falling through the dream.                */
/* Columns of glyphs drift down BG_B at per-column speeds via          */
/* VSCROLL_COLUMN (20 two-cell entries in H40). The tilemap is filled  */
/* once and never rewritten; ALL motion is VSRAM writes, so it costs   */
/* no VRAM and no DMA. When Elya emits a token, that ACTUAL character  */
/* is written into the top of a column: the background is made of her  */
/* own output, raining down behind her.                                */
/* ------------------------------------------------------------------ */
#define RAIN_COLS 16                 /* power of 2: no division anywhere */
static s16 rainPos[RAIN_COLS];       /* Q12.4 accumulators */
static s16 rainVS[RAIN_COLS];
static const s16 rainSpd[RAIN_COLS] = { 3,7,2,5,11,4,9,6,13,3,8,2,10,5,7,4 };
static u16 rainCol = 0;

static void rainInit(void)
{
    /* BG_A carries her face, the dialog and the text — it must NOT move.
     * VSCROLL_COLUMN is global to both planes, so zero A's entries once. */
    static s16 zeroVS[20];
    memset(zeroVS, 0, sizeof(zeroVS));
    VDP_setVerticalScrollTile(BG_A, 0, zeroVS, 20, CPU);

    /* seed the columns with faint glyphs so the rain exists before she
     * has said anything (PAL0 = the dim text palette) */
    for (u16 c = 0; c < RAIN_COLS; c++) {
        for (u16 row = 0; row < 28; row++) {
            u16 h = (u16)(c * 37 + row * 17);
            if ((h & 7) != 0) continue;
            u16 t = TILE_FONT_INDEX + (h % 60);
            VDP_setTileMapXY(BG_B, TILE_ATTR_FULL(PAL0, FALSE, FALSE, FALSE, t),
                             c * 2, row);
        }
    }
}

/* called once per generated token — the dream rains what she just said.
 *
 * SGDK's VDP helpers are NOT reentrant: each sets a VDP address and then
 * writes data. If the VBlank callback fires between those two steps it
 * retargets the VDP, and this write lands somewhere arbitrary — VRAM
 * corruption or a lockup. Every main-loop VDP access in this file is
 * therefore guarded. (Learned the hard way: the ROM crashed.) */
static void rainDropToken(u8 tok)
{
    if (tok < 32 || tok > 126) return;
    u16 row = (u16)((0 - (rainPos[rainCol] >> 7)) - 1) & 31;
    u16 t = TILE_FONT_INDEX + (tok - 32);
    SYS_disableInts();
    VDP_setTileMapXY(BG_B, TILE_ATTR_FULL(PAL1, FALSE, FALSE, FALSE, t),
                     rainCol * 2, row);
    VDP_setTileMapXY(BG_B, TILE_ATTR_FULL(PAL1, FALSE, FALSE, FALSE, t),
                     rainCol * 2 + 1, row);
    SYS_enableInts();
    rainCol = (rainCol + 7) & (RAIN_COLS - 1);   /* stride 7, never adjacent */
}

/* runs every vblank — the dream animates even while Elya thinks */
static void dreamscapeVInt(void)
{
    dsFrame++;

    mwAnimate((u16)dsFrame);
    runAnimate((u16)dsFrame);      /* the mind keeps ticking while she thinks */

    /* Floor: lower scanlines scroll faster = perspective illusion.
     * dsFrame is u16 and the accumulators are incremental — a 32-bit
     * multiply here would call the Sozobon lmul helper 12x per frame
     * (~3000-4800 cycles, ~4% of the frame, stolen straight from
     * inference). Incremental adds cost nothing. */
    u16 i = 0;
    for (u16 r = 0; r < FLOOR_ROWS; r++) {
        dsRowPos[r] = (s16)(dsRowPos[r] - (s16)(r + 3));
        s16 v = (s16)(dsRowPos[r] >> 3);
        for (u16 l = 0; l < 8; l++) dsScroll[i++] = v;
    }
    VDP_setHorizontalScrollLine(BG_B, FLOOR_ROW * 8, dsScroll,
                                FLOOR_ROWS * 8, CPU);

    /* star twinkle: cycle brightness every 16 frames */
    switch ((dsFrame >> 4) & 3) {
        case 0: PAL_setColor(32 + DC_WHITE, RGB24_TO_VDPCOLOR(0xFFFFFF)); break;
        case 1: PAL_setColor(32 + DC_WHITE, RGB24_TO_VDPCOLOR(0xA0A0C0)); break;
        case 2: PAL_setColor(32 + DC_WHITE, RGB24_TO_VDPCOLOR(0x606080)); break;
        case 3: PAL_setColor(32 + DC_WHITE, RGB24_TO_VDPCOLOR(0xC0C0E0)); break;
    }
}

static void initDreamscape(void)
{
    /* PAL2: dream colors */
    PAL_setColor(32 + DC_WHITE, RGB24_TO_VDPCOLOR(0xFFFFFF));
    PAL_setColor(32 + DC_CYAN,  RGB24_TO_VDPCOLOR(0x60C0E0));
    PAL_setColor(32 + DC_GOLD,  RGB24_TO_VDPCOLOR(0xE0D090));
    PAL_setColor(32 + DC_CHK_A, RGB24_TO_VDPCOLOR(0x483070)); /* deep violet */
    PAL_setColor(32 + DC_CHK_B, RGB24_TO_VDPCOLOR(0x101838)); /* midnight   */
    PAL_setColor(32 + DC_GLOW1, RGB24_TO_VDPCOLOR(0x8040A0)); /* dream glow */
    PAL_setColor(32 + DC_GLOW2, RGB24_TO_VDPCOLOR(0xC060A0));
    /* backdrop: deep navy night */
    PAL_setColor(0, RGB24_TO_VDPCOLOR(0x000818));

    buildDreamTiles();
    buildDreamMap();

    /* per-line hscroll; BG_A stays put (zeroed table) */
    VDP_setScrollingMode(HSCROLL_LINE, VSCROLL_PLANE);
    {
        static s16 zeros[224];
        memset(zeros, 0, sizeof(zeros));
        VDP_setHorizontalScrollLine(BG_A, 0, zeros, 224, CPU);
        VDP_setHorizontalScrollLine(BG_B, 0, zeros, 224, CPU);
    }
    mwInit();
    runInit();
    SYS_setVIntCallback(dreamscapeVInt);
}

/* Boot splash: Sophia belts the classic chant... then corrects herself.
 * "SEEEGAAA" (choir) -> "Errr... Elyan Labs!" (deadpan), 8-bit PCM
 * through the YM2612 DAC. Text flips in sync with the correction. */
static void elyanSplash(void)
{
    VDP_drawText("S E G A", 16, 12);
    SND_startPlay_PCM(elyan_chant, sizeof(elyan_chant),
                      SOUND_RATE_22050, SOUND_PAN_CENTER, FALSE);
    /* choir runs ~2.47s = 148 frames */
    for (u16 f = 0; f < 148; f++) SYS_doVBlankProcess();
    VDP_clearTextArea(16, 12, 8, 1);
    VDP_drawText("E R R . . .", 14, 12);
    for (u16 f = 0; f < 60; f++) SYS_doVBlankProcess();
    VDP_clearTextArea(14, 12, 12, 1);
    VDP_drawText("E L Y A N   L A B S", 10, 12);
    for (u16 f = 0; f < 150; f++) SYS_doVBlankProcess();
    VDP_clearTextArea(10, 12, 20, 1);
}

/* ------------------------------------------------------------------ */
/* Inference-driven dialog                                            */
/* ------------------------------------------------------------------ */
static const char *prompts[] = {
    "Who are you?: ",
    "When were you born?: ",
    "How old are you?: ",
    "What is your name?: ",
    "Where are you from?: ",
    "What is your purpose?: ",
    "What is RustChain?: ",
    "What is the G4?: ",
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
    SYS_disableInts();
    VDP_clearTextArea(DIALOG_X, DIALOG_Y, DIALOG_W, DIALOG_H);
    SYS_enableInts();
    col = 0; rowY = 0;
}

static void putGlyph(char c)
{
    if (c == '\n') { col = 0; rowY++; return; }
    char s[2] = { c, 0 };
    if (rowY < DIALOG_H) {
        SYS_disableInts();
        VDP_drawText(s, DIALOG_X + col, DIALOG_Y + rowY);
        SYS_enableInts();
    }
    col++;
    if (col >= DIALOG_W) { col = 0; rowY++; }
}

/* ================================================================== */
/* THE THINKING WINDOW                                                */
/* A little framed panel beside Elya showing which part of her mind   */
/* is active and what it is doing. Sega-flavoured chunky 8x8 icons,   */
/* drawn procedurally like her face (no art pipeline needed).         */
/*                                                                    */
/* Icons live in their own tile block; only the two tilemap entries   */
/* that change are rewritten, so an update is 4 bytes of VRAM.        */
/* ================================================================== */
#define TW_TILE_INDEX (DS_TILE_INDEX + DS_NTILES)
#define TW_NICONS 10
/* expert icons (index == expert id, matching train/experts.py order) */
#define TWI_IDENTITY  0   /* a face          */
#define TWI_QUEST     1   /* a sword         */
#define TWI_RUSTCHAIN 2   /* a coin          */
#define TWI_HARDWARE  3   /* a chip          */
/* state icons */
#define TWI_IDLE      4   /* dim dot         */
#define TWI_THINK1    5   /* spinner frame 1 */
#define TWI_THINK2    6   /* spinner frame 2 */
#define TWI_THINK3    7   /* spinner frame 3 */
#define TWI_SPEAK1    8   /* sound waves     */
#define TWI_SPEAK2    9

#define TW_X 26           /* tile coords of the panel */
#define TW_Y 6

static u32 mwTiles[TW_NICONS * 8];

static void mwPix(u16 tile, u16 x, u16 y, u16 c)
{
    u32 *row = &mwTiles[tile * 8 + y];
    u16 sh = (7 - x) * 4;
    *row = (*row & ~((u32)0xF << sh)) | ((u32)c << sh);
}

/* draw an 8x8 icon from a compact string map: '.'=clear, digits=color */
static void mwGlyph(u16 tile, const char *rows)
{
    for (u16 y = 0; y < 8; y++)
        for (u16 x = 0; x < 8; x++) {
            char ch = rows[y * 8 + x];
            if (ch != '.') mwPix(tile, x, y, (u16)(ch - '0'));
        }
}

static void mwBuildIcons(void)
{
    memset(mwTiles, 0, sizeof(mwTiles));
    /* palette here is PAL1 (Elya's): 1 skin 2 hair 3 hairD 4 eye
     * 5 mouth 6 outline 7 blush */
    mwGlyph(TWI_IDENTITY,                    /* a little face          */
        "..6666.."
        ".622226."
        "6241422."
        "6222226."
        "6255526."
        ".622226."
        "..6666.."
        "........");
    mwGlyph(TWI_QUEST,                       /* a sword                */
        "....6..."
        "...646.."
        "...646.."
        "...646.."
        ".6666666"
        "...646.."
        "...66..."
        "....6...");
    mwGlyph(TWI_RUSTCHAIN,                   /* a coin                 */
        "..6666.."
        ".644446."
        "6474474."
        "6447744."
        "6447744."
        "6474474."
        ".644446."
        "..6666..");
    mwGlyph(TWI_HARDWARE,                    /* a chip with legs       */
        ".6.66.6."
        "66666666"
        "64444446"
        "64744746"
        "64444446"
        "66666666"
        ".6.66.6."
        "........");
    mwGlyph(TWI_IDLE,
        "........"
        "........"
        "...66..."
        "..6336.."
        "..6336.."
        "...66..."
        "........"
        "........");
    /* three-frame spinner: she is thinking */
    mwGlyph(TWI_THINK1,
        "...44..."
        "..4444.."
        "...44..."
        "........"
        "........"
        "...33..."
        "..3333.."
        "...33...");
    mwGlyph(TWI_THINK2,
        "........"
        "......44"
        ".....444"
        "........"
        "........"
        "333....."
        "33......"
        "........");
    mwGlyph(TWI_THINK3,
        "........"
        "........"
        "44......"
        "444....."
        ".....333"
        "......33"
        "........"
        "........");
    /* speaking: sound waves radiating */
    mwGlyph(TWI_SPEAK1,
        "...5...."
        "..55.7.."
        ".555..7."
        ".555.7.7"
        ".555.7.7"
        ".555..7."
        "..55.7.."
        "...5....");
    mwGlyph(TWI_SPEAK2,
        "...5...."
        "..55...."
        ".555.7.."
        ".555.7.."
        ".555.7.."
        ".555.7.."
        "..55...."
        "...5....");
    VDP_loadTileData(mwTiles, TW_TILE_INDEX, TW_NICONS, DMA);
}

static u16 mwExpert = 0;      /* which expert is active   */
static u16 mwState  = 0;      /* 0 idle, 1 thinking, 2 speaking */

static void mwPut(u16 icon, u16 cx, u16 cy)
{
    SYS_disableInts();
    VDP_setTileMapXY(BG_A, TILE_ATTR_FULL(PAL1, TRUE, FALSE, FALSE,
                     TW_TILE_INDEX + icon), cx, cy);
    SYS_enableInts();
}

/* 8 experts now; names kept to 5 chars for the MIND window */
static const char *TW_NAMES[8] = { "BORN", "SELF", "ZELDA", "QUEST",
                                   "CHAIN", "RETRO", "IRON", "META" };

static void mwInit(void)
{
    mwBuildIcons();
    panelWindow(TW_X - 1, TW_Y - 2, 9, 7, "MIND");
    mwPut(TWI_IDENTITY, TW_X + 1, TW_Y + 1);
    mwPut(TWI_IDLE,     TW_X + 3, TW_Y + 1);
    SYS_disableInts();
    VDP_drawText(TW_NAMES[0], TW_X, TW_Y + 3);
    SYS_enableInts();
}

/* called when the router picks an expert (or at prompt start for now) */
static void mwSetExpert(u16 e)
{
    if (e > 7) e = 0;
    mwExpert = e;
    mwPut(TWI_IDENTITY + (e & 3), TW_X + 1, TW_Y + 1);
    SYS_disableInts();
    VDP_clearTextArea(TW_X, TW_Y + 3, 6, 1);
    VDP_drawText(TW_NAMES[e], TW_X, TW_Y + 3);
    SYS_enableInts();
}

/* animated from the VBlank callback so the mind keeps ticking even
 * while the 68000 is deep inside a forward pass */
static void mwAnimate(u16 frame)
{
    u16 icon;
    if (mwState == 1)                                  /* thinking */
        icon = TWI_THINK1 + ((frame >> 3) % 3);
    else if (mwState == 2)                             /* speaking */
        icon = TWI_SPEAK1 + ((frame >> 3) & 1);
    else
        icon = TWI_IDLE;
    mwPut(icon, TW_X + 3, TW_Y + 1);
}


/* ------------------------------------------------------------------ */
/* Terminal panels — opaque black windows over the Dreamscape.        */
/*                                                                    */
/* A BG_B tile of index 0 is TRANSPARENT and shows the backdrop        */
/* colour, which is the near-black night (0x000818). So "cut a black   */
/* window" is simply: clear BG_B in that region and let the backdrop   */
/* through. Costs no tiles and no palette entries.                     */
/* The border is drawn on BG_A with box glyphs so it reads as a        */
/* terminal frame rather than a dashed line.                           */
/* ------------------------------------------------------------------ */
static void panelClear(u16 x, u16 y, u16 w, u16 h)
{
    /* BG_B SCROLLS horizontally per line, so clearing a fixed rectangle
     * of the tilemap does NOT produce a fixed window on screen — the
     * cleared area scrolls away and floor tiles scroll into it. A window
     * over a scrolling plane must span the plane's FULL width (64 cells)
     * in the rows it covers. x and w are therefore ignored. */
    (void)x; (void)w;
    SYS_disableInts();
    for (u16 ry = 0; ry < h; ry++)
        for (u16 rx = 0; rx < 64; rx++)
            VDP_setTileMapXY(BG_B, 0, rx, y + ry);
    SYS_enableInts();
}

/* single-line box frame, IBM-PC style, using the SGDK font */
static void panelFrame(u16 x, u16 y, u16 w, u16 h)
{
    SYS_disableInts();
    VDP_drawText("+", x, y);
    VDP_drawText("+", x + w - 1, y);
    VDP_drawText("+", x, y + h - 1);
    VDP_drawText("+", x + w - 1, y + h - 1);
    for (u16 i = 1; i < w - 1; i++) {
        VDP_drawText("-", x + i, y);
        VDP_drawText("-", x + i, y + h - 1);
    }
    for (u16 i = 1; i < h - 1; i++) {
        VDP_drawText("|", x, y + i);
        VDP_drawText("|", x + w - 1, y + i);
    }
    SYS_enableInts();
}

/* a black window: opaque interior + frame, with an optional title */
static void panelWindow(u16 x, u16 y, u16 w, u16 h, const char *title)
{
    /* Frame only. Clearing BG_B behind the panel was tried and reverted:
     * over the scrolling floor a fixed rectangle scrolls away, and
     * clearing whole rows to compensate ate the Dreamscape down to a
     * thin stripe. The dream stays whole; the frames float over it. */
    panelFrame(x, y, w, h);
    if (title) {
        SYS_disableInts();
        VDP_drawText(title, x + 2, y);
        SYS_enableInts();
    }
}


/* ================================================================== */
/* THE RUNNER — a tiny Elya sprinting the floor while she thinks.     */
/*                                                                    */
/* Hardware sprites, so it costs the scroll planes nothing and does   */
/* not disturb the Dreamscape or the dialog. She runs, rings drift    */
/* toward her, and when one is caught it pops and respawns ahead.     */
/* Pure Sonic homage, and it doubles as an honest activity indicator: */
/* it is driven from the VBlank callback, so it only moves while the  */
/* 68000 is genuinely working.                                        */
/* ================================================================== */
#define RUN_TILE   (TW_TILE_INDEX + TW_NICONS)
#define RUN_NTILES 10      /* 2 runner frames x4 tiles + 2 ring frames */
#define RUN_Y      150     /* pixel row: on the checkerboard floor     */
#define N_RINGS    3

static u32 runTiles[RUN_NTILES * 8];
static s16 runX = 0;
static s16 ringX[N_RINGS], ringPop[N_RINGS];
static u16 runVisible = FALSE;

static void runPix(u16 t, u16 x, u16 y, u16 c)
{
    u32 *row = &runTiles[t * 8 + y];
    u16 sh = (7 - x) * 4;
    *row = (*row & ~((u32)0xF << sh)) | ((u32)c << sh);
}

static void runGlyph(u16 t, const char *rows)
{
    for (u16 y = 0; y < 8; y++)
        for (u16 x = 0; x < 8; x++) {
            char ch = rows[y * 8 + x];
            if (ch != '.') runPix(t, x, y, (u16)(ch - '0'));
        }
}

/* 16x16 runner = 4 tiles in VDP column order: TL,BL,TR,BR */
static void buildRunner(void)
{
    memset(runTiles, 0, sizeof(runTiles));
    /* frame 0 - stride out. PAL1: 1 skin 2 hair 3 hairD 5 dress 6 line */
    runGlyph(0, "....333." "...3333" "..33333" "..33222" "..32222" "..32211"
                "...2211" "....211");                      /* head/hair TL */
    runGlyph(1, "....55.." "...5555." "..555555" "..55555." "...555.."
                "...1.1.." "..1...1." ".1.....1");          /* body/legs BL */
    runGlyph(2, "3......." "33......" "333....." "222....." "222....."
                "11......" "1......." "........");          /* hair trail TR */
    runGlyph(3, "5......." "55......" "5......." "........" "........"
                "........" "........" "........");          /* dress tail BR */
    /* frame 1 - legs together */
    runGlyph(4, "....333." "...3333" "..33333" "..33222" "..32222" "..32211"
                "...2211" "....211");
    runGlyph(5, "....55.." "...5555." "..555555" "..55555." "...555.."
                "...11..." "...1.1.." "..1...1.");
    runGlyph(6, "3......." "33......" "333....." "222....." "222....."
                "11......" "1......." "........");
    runGlyph(7, "5......." "55......" "5......." "........" "........"
                "........" "........" "........");
    /* ring, two frames (wide / edge-on) */
    runGlyph(8, "..3333.." ".3....3." "3......3" "3......3" "3......3"
                "3......3" ".3....3." "..3333..");
    runGlyph(9, "...33..." "...33..." "..3..3.." "..3..3.." "..3..3.."
                "..3..3.." "...33..." "...33...");
    VDP_loadTileData(runTiles, RUN_TILE, RUN_NTILES, DMA);
}

static void runInit(void)
{
    buildRunner();
    runX = 8;
    for (u16 i = 0; i < N_RINGS; i++) {
        ringX[i] = (s16)(120 + i * 70);
        ringPop[i] = 0;
    }
}

/* called from VBlank: only animates while she is actually working */
static void runAnimate(u16 frame)
{
    if (!runVisible) {
        VDP_setSpriteFull(0, 0, 0, SPRITE_SIZE(1, 1), 0, 0);
        return;
    }

    runX += 2;
    if (runX > 320) runX = -16;

    for (u16 i = 0; i < N_RINGS; i++) {
        if (ringPop[i]) {
            if (--ringPop[i] == 0) ringX[i] = (s16)(320 + (i * 40));
        } else {
            ringX[i] -= 1;                       /* drift toward her */
            if (ringX[i] < -8) ringX[i] = 320;
            s16 d = (s16)(ringX[i] - runX);
            if (d > -6 && d < 14) ringPop[i] = 12;   /* caught! */
        }
    }

    u16 f = ((frame >> 2) & 1) ? 4 : 0;           /* 2-frame run cycle */
    VDP_setSpriteFull(0, runX + 128, RUN_Y + 128, SPRITE_SIZE(2, 2),
                      TILE_ATTR_FULL(PAL1, TRUE, FALSE, FALSE, RUN_TILE + f),
                      1);
    for (u16 i = 0; i < N_RINGS; i++) {
        u16 rt = RUN_TILE + 8 + (ringPop[i] ? 1 : ((frame >> 3) & 1));
        s16 ry = RUN_Y + 4 - (ringPop[i] ? (12 - ringPop[i]) : 0);
        VDP_setSpriteFull(1 + i, ringX[i] + 128, ry + 128, SPRITE_SIZE(1, 1),
                          TILE_ATTR_FULL(PAL2, TRUE, FALSE, FALSE, rt),
                          (i == N_RINGS - 1) ? 0 : (2 + i));
    }
}

static void drawTokSpeed(void)
{
    char buf[24];
    /* x100 fixed point: vblank frames @60Hz -> tok/s (honest wall-clock) */
    u32 elapsed = vtimer - genStart;
    u32 t100 = elapsed ? (genTokens * 6000) / elapsed : 0;
    sprintf(buf, "%lu.%02lu tok/s", (u32)(t100 / 100), (u32)(t100 % 100));
    SYS_disableInts();
    VDP_drawText(buf, 26, 27);
    SYS_enableInts();
}

/* UP/DOWN browse the question list; A asks the highlighted one. */
static void showSelected(u16 idx)
{
    SYS_disableInts();
    VDP_clearTextArea(0, 26, 40, 1);
    VDP_drawText("<", 1, 26);
    VDP_drawText(">", 38, 26);
    {
        const char *q = prompts[idx];
        u16 n = 0;
        while (q[n] && n < 34) n++;
        if (n > 2) n -= 2;                  /* trim the trailing ": " */
        char buf[36];
        for (u16 i = 0; i < n; i++) buf[i] = q[i];
        buf[n] = 0;
        VDP_drawText(buf, (u16)(20 - (n >> 1)), 26);
    }
    SYS_enableInts();
}

static void startPrompt(u16 idx)
{
    eg_reset(&elya);
    curPrompt = prompts[idx];
    feedPos = 0;
    genCount = 0;
    genStart = vtimer;
    genTokens = 0;
    /* Lock-On MoE: pick which part of her mind answers this. With a
     * single-expert blob eg_route returns 0 and this is a no-op, so the
     * Mind Window still shows the right thing either way. */
    {
        u16 e = eg_route(&elya, curPrompt);
        eg_select_expert(&elya, e);
        mwSetExpert(e);
    }
    mode = ST_FEED;
    mwState = 1;                  /* thinking */
    runVisible = TRUE;
    clearDialog();
    SYS_disableInts();
    VDP_clearTextArea(0, 27, 40, 1);
    VDP_drawText("Elya dreams...", 13, 27);
    SYS_enableInts();
    /* echo the question in the dialog box */
    for (const char *p = curPrompt; *p; p++) putGlyph(*p);
    putGlyph('\n');
}

#ifdef ELYA_BENCH
/* Cycle-accurate throughput benchmark, run at boot with no input.
 * BlastEm is cycle-accurate, so vtimer (vblank count) gives a REAL
 * 68000 measurement rather than an x86 host proxy. Build two ROMs with
 * identical weights in different formats and compare the number.
 * See docs/SPEED_PLAN.md - this closes the "host x86" measurement gap. */
#define BENCH_TOKENS 40
static void runBenchmark(void)
{
    char buf[40];
    VDP_drawText("THROUGHPUT BENCHMARK", 10, 6);
#if EG_FORMAT_INDEX_STREAMS
    VDP_drawText("format: SGT2 index streams", 7, 8);
#else
    VDP_drawText("format: SGT1 2-bit packed", 7, 8);
#endif
    VDP_drawText("running...", 15, 10);

    eg_reset(&elya);
    u32 t0 = vtimer;
    u8 tok = 'a';
    for (u16 i = 0; i < BENCH_TOKENS; i++)
        tok = eg_next_token(&elya, tok);
    u32 frames = vtimer - t0;

    /* tok/s x100, integer only (no sprintf float, no 32-bit divide in
     * the measured region itself) */
    u32 t100 = frames ? ((u32)BENCH_TOKENS * 6000) / frames : 0;
    sprintf(buf, "%u tokens in %u frames", (u16)BENCH_TOKENS, (u16)frames);
    VDP_clearTextArea(0, 10, 40, 1);
    VDP_drawText(buf, 6, 10);
    sprintf(buf, "%u.%02u tok/s   %u ms/token",
            (u16)(t100 / 100), (u16)(t100 % 100),
            (u16)((frames * 1000UL) / (60UL * BENCH_TOKENS)));
    VDP_drawText(buf, 6, 12);
    VDP_drawText("(NTSC 60Hz, cycle-accurate)", 6, 14);
    while (TRUE) SYS_doVBlankProcess();
}
#endif

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

    initDreamscape();
    elyanSplash();

    /* original YM2612 + PSG dream theme (train/make_music.py) */
    XGM_setLoopNumber(-1);
    XGM_startPlay(dream_theme);

    VDP_drawText("ELYA INTO DREAMS", 12, 1);
    VDP_drawText("A transformer dreams on the 68000", 3, 2);

    if (eg_init(&elya, (const uint8_t *)elya_weights) == 0) {
        engineOk = TRUE;
        SYS_disableInts();
        VDP_clearTextArea(0, 27, 40, 1);
        VDP_drawText("UP/DOWN choose   A ask", 9, 27);
        SYS_enableInts();
        showSelected(promptIdx);
    } else {
        VDP_drawText("WEIGHTS MISSING - STUB ROM", 7, 26);
    }

    drawFacePixels(MOUTH_CLOSED);
    uploadFace();
    placeFaceTilemap(4, 6);

    /* black terminal window for her speech */
    panelWindow(DIALOG_X - 1, DIALOG_Y - 1, DIALOG_W + 2, DIALOG_H + 2,
                "ELYA");

    u16 prevJoy = 0;
    u16 blink = 0;

    while (TRUE) {
        u16 joy = JOY_readJoypad(JOY_1);

        if (mode == ST_IDLE && engineOk) {
            if ((joy & BUTTON_UP) && !(prevJoy & BUTTON_UP)) {
                promptIdx = (promptIdx + NUM_PROMPTS - 1) % NUM_PROMPTS;
                showSelected(promptIdx);
            }
            if ((joy & BUTTON_DOWN) && !(prevJoy & BUTTON_DOWN)) {
                promptIdx = (promptIdx + 1) % NUM_PROMPTS;
                showSelected(promptIdx);
            }
            if ((joy & BUTTON_A) && !(prevJoy & BUTTON_A))
                startPrompt(promptIdx);
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
                mwState = 2;      /* speaking */
                genCount = 0;
            }
            drawTokSpeed();
            break;

        case ST_GEN:
            if (lastTok == '\n' || genCount >= MAX_GEN
                || rowY >= DIALOG_H) {
                mode = ST_IDLE;
                mwState = 0;      /* idle */
                runVisible = FALSE;
                setMouth(MOUTH_CLOSED);
                showSelected(promptIdx);
                SYS_disableInts();
                VDP_clearTextArea(0, 27, 40, 1);
                VDP_drawText("UP/DOWN choose   A ask", 9, 27);
                SYS_enableInts();
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
