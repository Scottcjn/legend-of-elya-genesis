#!/usr/bin/env python3
"""
make_portrait.py — convert the canonical Elya reference into a Genesis
64x96 bust portrait: 15 colors + transparent, all legal 9-bit VDP colors.

Implements the portrait spec in docs/PORTRAIT_SPEC.md. Emits:
  res/elya_portrait.png   indexed PNG (index 0 = transparent), for rescomp
  res/elya_portrait.pal   16 VDP color words, big-endian
  train/portrait_preview.png  8x upscale, for eyeballing on a monitor

The palette is FIXED (hand-designed, not k-means): a tiny palette chosen
by role beats a statistically-optimal one, because the roles are what
carry the likeness at 64 px.
"""
from pathlib import Path
import numpy as np
from PIL import Image, ImageEnhance, ImageOps

HERE = Path(__file__).resolve().parent
import os as _os
# Victorian Elya (the Study portrait) is the canon face for the ROM. The
# armored render also converts well, but a close-framed FACE survives a
# 64x96 downscale far better than a full-body composition - at this size
# the likeness lives entirely in the eyes and mouth.
SRC  = Path(_os.environ.get("ELYA_PORTRAIT_SRC",
                            "/home/scott/sophia_victorian_frame.png"))
RES  = HERE.parent / "res"
W, H = 64, 96

# idx: (name, r, g, b) — all channels on the 8-level Genesis grid
PALETTE = [
    ("TRANSPARENT",   0,   0,   0),
    ("OUTLINE",      36,   0,  36),
    ("SKIN_SHADOW", 182, 109, 109),
    ("SKIN_BASE",   218, 182, 145),
    ("SKIN_HI",     255, 218, 182),
    ("HAIR_SHADOW",  36,   0,   0),   # near-black brown, parted centre
    ("HAIR_BASE",    72,  36,  36),
    ("HAIR_LIGHT",  109,  72,  36),   # the warm rim the lamp puts on it
    ("GOLD",        182, 145,  72),   # brooch, picture frame
    ("DRESS_DARK",   36,  36,  36),   # the black high-collar dress
    ("DRESS_MID",    72,  72,  72),
    ("LIPS",        218, 145, 145),
    ("IRIS",        145, 182, 182),   # pale green-grey eyes
    ("ROOM_WARM",   145, 109,  72),   # sepia wall behind her
    ("ROOM_DARK",    72,  36,   0),
    ("HIGHLIGHT",   255, 218, 182),
]

def vdp_word(r, g, b):
    """VDP CRAM word: ----BBB- GGG- RRR-  (3 bits per channel, <<1)."""
    q = lambda v: min(7, int(round(v / 255.0 * 7)))
    return (q(b) << 9) | (q(g) << 5) | (q(r) << 1)

def main():
    img = Image.open(SRC).convert("RGB")
    sw, sh = img.size

    # Crop is SOURCE-DEPENDENT and must be given, not guessed. The first
    # version hard-coded percentages tuned for a full-body render; applied
    # to a close-up portrait they zoomed into her cheek. Defaults below
    # suit the Victorian Study frame (1280x718, face centred, full height).
    # Override with ELYA_CROP="left,top,right,bottom" as fractions.
    crop = _os.environ.get("ELYA_CROP", "0.30,0.00,0.68,1.00")
    cl, ct, cr, cb = (float(v) for v in crop.split(","))
    img = img.crop((int(sw * cl), int(sh * ct), int(sw * cr), int(sh * cb)))

    # match target aspect before the downscale so nothing gets squashed
    # Match the 64:96 aspect by PADDING, never by trimming the sides.
    # Trimming was cutting 106 px off each edge of a 691 px crop - exactly
    # where her hair falls - so she arrived bald. Padding keeps the full
    # width and the vignette absorbs the empty space top and bottom.
    tw, th = img.size
    target_ar = W / H
    if tw / th > target_ar:                    # too wide: pad top/bottom
        new_h = int(tw / target_ar)
        pad = Image.new("RGB", (tw, new_h), (0, 0, 0))
        pad.paste(img, (0, (new_h - th) // 3))   # bias up: keep the chin
        img = pad
    else:                                      # too tall: pad sides
        new_w = int(th * target_ar)
        pad = Image.new("RGB", (new_w, th), (0, 0, 0))
        pad.paste(img, ((new_w - tw) // 2, 0))
        img = pad

    # STRETCH CONTRAST BEFORE QUANTIZING. The Victorian frame is warm,
    # low-contrast and nearly monochrome; mapped straight onto 15 colours
    # the hair, dress and background all collapse into the same browns and
    # the likeness dissolves. autocontrast reclaims the range, then a
    # saturation lift keeps skin and hair on separate palette entries.
    # Do it BEFORE the downscale so the resampler averages good pixels.
    img = ImageOps.autocontrast(img, cutoff=2)
    img = ImageEnhance.Color(img).enhance(1.25)

    # gentle pre-sharpen survives the brutal downscale better
    small = img.resize((W * 2, H * 2), Image.LANCZOS)
    small = small.resize((W, H), Image.LANCZOS)
    arr = np.asarray(small).astype(np.float32)

    # PALETTE: hand-designed by role, or DERIVED from the image.
    # The role-based palette was tuned for one specific render; applied to
    # a different photograph its hard blacks and saturated entries eat the
    # face. Deriving 15 colours from the actual pixels (median cut, then
    # snapped to the VDP's 8-level grid) is what made the intro conversion
    # work, and it generalises to any source. ELYA_PALETTE=role restores
    # the hand-designed one.
    if _os.environ.get("ELYA_PALETTE", "auto") == "auto":
        q = small.quantize(colors=15, method=Image.MEDIANCUT)
        raw = q.getpalette()[:45]
        snap = lambda v: min(255, int(round(v / 255.0 * 7)) * 36)
        derived = [(snap(raw[i*3]), snap(raw[i*3+1]), snap(raw[i*3+2]))
                   for i in range(15)]
        globals()["PALETTE"] = [("TRANSPARENT", 0, 0, 0)] + [
            (f"C{i}", *derived[i]) for i in range(15)]

    # nearest-color in a perceptually weighted space (green matters most)
    pal = np.array([[r, g, b] for _n, r, g, b in PALETTE[1:]], np.float32)
    wgt = np.array([0.30, 0.59, 0.11], np.float32)
    d = arr[:, :, None, :] - pal[None, None, :, :]
    dist = ((d * d) * wgt).sum(-1)
    idx = dist.argmin(-1).astype(np.uint8) + 1          # 0 stays transparent

    # Background removal by flood-fill from the frame edges. The
    # reference backdrop is BOTH dark (left) and bright/washed (right),
    # so keying on brightness alone fails; instead grow inward through
    # any pixel that is low-saturation — skin, hair and gold are all
    # strongly saturated, the backdrop is not.
    mx = arr.max(-1)
    mn = arr.min(-1)
    sat = (mx - mn) / np.maximum(mx, 1.0)
    lum = (arr * wgt).sum(-1)
    # Dark hair is low-saturation too, so this flood-fill happily eats it.
    # Require the pixel to be low-saturation AND bright (a washed backdrop);
    # anything dark is assumed to be her, not the room.
    bgish = ((sat < 0.28) & (lum > 120)) | (lum > 215)

    bg = np.zeros((H, W), bool)
    bg[0, :] = bgish[0, :]
    bg[:, 0] = bgish[:, 0]
    bg[:, -1] = bgish[:, -1]
    for _ in range(W + H):                 # iterate to fixpoint
        nb = bg.copy()
        nb[1:, :]  |= bg[:-1, :]
        nb[:-1, :] |= bg[1:, :]
        nb[:, 1:]  |= bg[:, :-1]
        nb[:, :-1] |= bg[:, 1:]
        nb &= bgish
        if (nb == bg).all():
            break
        bg = nb
    bg[H - 12:, :] = False                 # never eat the chest/gem
    idx[bg] = 0

    # VIGNETTE TO TRANSPARENT. Saturation-based background removal works on
    # a subject shot against a flat backdrop, but the Victorian frame has a
    # warm SATURATED sepia room behind her, so nothing keys out and she
    # ships as a hard rectangle pasted over the Dreamscape. An elliptical
    # falloff to transparency both solves that and suits the source: a
    # Victorian cameo portrait is exactly this shape.
    cy, cx = H * 0.46, W * 0.5
    # Reach the frame edges. At 0.46 the ellipse cut inside the canvas and
    # sliced off the hair falling on both sides of her face.
    ry, rx = H * 0.56, W * 0.54
    for y in range(H):
        for x in range(W):
            dy = (y - cy) / ry
            dx = (x - cx) / rx
            r = (dx * dx + dy * dy) ** 0.5
            if r > 1.0:
                idx[y][x] = 0                      # outside the cameo
            elif r > 0.90:
                # soft edge: dither the last band so it feathers instead
                # of showing a hard ellipse (no alpha on this hardware)
                if ((x + y) & 1) == 0:
                    idx[y][x] = 0

    out = Image.fromarray(idx, mode="P")
    flat = []
    for _n, r, g, b in PALETTE:
        flat += [r, g, b]
    out.putpalette(flat + [0] * (768 - len(flat)))
    RES.mkdir(exist_ok=True)
    out.save(RES / "elya_portrait.png")

    with (RES / "elya_portrait.pal").open("wb") as fh:
        for _n, r, g, b in PALETTE:
            w = vdp_word(r, g, b)
            fh.write(bytes([(w >> 8) & 0xFF, w & 0xFF]))

    # ---- VISEME MOUTHS -------------------------------------------------
    # Only the mouth tiles change while she speaks, so they are emitted as
    # separate small images. At runtime the game swaps 2 tilemap entries
    # (4 bytes) per token instead of re-uploading the whole face - the
    # difference between free and 3 KB of DMA per token.
    #
    # The mouth region is fixed at MOUTH_X/Y below and is 16x8 px = 2 tiles.
    # Shapes are drawn over the base rather than extracted from video, so
    # this works with the still reference we already have; a talking-head
    # clip can replace them later without touching the ROM side.
    MX, MY, MW, MH = 24, 62, 16, 8
    LIP, DARK, TEETH = 11, 1, 4        # palette indices from PALETTE above
    variants = {}
    for name in ("open", "wide"):
        v = idx.copy()
        for y in range(MH):
            for x in range(MW):
                v[MY + y][MX + x] = idx[MY + y][MX + x]
        cy = MH // 2
        if name == "open":
            for y in range(cy - 1, cy + 2):
                for x in range(4, MW - 4):
                    v[MY + y][MX + x] = DARK
            for x in range(3, MW - 3):
                v[MY + cy - 2][MX + x] = LIP
                v[MY + cy + 2][MX + x] = LIP
        else:                            # wide: taller, teeth showing
            for y in range(cy - 2, cy + 3):
                for x in range(2, MW - 2):
                    v[MY + y][MX + x] = DARK
            for x in range(3, MW - 3):
                v[MY + cy - 2][MX + x] = TEETH
            for x in range(1, MW - 1):
                v[MY + cy - 3][MX + x] = LIP
                v[MY + cy + 3][MX + x] = LIP
        strip = v[MY:MY + MH, MX:MX + MW]
        im = Image.fromarray(strip, mode="P")
        im.putpalette(flat + [0] * (768 - len(flat)))
        im.save(RES / f"elya_mouth_{name}.png")
        variants[name] = strip
    print(f"mouth variants: {MW}x{MH} px = {(MW//8)*(MH//8)} tiles each, "
          f"swap cost {2*(MW//8)*(MH//8)} bytes/token")

    out.convert("RGB").resize((W * 8, H * 8), Image.NEAREST) \
       .save(HERE / "portrait_preview.png")

    used = np.bincount(idx.flatten(), minlength=16)
    print(f"{W}x{H} portrait -> {RES/'elya_portrait.png'}")
    print(f"tiles: {(W//8)*(H//8)}  VRAM: {(W//8)*(H//8)*32:,} bytes")
    for i, (n, *_rgb) in enumerate(PALETTE):
        print(f"  {i:2d} {n:12s} {used[i]:5d} px "
              f"({100*used[i]/(W*H):5.1f}%)")

if __name__ == "__main__":
    main()
