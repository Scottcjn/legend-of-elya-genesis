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
from PIL import Image

HERE = Path(__file__).resolve().parent
SRC  = Path("/home/scott/sophia_heygen_headshot_tight.png")
RES  = HERE.parent / "res"
W, H = 64, 96

# idx: (name, r, g, b) — all channels on the 8-level Genesis grid
PALETTE = [
    ("TRANSPARENT",   0,   0,   0),
    ("OUTLINE",      36,   0,  36),
    ("SKIN_SHADOW", 182, 109, 109),
    ("SKIN_BASE",   218, 182, 145),
    ("SKIN_HI",     255, 218, 182),
    ("HAIR_SHADOW", 109,  36,   0),
    ("HAIR_BASE",   182,  72,  36),
    ("HAIR_LIGHT",  218, 109,  36),
    ("GOLD",        255, 182,  72),
    ("ARMOR_DARK",   72,  36,  36),
    ("ARMOR_MID",   109,  72,  72),
    ("LIPS",        218, 109, 109),
    ("IRIS",        145,  72,  36),
    ("GEM_CORE",    255, 145,  36),
    ("GEM_EDGE",    218,  72,   0),
    ("TECH_BLUE",   109, 182, 255),
]

def vdp_word(r, g, b):
    """VDP CRAM word: ----BBB- GGG- RRR-  (3 bits per channel, <<1)."""
    q = lambda v: min(7, int(round(v / 255.0 * 7)))
    return (q(b) << 9) | (q(g) << 5) | (q(r) << 1)

def main():
    img = Image.open(SRC).convert("RGB")
    sw, sh = img.size

    # Bust crop: head + shoulders + chest gem. The reference is a 512px
    # square with her head in the upper-middle; this window keeps the
    # circlet at top and the gem inside the lower third.
    left  = int(sw * 0.20)
    right = int(sw * 0.86)
    top   = int(sh * 0.06)
    bot   = int(sh * 0.78)
    img = img.crop((left, top, right, bot))

    # match target aspect before the downscale so nothing gets squashed
    tw, th = img.size
    target_ar = W / H
    if tw / th > target_ar:                    # too wide: trim sides
        new_w = int(th * target_ar)
        off = (tw - new_w) // 2
        img = img.crop((off, 0, off + new_w, th))
    else:                                      # too tall: trim bottom
        new_h = int(tw / target_ar)
        img = img.crop((0, 0, tw, new_h))

    # gentle pre-sharpen survives the brutal downscale better
    small = img.resize((W * 2, H * 2), Image.LANCZOS)
    small = small.resize((W, H), Image.LANCZOS)
    arr = np.asarray(small).astype(np.float32)

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
    bgish = (sat < 0.34) | (lum > 205)

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
