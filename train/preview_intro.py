#!/usr/bin/env python3
"""
preview_intro.py — decode an EIV2 blob back into a video.

Deliberately decodes the ENCODED DATA, not the source clip: it replays
tile slots, map deltas and the global palette exactly as the ROM player
will, so what you watch is what the Genesis will actually draw, including
every quantization and delta artefact. Previewing the source instead
would show a picture the console cannot produce.

Usage: preview_intro.py res/intro.bin [out.mp4] [--scale 4]
"""
import struct, subprocess, sys, tempfile
from pathlib import Path

import numpy as np
from PIL import Image

def vdp_to_rgb(w):
    r = ((w >> 1) & 7) * 36
    g = ((w >> 5) & 7) * 36
    b = ((w >> 9) & 7) * 36
    return (r, g, b)

def main():
    src = Path(sys.argv[1])
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else src.with_suffix(".mp4")
    scale = 4
    if "--scale" in sys.argv:
        scale = int(sys.argv[sys.argv.index("--scale") + 1])

    d = src.read_bytes()
    magic, tw, th, nfr, fps, peak = struct.unpack(">4sHHHHI", d[:16])
    assert magic == b"EIV2", magic
    pal = [vdp_to_rgb(struct.unpack(">H", d[16+i*2:18+i*2])[0]) for i in range(16)]
    p = 16 + 32

    W, H = tw * 8, th * 8
    tiles = {}                       # slot -> 8x8 index array
    tmap = [0] * (tw * th)

    with tempfile.TemporaryDirectory() as tmp:
        for f in range(nfr):
            n_t = struct.unpack(">H", d[p:p+2])[0]; p += 2
            for _ in range(n_t):
                slot = struct.unpack(">H", d[p:p+2])[0]; p += 2
                raw = d[p:p+32]; p += 32
                a = np.zeros((8, 8), np.uint8)
                for y in range(8):
                    for x in range(0, 8, 2):
                        b = raw[y*4 + x//2]
                        a[y][x] = b >> 4
                        a[y][x+1] = b & 0xF
                tiles[slot] = a
            n_m = struct.unpack(">H", d[p:p+2])[0]; p += 2
            for _ in range(n_m):
                cell, slot = struct.unpack(">HH", d[p:p+4]); p += 4
                tmap[cell] = slot

            img = np.zeros((H, W), np.uint8)
            for c, slot in enumerate(tmap):
                t = tiles.get(slot)
                if t is None:
                    continue
                ty, tx = divmod(c, tw)
                img[ty*8:ty*8+8, tx*8:tx*8+8] = t
            rgb = np.zeros((H, W, 3), np.uint8)
            for i, col in enumerate(pal):
                rgb[img == i] = col
            Image.fromarray(rgb).resize((W*scale, H*scale), Image.NEAREST) \
                 .save(Path(tmp) / f"p{f:05d}.png")

        subprocess.run(["ffmpeg", "-v", "error", "-y", "-framerate", str(fps),
                        "-i", str(Path(tmp) / "p%05d.png"),
                        "-c:v", "libx264", "-pix_fmt", "yuv420p",
                        str(out)], check=True)
    print(f"{out}  {nfr} frames, {tw*8}x{th*8} px at {scale}x, {fps} fps")

if __name__ == "__main__":
    main()
