#!/usr/bin/env python3
"""
make_intro.py — convert an .mp4 into a streaming 16-bit Genesis animation.

Usage:
  make_intro.py in.mp4 [--w 128] [--h 96] [--fps 12] [--out res/intro.bin]

WHY IT IS SHAPED THIS WAY (the Genesis budget, not artistic choice):

  VBlank DMA capacity   7,200 bytes/frame   (NTSC, 40-cell — SGDK's
                                             DMA_TRANSFER_CAPACITY_NTSC)
  free VRAM             ~1,600 tiles (51 KB)
  a tile                32 bytes, 4bpp, 8x8 px

A full 320x224 screen is 1,120 tiles = 35 KB — five VBlanks' worth of DMA
for ONE frame. So full-screen video at speed is impossible. A 128x96
window is 16x12 = 192 tiles = 6,144 bytes, which fits inside a single
VBlank with room spare. That is why the window is small: it is the largest
size that can be replaced every frame without tearing.

Three compressions, in order of payoff:

 1. PER-FRAME PALETTE. 16 CRAM words per frame is ~free, and re-quantizing
    each frame to its own 15 colours is a large quality win over one
    global palette. This is the cheapest thing on the machine.
 2. TILE DEDUPLICATION. Identical 8x8 blocks (flat backgrounds, letterbox
    bars) collapse to one tile.
 3. DELTA FRAMES. Only tiles that actually changed since the previous
    frame are emitted. On a typical clip this removes most of the data.

Output format (all multi-byte fields BIG-endian, 68000-native):

  header 16 B: 'EIV1' u16 tiles_w, u16 tiles_h, u16 n_frames, u16 fps,
               u32 max_tiles_per_frame
  per frame:   u16 pal[16]                     (CRAM words)
               u16 n_tiles, then n_tiles x { u16 slot, 32 B tile }
               u16 n_map,   then n_map   x { u16 cell, u16 slot }

A player DMAs the tile payload and pokes the map deltas; both are bounded
by max_tiles_per_frame so the ROM can size its buffer statically.
"""
import argparse, struct, subprocess, sys, tempfile
from pathlib import Path

import numpy as np
from PIL import Image

def vdp_color(r, g, b):
    q = lambda v: min(7, int(round(v / 255.0 * 7)))
    return (q(b) << 9) | (q(g) << 5) | (q(r) << 1)

def extract(path, w, h, fps, tmp):
    """ffmpeg -> PNG frames at the target size, letterboxed not stretched."""
    out = Path(tmp) / "f%05d.png"
    # TEMPORAL DENOISE FIRST. Delta-frame encoding only pays off when
    # unchanged areas are byte-identical, and camera grain in filmed
    # footage makes every tile differ slightly every frame - on the first
    # run of this converter dedup+delta actually made the file 3% LARGER.
    # hqdn3d's temporal terms flatten that noise so static regions really
    # are static. Scale down BEFORE denoising so the filter works on the
    # pixels we will actually keep.
    vf = (f"fps={fps},scale={w}:{h}:force_original_aspect_ratio=decrease,"
          f"pad={w}:{h}:(ow-iw)/2:(oh-ih)/2:color=black,"
          f"hqdn3d=4:3:12:9")
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(path),
                    "-vf", vf, str(out)], check=True)
    return sorted(Path(tmp).glob("f*.png"))

def build_palette(frames, n=15):
    """One palette for the whole clip, from a sample of frames.

    A PER-FRAME palette looked like the obvious win (16 CRAM words is
    nearly free) but it is actively harmful for compression: identical
    visual content lands on DIFFERENT indices each frame, so no tile ever
    matches the previous frame and delta encoding collapses to zero. Same
    for Floyd-Steinberg dithering - it randomises pixels frame to frame.
    A stable global palette + no dither makes static regions byte-identical,
    which is the whole precondition for delta frames to work."""
    step = max(1, len(frames) // 24)
    sheet = Image.new("RGB", (128, 96 * len(frames[::step])))
    for i, f in enumerate(frames[::step]):
        sheet.paste(Image.open(f).convert("RGB").resize((128, 96)), (0, i * 96))
    q = sheet.quantize(colors=n, method=Image.MEDIANCUT)
    pal = q.getpalette()[:n * 3]
    snap = lambda v: min(255, int(round(v / 255.0 * 7)) * 36)
    colors = [(0, 0, 0)]
    for i in range(n):
        r, g, b = pal[i*3:i*3+3]
        colors.append((snap(r), snap(g), snap(b)))
    return colors

def quantize(img, colors):
    """Map to the fixed palette. No dither: stability beats gradients here."""
    pimg = Image.new("P", (1, 1))
    flat = []
    for c in colors[1:]:
        flat += list(c)
    pimg.putpalette(flat + [0] * (768 - len(flat)))
    q = img.convert("RGB").quantize(palette=pimg, dither=Image.NONE)
    return np.asarray(q, dtype=np.uint8) + 1

def to_tiles(idx, tw, th):
    """Return (list_of_32B_tiles, tilemap) with duplicates collapsed."""
    tiles, lut, tmap = [], {}, []
    for ty in range(th):
        for tx in range(tw):
            blk = idx[ty*8:ty*8+8, tx*8:tx*8+8]
            raw = bytearray()
            for row in blk:
                for x in range(0, 8, 2):
                    raw.append(((int(row[x]) & 0xF) << 4) | (int(row[x+1]) & 0xF))
            key = bytes(raw)
            slot = lut.get(key)
            if slot is None:
                slot = len(tiles)
                lut[key] = slot
                tiles.append(key)
            tmap.append(slot)
    return tiles, tmap

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input")
    ap.add_argument("--w", type=int, default=128)
    ap.add_argument("--h", type=int, default=96)
    ap.add_argument("--fps", type=int, default=12)
    ap.add_argument("--out", default="res/intro.bin")
    a = ap.parse_args()
    assert a.w % 8 == 0 and a.h % 8 == 0, "size must be a multiple of 8"
    tw, th = a.w // 8, a.h // 8

    with tempfile.TemporaryDirectory() as tmp:
        frames = extract(a.input, a.w, a.h, a.fps, tmp)
        if not frames:
            print("no frames extracted"); sys.exit(1)
        print(f"{len(frames)} frames at {a.w}x{a.h} ({tw}x{th} tiles), {a.fps} fps")
        colors = build_palette(frames)
        print(f"global palette: {len(colors)} colours (stable across frames)")

        body = bytearray()
        prev_tiles, prev_map = {}, {}
        peak = 0
        raw_total = 0
        for fi, fp in enumerate(frames):
            idx = quantize(Image.open(fp), colors)
            tiles, tmap = to_tiles(idx, tw, th)
            raw_total += len(tiles) * 32

            fr = bytearray()

            # delta: only tiles whose CONTENT changed in that slot
            changed = [(s, t) for s, t in enumerate(tiles)
                       if prev_tiles.get(s) != t]
            fr += struct.pack(">H", len(changed))
            for s, t in changed:
                fr += struct.pack(">H", s) + t
            peak = max(peak, len(changed))

            # delta: only map cells that changed
            mchg = [(c, s) for c, s in enumerate(tmap) if prev_map.get(c) != s]
            fr += struct.pack(">H", len(mchg))
            for c, s in mchg:
                fr += struct.pack(">HH", c, s)

            body += fr
            prev_tiles = dict(enumerate(tiles))
            prev_map = dict(enumerate(tmap))
            if fi % 25 == 0:
                print(f"  frame {fi:4d}  {len(tiles):4d} tiles  "
                      f"{len(changed):4d} changed  {len(mchg):4d} map")

        hdr = struct.pack(">4sHHHHI", b"EIV2", tw, th, len(frames), a.fps, peak)
        for r, g, b in colors:                       # one global palette
            hdr += struct.pack(">H", vdp_color(r, g, b))
        out = Path(a.out)
        out.parent.mkdir(exist_ok=True)
        out.write_bytes(hdr + bytes(body))

        n = len(frames)
        print(f"\n{out}: {len(hdr)+len(body):,} bytes "
              f"({(len(hdr)+len(body))/1024:.0f} KB) for {n/a.fps:.1f}s")
        print(f"  raw (no dedup/delta) would be {raw_total/1024:.0f} KB "
              f"-> saved {100*(1-(len(body)/max(raw_total,1))):.0f}%")
        print(f"  peak {peak} tiles/frame = {peak*32:,} B DMA "
              f"({'FITS' if peak*32 <= 7200 else 'OVER'} the 7,200 B VBlank budget)")

if __name__ == "__main__":
    main()
