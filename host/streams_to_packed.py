#!/usr/bin/env python3
"""
streams_to_packed.py — convert an SGT3/SGTM index-stream blob to the
2-bit packed layout, keeping the SAME weights.

This exists for one reason: to measure the SGT2 index-stream speedup on a
REAL 68000 instead of on an x86 host. The docs' headline 11.3x was a host
number and the honest 68000 figure was never taken. Converting a single
model into both layouts lets two otherwise-identical ROMs be benchmarked
against each other in a cycle-accurate emulator.

Flags bit1 selects the layout at RUNTIME (1 = index streams, 0 = packed),
so one engine binary reads both and the comparison isolates the format.

Usage: streams_to_packed.py in.bin out.bin
"""
import struct, sys

E, V, C = 64, 256, 64
FFN = E * 4
# (out_dim, in_dim) in blob order
TENSORS = [(E, E), (E, E), (E, E), (E, E), (FFN, E), (E, FFN)]

def pack_rows(rows, in_dim):
    """rows: list of dicts {idx: weight}. Emit 2-bit codes, 4 per byte."""
    out = bytearray()
    for row in rows:
        codes = []
        for i in range(in_dim):
            w = row.get(i, 0)
            codes.append(0 if w == 0 else (1 if w > 0 else 2))
        for b in range(0, in_dim, 4):
            out.append((codes[b] << 6) | (codes[b+1] << 4)
                       | (codes[b+2] << 2) | codes[b+3])
    return bytes(out)

def read_streams(src, p, out_dim):
    """Return (rows, new_p). Duplicate indices accumulate (quinary)."""
    rows = []
    for _ in range(out_dim):
        na = struct.unpack(">H", src[p:p+2])[0]; p += 2
        ns = struct.unpack(">H", src[p:p+2])[0]; p += 2
        row = {}
        for _i in range(na):
            row[src[p]] = row.get(src[p], 0) + 1; p += 1
        for _i in range(ns):
            row[src[p]] = row.get(src[p], 0) - 1; p += 1
        rows.append(row)
    return rows, p

def main():
    src = open(sys.argv[1], "rb").read()
    magic = src[:4]
    out = bytearray()

    if magic == b"SGT3" or magic == b"SGT2":
        layers = src[4]
        flags = src[12]
        hdr = 14
        out += src[:12] + bytes([flags & ~2, src[13]])   # clear bit1
        p = hdr
        out += src[p:p + V * E]; p += V * E              # embedding
        if flags & 4:
            out += src[p:p + C * E]; p += C * E          # PE
        n_exp = 1
        exp_starts = [p]
        lut_after = True
    elif magic == b"SGTM":
        n_exp = src[4]; layers = src[5]
        flags = struct.unpack(">H", src[14:16])[0]
        out += src[:14] + struct.pack(">H", flags & ~2)  # clear bit1
        p = 16
        out += src[p:p + V * E]; p += V * E
        if flags & 4:
            out += src[p:p + C * E]; p += C * E
        out += src[p:p + 512]; p += 512                  # exp LUT
        # router
        rout_in = E * 2 if len(sys.argv) < 4 else int(sys.argv[3])
        out += src[p:p+4]
        rows, q = read_streams(src, p + 4, n_exp)
        out += pack_rows(rows, rout_in)
        p = q
        off_pos = len(out)
        out += b"\0" * (4 * n_exp)
        p += 4 * n_exp                                   # skip old table
        offsets = []
        for _e in range(n_exp):
            offsets.append(len(out))
            for od, idim in TENSORS * layers:
                out += src[p:p+4]
                rows, p = read_streams(src, p + 4, od)
                out += pack_rows(rows, idim)
        for i, o in enumerate(offsets):
            out[off_pos + i*4: off_pos + i*4 + 4] = struct.pack(">I", o)
        open(sys.argv[2], "wb").write(bytes(out))
        print(f"{len(src):,} -> {len(out):,} bytes (packed is "
              f"{len(out)/len(src):.2f}x the streams size)")
        return
    else:
        print(f"unknown magic {magic}"); sys.exit(2)

    # single-model path
    for od, idim in TENSORS * layers:
        out += src[p:p+4]
        rows, p = read_streams(src, p + 4, od)
        out += pack_rows(rows, idim)
    out += src[p:]                                       # exp LUT
    open(sys.argv[2], "wb").write(bytes(out))
    print(f"{len(src):,} -> {len(out):,} bytes (packed is "
          f"{len(out)/len(src):.2f}x the streams size)")

if __name__ == "__main__":
    main()
