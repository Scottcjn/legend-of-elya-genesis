#!/usr/bin/env python3
"""Convert an SGT1 (2-bit packed) blob to SGT2 (index streams).

Used to prove the two engine paths are equivalent: the SGT1 engine on the
SGT1 blob and the SGT2 engine on the converted blob must produce
byte-identical generations.
"""
import struct, sys

L, E, H, V, C = 2, 64, 4, 256, 64
FFN = E * 4
TENSORS = [("wq", E, E), ("wk", E, E), ("wv", E, E), ("wo", E, E),
           ("wff1", FFN, E), ("wff2", E, FFN)]   # (name, out_dim, in_dim)

# header is 14 bytes: 4s magic, u8 layers, u8 heads, u16 embed,
# u16 vocab, u16 ctx, u8 flags, u8 pad
src = open(sys.argv[1], "rb").read()
assert src[:4] == b"SGT1", src[:4]
out = bytearray(b"SGT2" + src[4:12])          # through ctx (bytes 4..11)
out += bytes([3, 0])                          # flags: ternary|index-streams
p = 14
out += src[p:p + V * E]                       # embedding table verbatim
p += V * E

for _ in range(L):
    for _name, od, idim in TENSORS:
        out += src[p:p + 4]                   # M, S, pad
        p += 4
        nbytes = od * idim // 4
        packed = src[p:p + nbytes]
        p += nbytes
        per_row = idim // 4
        for r in range(od):
            row = packed[r * per_row:(r + 1) * per_row]
            adds, subs = [], []
            for bi, b in enumerate(row):
                for k in range(4):
                    c = (b >> (6 - 2 * k)) & 3
                    if c == 1:   adds.append(bi * 4 + k)
                    elif c == 2: subs.append(bi * 4 + k)
            out += struct.pack(">HH", len(adds), len(subs))
            out += bytes(adds) + bytes(subs)

out += src[p:]                                # exp LUT
open(sys.argv[2], "wb").write(bytes(out))
print(f"{len(src):,} -> {len(out):,} bytes "
      f"({len(out)/len(src):.2f}x ROM)")
