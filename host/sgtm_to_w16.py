#!/usr/bin/env python3
"""Convert an SGTM (or SGT2) byte-index-stream blob to WORD index streams.

Why: the 68000 inner loop spends most of the model's cycles turning a byte
index into an address:

    moveq #0,d0 ; move.b (a2)+,d0 ; add.l d0,d0 ; move.w (a5,d0.l),a1

Three of those four instructions exist only because the index is a byte that
has to be zero-extended and doubled. Storing the already-doubled BYTE OFFSET
as a big-endian u16 lets the same work be done with

    move.w (a2)+,d0 ; move.w (a5,d0.w),a1

which is also half as many cart-ROM bus accesses (the cart is 16 bits wide,
so a byte read costs a whole word fetch anyway).

Cost: the index streams double in size. Nothing else changes -- same weights,
same order, same arithmetic, bit-identical output.

Marked by flags bit3 (0x08). Rows stay 4-byte aligned-by-construction so the
68000 never sees a misaligned MOVE.W.
"""
import struct
import sys


def conv(src: bytes) -> bytes:
    assert src[:3] == b'SGT', "not an SGT blob"
    kind = src[3:4]
    out = bytearray()

    if kind == b'M':
        ne = src[4]
        embed = struct.unpack('>H', src[8:10])[0]
        vocab = struct.unpack('>H', src[10:12])[0]
        ctx = struct.unpack('>H', src[12:14])[0]
        flags = struct.unpack('>H', src[14:16])[0]
        layers = src[5]
        hdr_len = 16
    else:
        raise SystemExit("only SGTM handled here")

    assert flags & 2, "source must already be index-stream (flags bit1)"
    assert not (flags & 8), "source is already word-index"

    p = hdr_len
    out += src[:14]
    out += struct.pack('>H', flags | 8)
    p_emb = p
    p += vocab * embed
    if flags & 4:
        p += ctx * embed
    p_explut_end = p + 512
    out += src[p_emb:p_explut_end]
    p = p_explut_end

    ffn = embed * 4

    def conv_tensor(p):
        """Read one byte-index tensor at p; return (new_bytes, next_p)."""
        raise AssertionError

    def do_tensor(p, rows):
        blk = bytearray(src[p:p + 4])          # M, S, pad
        p += 4
        for _ in range(rows):
            na = struct.unpack('>H', src[p:p + 2])[0]
            ns = struct.unpack('>H', src[p + 2:p + 4])[0]
            p += 4
            blk += struct.pack('>HH', na, ns)
            for k in range(na + ns):
                blk += struct.pack('>H', src[p + k] * 2)
            p += na + ns
        return bytes(blk), p

    # router
    rblk, p = do_tensor(p, ne)
    out += rblk

    offs_pos = len(out)
    old_offs = []
    for e in range(ne):
        old_offs.append(struct.unpack('>I', src[p:p + 4])[0])
        p += 4
    out += b'\0' * (4 * ne)

    new_offs = []
    for e in range(ne):
        assert len(out) % 2 == 0, "expert block must start even"
        new_offs.append(len(out))
        q = old_offs[e]
        for _li in range(layers):
            for rows in (embed, embed, embed, embed, ffn, embed):
                blk, q = do_tensor(q, rows)
                out += blk

    for e in range(ne):
        struct.pack_into('>I', out, offs_pos + 4 * e, new_offs[e])
    return bytes(out)


if __name__ == '__main__':
    if len(sys.argv) != 3:
        raise SystemExit("usage: sgtm_to_w16.py in.bin out.bin")
    d = open(sys.argv[1], 'rb').read()
    o = conv(d)
    open(sys.argv[2], 'wb').write(o)
    print("in  %d bytes\nout %d bytes (+%.1f%%)" % (
        len(d), len(o), 100.0 * (len(o) - len(d)) / len(d)))
