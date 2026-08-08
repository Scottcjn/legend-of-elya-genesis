#!/usr/bin/env python3
"""Transpose the `wff2` tensors of a W16 SGTM blob to INPUT-MAJOR order.

Why
---
`wff2` is the only matvec in the engine whose input is post-ReLU, and 55.8%
of that input is zero. In the shipped OUTPUT-MAJOR layout the kernel walks
rows and indexes into `in[]`, so it must touch a weight before it can
discover that the input it points at is zero — 56.5% of `wff2`'s weight
reads (18.7% of every nonzero-weight read in the whole engine) do exactly
that and contribute nothing.

Storing `wff2` INPUT-MAJOR inverts the loop: walk the input, and when
`in[i] == 0` skip that input's entire weight list with one pointer add,
without reading a single index from cart ROM.

Format (flags bit4 = 0x10; requires bit3 = W16 index streams)
-------------------------------------------------------------
Per `wff2` tensor, the u16 M / u8 S / u8 pad header is unchanged. What
follows is `in_dim` (= 4*embed = 256) COLUMNS instead of `out_dim` (= 64)
rows, each column i being:

    u16 n_add            outputs o with W[o][i] == +1
    u16 n_sub            outputs o with W[o][i] == -1
    n_add x u16be        o * 4 — byte offset into the int32 accumulator array
    n_sub x u16be        same

Offsets are pre-multiplied by 4 for the same reason the W16 conversion
pre-doubled its own: the 68000 then needs no shift and no zero-extend,
just `move.w (a2)+,d1 / add.l d0,(a3,d1.w)`. o*4 is 0..252, well inside
the signed 16-bit displacement the (An,Dn.W) mode uses.

Bit-identical output: each accumulator still receives exactly the same set
of +/- terms, only in a different order, and int32 addition is exact and
cannot overflow here (256 terms x 32767 = 8,388,352 << 2^31).

Only `wff2` is transposed. Every other tensor reads a dense vector with no
zeros worth skipping, so transposing them would pay the indirection for
nothing.

Old blobs keep working: the engine picks the layout from the flags word at
runtime, exactly the way bit3 already selects W16 over byte streams.

Usage: sgtm_wff2_transpose.py in_w16.bin out.bin
"""
import struct
import sys

FLAG_STREAMS = 2
FLAG_PE = 4
FLAG_W16 = 8
FLAG_WFF2_T = 0x10


def _read_tensor(src, p, rows, width):
    """Return (raw_bytes, rows_as_lists, next_p) for one index-stream tensor."""
    start = p
    hdr = src[p:p + 4]                       # M, S, pad
    p += 4
    rowdata = []
    for _ in range(rows):
        na = struct.unpack('>H', src[p:p + 2])[0]
        ns = struct.unpack('>H', src[p + 2:p + 4])[0]
        p += 4
        if width == 2:
            idx = list(struct.unpack('>%dH' % (na + ns), src[p:p + 2 * (na + ns)])
                       ) if (na + ns) else []
        else:
            idx = list(src[p:p + na + ns])
        p += (na + ns) * width
        rowdata.append((idx[:na], idx[na:]))
    return src[start:p], (hdr, rowdata), p


def transpose_wff2(hdr, rowdata, in_dim):
    """rowdata is [(add_idx, sub_idx)] per OUTPUT row, indices being byte
    offsets into the int16 input (= input_index * 2). Emit columns."""
    add_cols = [[] for _ in range(in_dim)]
    sub_cols = [[] for _ in range(in_dim)]
    for o, (adds, subs) in enumerate(rowdata):
        for byteoff in adds:
            assert byteoff % 2 == 0 and byteoff // 2 < in_dim, byteoff
            add_cols[byteoff // 2].append(o * 4)
        for byteoff in subs:
            assert byteoff % 2 == 0 and byteoff // 2 < in_dim, byteoff
            sub_cols[byteoff // 2].append(o * 4)

    blk = bytearray(hdr)
    for i in range(in_dim):
        a, s = add_cols[i], sub_cols[i]
        blk += struct.pack('>HH', len(a), len(s))
        for o in a:
            blk += struct.pack('>H', o)
        for o in s:
            blk += struct.pack('>H', o)
    return bytes(blk)


def conv(src):
    assert src[:4] == b'SGTM', "only SGTM handled"
    ne = src[4]
    layers = src[5]
    embed = struct.unpack('>H', src[8:10])[0]
    vocab = struct.unpack('>H', src[10:12])[0]
    ctx = struct.unpack('>H', src[12:14])[0]
    flags = struct.unpack('>H', src[14:16])[0]
    assert flags & FLAG_W16, "source must be a W16 blob (flags bit3)"
    assert not (flags & FLAG_WFF2_T), "source already has wff2 transposed"
    ffn = embed * 4

    out = bytearray(src[:14])
    out += struct.pack('>H', flags | FLAG_WFF2_T)

    p = 16
    p_emb = p
    p += vocab * embed
    if flags & FLAG_PE:
        p += ctx * embed
    p += 512                                  # explut
    out += src[p_emb:p]

    raw, _, p = _read_tensor(src, p, ne, 2)   # router: verbatim
    out += raw

    offs_pos = len(out)
    old_offs = [struct.unpack('>I', src[p + 4 * e:p + 4 * e + 4])[0]
                for e in range(ne)]
    p += 4 * ne
    out += b'\0' * (4 * ne)

    names = ('wq', 'wk', 'wv', 'wo', 'wff1', 'wff2')
    rowcnt = (embed, embed, embed, embed, ffn, embed)

    new_offs = []
    for e in range(ne):
        assert len(out) % 2 == 0, "expert block must start even"
        new_offs.append(len(out))
        q = old_offs[e]
        for _li in range(layers):
            for nm, rows in zip(names, rowcnt):
                raw, (hdr, rowdata), q = _read_tensor(src, q, rows, 2)
                if nm == 'wff2':
                    out += transpose_wff2(hdr, rowdata, ffn)
                else:
                    out += raw

    for e in range(ne):
        struct.pack_into('>I', out, offs_pos + 4 * e, new_offs[e])
    return bytes(out)


if __name__ == '__main__':
    if len(sys.argv) != 3:
        raise SystemExit("usage: sgtm_wff2_transpose.py in_w16.bin out.bin")
    d = open(sys.argv[1], 'rb').read()
    o = conv(d)
    open(sys.argv[2], 'wb').write(o)
    print("in  %d bytes\nout %d bytes (%+.2f%%)" % (
        len(d), len(o), 100.0 * (len(o) - len(d)) / len(d)))
