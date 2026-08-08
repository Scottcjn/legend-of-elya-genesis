#!/usr/bin/env python3
"""Count nonzero ternary weights per tensor in an SGTM index-stream blob.

Reports the per-forward-pass nonzero-weight budget (each tensor is walked
exactly once per layer per token), which is what the 68000 inner loop
actually costs. Handles both byte index streams (flags bit1) and W16
streams (flags bit3).

Usage: blob_stats.py model.bin [expert]
"""
import struct
import sys


def parse(src, want_expert=None):
    assert src[:4] == b'SGTM', "only SGTM handled"
    ne = src[4]
    layers = src[5]
    embed = struct.unpack('>H', src[8:10])[0]
    vocab = struct.unpack('>H', src[10:12])[0]
    ctx = struct.unpack('>H', src[12:14])[0]
    flags = struct.unpack('>H', src[14:16])[0]
    width = 2 if (flags & 8) else 1
    assert flags & 2, "not an index-stream blob"

    p = 16
    p += vocab * embed
    if flags & 4:
        p += ctx * embed
    p += 512                                    # explut

    def walk(p, rows):
        """Walk one tensor; return (nonzeros, n_add, n_sub, next_p)."""
        p += 4                                  # M, S, pad
        nz = na_t = ns_t = 0
        for _ in range(rows):
            na = struct.unpack('>H', src[p:p + 2])[0]
            ns = struct.unpack('>H', src[p + 2:p + 4])[0]
            p += 4 + (na + ns) * width
            nz += na + ns
            na_t += na
            ns_t += ns
        return nz, na_t, ns_t, p

    _, _, _, p = walk(p, ne)                    # router
    offs = []
    for _ in range(ne):
        offs.append(struct.unpack('>I', src[p:p + 4])[0])
        p += 4

    ffn = embed * 4
    names = ('wq', 'wk', 'wv', 'wo', 'wff1', 'wff2')
    rowcnt = (embed, embed, embed, embed, ffn, embed)
    incnt = (embed, embed, embed, embed, embed, ffn)

    out = {}
    for e in range(ne):
        if want_expert is not None and e != want_expert:
            continue
        q = offs[e]
        per = {}
        for li in range(layers):
            for nm, rows, ind in zip(names, rowcnt, incnt):
                nz, na, ns, q = walk(q, rows)
                per[(li, nm)] = (nz, na, ns, rows, ind)
        out[e] = per
    return out, layers, embed, ffn


def main():
    src = open(sys.argv[1], 'rb').read()
    want = int(sys.argv[2]) if len(sys.argv) > 2 else None
    experts, layers, embed, ffn = parse(src, want)
    for e, per in sorted(experts.items()):
        tot = sum(v[0] for v in per.values())
        print("expert %d: %d nonzero weights per forward pass" % (e, tot))
        bytens = {}
        for (li, nm), (nz, na, ns, rows, ind) in sorted(per.items()):
            bytens[nm] = bytens.get(nm, 0) + nz
            dens = 100.0 * nz / (rows * ind)
            print("  L%d %-5s nz=%6d (add %5d sub %5d) rows=%4d in=%4d "
                  "density=%5.1f%%" % (li, nm, nz, na, ns, rows, ind, dens))
        print("  --- per-tensor totals across layers ---")
        for nm in ('wq', 'wk', 'wv', 'wo', 'wff1', 'wff2'):
            print("  %-5s %6d  %5.2f%% of all nonzeros"
                  % (nm, bytens[nm], 100.0 * bytens[nm] / tot))


if __name__ == '__main__':
    main()
