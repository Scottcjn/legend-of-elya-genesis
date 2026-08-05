#!/usr/bin/env python3
"""Sampling profiler for the bench ROM: one 68000 PC sample per emulated
frame, bucketed by nearest preceding symbol from out/symbol.txt."""
import bisect, collections, os, subprocess, sys, tempfile
bench = sys.argv[1] if len(sys.argv) > 1 else "../bench"
nframes = int(sys.argv[2]) if len(sys.argv) > 2 else 1200
syms = []
for line in open(os.path.join(bench, "out", "symbol.txt")):
    p = line.split()
    if len(p) == 3 and p[1].lower() in "tt":
        try: syms.append((int(p[0], 16) & 0xFFFFFF, p[2]))
        except ValueError: pass
syms.sort()
addrs = [a for a, _ in syms]
out = tempfile.mktemp(suffix=".txt")
lua = tempfile.mktemp(suffix=".lua")
open(lua, "w").write('OUT=%r\nNFRAMES=%d\n' % (out, nframes) + open("profile.lua").read())
subprocess.run(["/usr/games/mame", "genesis", "-cart",
                os.path.join(bench, "out", "rom.bin"), "-autoboot_script", lua,
                "-sound", "none", "-video", "none", "-nothrottle", "-window",
                "-skip_gameinfo", "-seconds_to_run", str(nframes // 60 + 5)],
               capture_output=True)
c = collections.Counter()
tot = 0
for line in open(out):
    pc = int(line.strip(), 16)
    i = bisect.bisect_right(addrs, pc) - 1
    c[syms[i][1] if i >= 0 else "?"] += 1
    tot += 1
for name, k in c.most_common(20):
    print("%6.2f%%  %6d  %s" % (100.0 * k / tot, k, name))
print("total samples", tot)
