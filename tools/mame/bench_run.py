#!/usr/bin/env python3
"""Host runner for the Elya Genesis bench ROM.

Boots the ROM in MAME's `genesis` driver headless, measures the elapsed
emulated time between the ROM's START and END marker writes using MAME
memory write taps, converts it to 68000 cycles, and reads the generated
token ids back out of work RAM.

Everything is driven from the write tap - there is NO polling. Polling
from Lua (emu.register_periodic) made MAME run at ~0.07x realtime.

Usage: bench_run.py <bench_dir> [--label NAME] [--repeat N]
  <bench_dir> must contain out/rom.bin and out/symbol.txt
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile

MAME = "/usr/games/mame"

# NTSC Mega Drive 68000: master 53.693175 MHz / 7.
M68K_CLOCK_NTSC = 53693175.0 / 7.0

RUNNER_BODY = r'''
local mac = manager.machine
local cpu = mac.devices[":maincpu"]
local sp  = cpu.spaces["program"]

local function t2as(t) return t.seconds * 1e18 + t.attoseconds end

local t_start, t_end = nil, nil
local done = false
-- MAME's Lua bindings garbage-collect notifier/tap subscriptions if you do
-- not hold a reference. Dropping the handle silently stops every callback a
-- few emulated seconds in, which looks exactly like the ROM hanging.
KEEP = {}
local marks = {}

local function finish(body)
  if done then return end
  done = true
  local f = io.open(CFG.out, "w")
  f:write(body)
  f:close()
  mac:exit()
end

-- The 68000 splits a move.l into two word writes, so the tap fires twice
-- per marker store. Only the LOW word (marker+2) carries the value.
local m = CFG.marker
KEEP[#KEEP+1] = sp:install_write_tap(m, m + 3, "elya_bench_marker", function(offset, data, mask)
  if offset ~= m + 2 then return data end
  local v = data
  if v >= 0x100 then
    marks[#marks+1] = string.format("%d:%.0f", v, t2as(mac.time))
  end
  if v == 1 then
    t_start = t2as(mac.time)          -- last START wins (the console
                                      -- resets once during MAME boot)
  elseif v == 2 then
    if t_end == nil then t_end = t2as(mac.time) end
  elseif v == 4 then
    local st = sp:read_u8(CFG.status)
    if st ~= 0x5A then finish("status=BAD:" .. st .. "\n"); return data end
    local n = sp:read_u8(CFG.ntok)
    local toks = {}
    for i = 0, n - 1 do toks[#toks+1] = sp:read_u8(CFG.tok + i) end
    finish(string.format(
      "status=OK\nas_start=%.0f\nas_end=%.0f\nas_delta=%.0f\nexpert=%d\nntok=%d\ntokens=%s\n",
      t_start or -1, t_end or -1, (t_end or 0) - (t_start or 0),
      sp:read_u16(CFG.expert), n, table.concat(toks, ",")) ..
      "marks=" .. table.concat(marks, " ") .. "\n")
  end
  return data
end)

-- Safety net: if the ROM never finishes, stop after a wall-clock-free
-- emulated-time budget enforced by MAME's own -seconds_to_run.
KEEP[#KEEP+1] = emu.add_machine_stop_notifier(function()
  if not done then
    local f = io.open(CFG.out, "w")
    f:write("status=TIMEOUT\n")
    f:close()
  end
end)
'''


def read_symbols(symfile):
    syms = {}
    with open(symfile) as f:
        for line in f:
            parts = line.split()
            if len(parts) != 3:
                continue
            try:
                a = int(parts[0], 16)
            except ValueError:
                continue
            syms[parts[2]] = a & 0xFFFFFF
    return syms


def run_once(bench_dir, seconds=400):
    sym = read_symbols(os.path.join(bench_dir, "out", "symbol.txt"))
    need = ["bench_marker", "bench_status", "bench_ntok", "bench_tok",
            "bench_expert"]
    missing = [n for n in need if n not in sym]
    if missing:
        raise SystemExit("missing symbols: %s" % missing)

    outfd, outpath = tempfile.mkstemp(suffix=".txt")
    os.close(outfd)
    luafd, luapath = tempfile.mkstemp(suffix=".lua")
    os.close(luafd)
    with open(luapath, "w") as f:
        f.write("CFG = { marker=0x%06X, status=0x%06X, ntok=0x%06X, "
                "tok=0x%06X, expert=0x%06X, out=%r }\n" % (
                    sym["bench_marker"], sym["bench_status"], sym["bench_ntok"],
                    sym["bench_tok"], sym["bench_expert"], outpath))
        f.write(RUNNER_BODY)

    cmd = [MAME, "genesis",
           "-cart", os.path.join(bench_dir, "out", "rom.bin"),
           "-autoboot_script", luapath,
           "-sound", "none", "-video", "none", "-nothrottle",
           "-window", "-skip_gameinfo",
           "-seconds_to_run", str(seconds)]
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)
    res = {}
    with open(outpath) as f:
        for line in f:
            if "=" in line:
                k, v = line.strip().split("=", 1)
                res[k] = v
    os.unlink(outpath)
    os.unlink(luapath)
    if not res:
        raise SystemExit("no result; mame stderr:\n" + p.stderr[-4000:])
    if res.get("status") == "OK":
        res["cycles"] = int(round(float(res["as_delta"]) / 1e18 * M68K_CLOCK_NTSC))
        res["frames_equiv"] = round(int(res["cycles"]) / (M68K_CLOCK_NTSC / 60.0), 2)
        # per-forward-pass cycle profile from the progress markers
        prof = []
        mk = [x.split(":") for x in res.get("marks", "").split() if ":" in x]
        for i in range(1, len(mk)):
            prof.append((int(mk[i - 1][0]),
                         int(round((float(mk[i][1]) - float(mk[i - 1][1]))
                                   / 1e18 * M68K_CLOCK_NTSC))))
        if mk:
            prof.append((int(mk[-1][0]),
                         int(round((float(res["as_end"]) - float(mk[-1][1]))
                                   / 1e18 * M68K_CLOCK_NTSC))))
        res["profile"] = prof
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bench_dir")
    ap.add_argument("--label", default="")
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--seconds", type=int, default=400)
    ap.add_argument("--json", default="")
    a = ap.parse_args()

    label = a.label or os.path.basename(os.path.abspath(a.bench_dir))
    runs = []
    for i in range(a.repeat):
        r = run_once(a.bench_dir, a.seconds)
        runs.append(r)
        toks = r.get("tokens", "")
        text = "".join(chr(int(t)) for t in toks.split(",") if t) if toks else ""
        print("[%s run %d] status=%s cycles=%s (=%s frames @60Hz) expert=%s" % (
            label, i + 1, r.get("status"), r.get("cycles"),
            r.get("frames_equiv"), r.get("expert")))
        print("   ntok=%s tokens=%s" % (r.get("ntok"), toks))
        print("   text=%r" % text)
        sys.stdout.flush()
    if a.repeat > 1:
        cycs = {r.get("cycles") for r in runs}
        toks = {r.get("tokens") for r in runs}
        print("REPRODUCIBLE cycles: %s  (%d distinct: %s)" % (
            len(cycs) == 1, len(cycs), sorted(cycs)))
        print("REPRODUCIBLE tokens: %s" % (len(toks) == 1))
    if a.json:
        with open(a.json, "w") as f:
            json.dump({"label": label, "runs": runs}, f, indent=1)


if __name__ == "__main__":
    main()
