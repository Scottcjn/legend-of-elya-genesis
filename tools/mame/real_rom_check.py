#!/usr/bin/env python3
"""End-to-end check on the REAL game ROM, not the bench.

Boots out/rom.bin in MAME's `genesis` driver, presses P1 A on the emulated
pad to ask the default prompt, and samples `lastTok` every time `genCount`
advances. Prints the generated string.

This exists because the bench ROM is a stripped shell: it proves the engine
is right, not that the shipped cartridge still boots, still passes
`eg_init`, and still answers. A blob-format change has to clear both.

Usage: real_rom_check.py [rom_dir] [--tokens N]
"""
import argparse
import os
import subprocess
import sys
import tempfile

MAME = "/usr/games/mame"

RUNNER = r'''
local mac = manager.machine
local sp  = mac.devices[":maincpu"].spaces["program"]
-- MAME's Lua bindings garbage-collect notifier/tap handles that nothing
-- references; dropping them silently stops every callback and looks
-- exactly like a hung ROM.
KEEP = {}

local frame, last_gc, out, pressed = 0, -1, {}, false

KEEP[#KEEP+1] = emu.add_machine_frame_notifier(function()
  frame = frame + 1
  -- let the intro run, then hold P1 A for a few frames
  if frame > CFG.press and frame <= CFG.press + 8 then
    if not pressed then
      mac.ioport.ports[":MD1_3B"]:field("P1 A"):set_value(1)
      pressed = true
    end
  elseif pressed and frame > CFG.press + 8 then
    mac.ioport.ports[":MD1_3B"]:field("P1 A"):set_value(0)
    pressed = false
  end

  local gc = sp:read_u16(CFG.gencount)
  if gc ~= last_gc then
    if gc > 0 and gc <= CFG.ntok then
      out[#out+1] = sp:read_u8(CFG.lasttok)
    end
    last_gc = gc
  end

  if #out >= CFG.ntok or frame > CFG.limit then
    local f = io.open(CFG.out, "w")
    f:write("ok=1\nframes=" .. frame .. "\ntokens=" ..
            table.concat(out, ",") .. "\n")
    f:close()
    mac:exit()
  end
end)
'''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("rom_dir", nargs="?", default=".")
    ap.add_argument("--tokens", type=int, default=24)
    ap.add_argument("--press", type=int, default=1200)
    ap.add_argument("--limit", type=int, default=40000)
    ap.add_argument("--seconds", type=int, default=900)
    a = ap.parse_args()

    syms = {}
    with open(os.path.join(a.rom_dir, "out", "symbol.txt")) as f:
        for line in f:
            p = line.split()
            if len(p) == 3:
                try:
                    syms[p[2]] = int(p[0], 16) & 0xFFFFFF
                except ValueError:
                    pass
    for n in ("lastTok", "genCount"):
        if n not in syms:
            raise SystemExit("missing symbol: " + n)

    outfd, outpath = tempfile.mkstemp(suffix=".txt")
    os.close(outfd)
    luafd, luapath = tempfile.mkstemp(suffix=".lua")
    os.close(luafd)
    with open(luapath, "w") as f:
        f.write("CFG = { lasttok=0x%06X, gencount=0x%06X, ntok=%d, "
                "press=%d, limit=%d, out=%r }\n" % (
                    syms["lastTok"], syms["genCount"], a.tokens,
                    a.press, a.limit, outpath))
        f.write(RUNNER)

    cmd = [MAME, "genesis", "-cart", os.path.join(a.rom_dir, "out", "rom.bin"),
           "-autoboot_script", luapath, "-sound", "none", "-video", "none",
           "-nothrottle", "-window", "-skip_gameinfo",
           "-seconds_to_run", str(a.seconds)]
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
        raise SystemExit("no result; mame stderr:\n" + p.stderr[-3000:])
    toks = [int(t) for t in res.get("tokens", "").split(",") if t]
    print("frames=%s ntok=%d" % (res.get("frames"), len(toks)))
    print("tokens=%s" % ",".join(str(t) for t in toks))
    print("text=%r" % "".join(chr(t) for t in toks))
    return 0


if __name__ == "__main__":
    sys.exit(main())
