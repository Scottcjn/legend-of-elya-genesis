#!/usr/bin/env python3
"""
make_music.py — original VGM chiptune for ELYA INTO DREAMS.

100% original composition for the real Genesis sound hardware: YM2612 FM
(6 channels) + SN76489 PSG (3 square + noise). No sampled, borrowed, or
"in the style of a specific track" material — SEGA's music is not public
domain and not licensed for fan use, so every note here is ours.

Output: res/dream_theme.vgm, playable by SGDK's XGM driver.

Mood brief: the Dreamscape. Slow, weightless, bittersweet — a lullaby for
something that knows it is dreaming. D minor, 84 BPM, 6/8 lilt.
  FM1  bell/e-piano lead      (the melody she hums)
  FM2  glass pad, high        (the starfield)
  FM3  warm pad, low          (the floor)
  FM4  plucked arpeggio       (motion)
  FM6  bass                   (gravity)
  PSG1 sparkle counter-melody (twinkle)
  PSG3+noise  soft brush      (pulse)
"""
import struct, math
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "res" / "dream_theme.vgm"

SR = 44100
BPM = 84
BEAT = SR * 60 // BPM          # samples per quarter note

# ── VGM command stream ────────────────────────────────────────────────
cmds = bytearray()
def ym0(reg, val): cmds.extend((0x52, reg & 0xFF, val & 0xFF))   # port 0
def ym1(reg, val): cmds.extend((0x53, reg & 0xFF, val & 0xFF))   # port 1
def psg(val):      cmds.extend((0x50, val & 0xFF))
def wait(n):
    while n > 65535:
        cmds.extend(b"\x61" + struct.pack("<H", 65535)); n -= 65535
    if n > 0:
        cmds.extend(b"\x61" + struct.pack("<H", n))

def ymw(ch, reg, val):
    """Write a per-channel register, routing to the right port."""
    (ym0 if ch < 3 else ym1)(reg + (ch % 3), val)

def ymop(ch, op, reg, val):
    """Per-operator register. Operator order in registers is 0,2,1,3."""
    (ym0 if ch < 3 else ym1)(reg + (ch % 3) + 4 * op, val)

# ── FM patch definition ───────────────────────────────────────────────
# Each operator: (DT, MUL, TL, RS, AR, AM, D1R, D2R, D1L, RR, SSG)
def set_patch(ch, alg, fb, ops):
    ymw(ch, 0xB0, ((fb & 7) << 3) | (alg & 7))
    ymw(ch, 0xB4, 0xC0)                       # L+R on (silence if omitted!)
    for op, (dt, mul, tl, rs, ar, am, d1r, d2r, d1l, rr, ssg) in enumerate(ops):
        ymop(ch, op, 0x30, ((dt & 7) << 4) | (mul & 15))
        ymop(ch, op, 0x40, tl & 0x7F)
        ymop(ch, op, 0x50, ((rs & 3) << 6) | (ar & 31))
        ymop(ch, op, 0x60, ((am & 1) << 7) | (d1r & 31))
        ymop(ch, op, 0x70, d2r & 31)
        ymop(ch, op, 0x80, ((d1l & 15) << 4) | (rr & 15))
        ymop(ch, op, 0x90, ssg & 15)

#            DT MUL  TL RS  AR AM D1R D2R D1L RR SSG
BELL = [(0,  1, 34, 0, 31, 0,  9,  6,  3, 7, 0),   # carrier-ish
        (0,  4, 26, 0, 31, 0, 14,  8,  6, 7, 0),
        (0,  2, 40, 0, 31, 0, 10,  5,  4, 7, 0),
        (0,  1,  8, 0, 31, 0,  8,  4,  2, 6, 0)]
GLASS= [(1,  2, 44, 0, 18, 0,  6,  3,  2, 5, 0),
        (2,  1, 30, 0, 16, 0,  5,  2,  1, 5, 0),
        (0,  4, 46, 0, 17, 0,  6,  3,  2, 5, 0),
        (0,  1, 14, 0, 15, 0,  4,  2,  1, 4, 0)]
WARM = [(0,  1, 40, 0, 14, 0,  5,  2,  1, 4, 0),
        (0,  1, 28, 0, 13, 0,  4,  2,  1, 4, 0),
        (3,  2, 44, 0, 14, 0,  5,  2,  1, 4, 0),
        (0,  1, 16, 0, 12, 0,  4,  1,  1, 3, 0)]
PLUCK= [(0,  1, 32, 1, 31, 0, 16, 10,  5, 9, 0),
        (0,  3, 30, 1, 31, 0, 18, 12,  7, 9, 0),
        (0,  1, 38, 1, 31, 0, 16, 10,  5, 9, 0),
        (0,  1, 10, 1, 31, 0, 14,  9,  4, 8, 0)]
BASS = [(0,  1, 30, 0, 31, 0, 10,  4,  2, 6, 0),
        (0,  1, 18, 0, 31, 0, 12,  5,  3, 6, 0),
        (0,  2, 36, 0, 31, 0, 10,  4,  2, 6, 0),
        (0,  1,  6, 0, 31, 0,  9,  3,  2, 5, 0)]

# ── note -> YM2612 F-number / block ───────────────────────────────────
NOTE = {'C':0,'C#':1,'D':2,'D#':3,'E':4,'F':5,'F#':6,'G':7,'G#':8,
        'A':9,'A#':10,'B':11}

def freq_of(name):
    p, octv = name[:-1], int(name[-1])
    return 440.0 * 2 ** ((NOTE[p] + (octv - 4) * 12 - 9) / 12.0)

def fnum_block(f):
    """YM2612: fnum = f * 2^20 / clock * 2^(21-block); clock ~7.67MHz."""
    block = 4
    while f < 261.6 and block > 0:
        f *= 2; block -= 1
    while f >= 523.3 and block < 7:
        f /= 2; block += 1
    fnum = int(round(f * 1048576.0 / 7670453.0 * 2))
    return max(0, min(2047, fnum)), block

def key_on(ch, name):
    fnum, block = fnum_block(freq_of(name))
    ymw(ch, 0xA4, ((block & 7) << 3) | ((fnum >> 8) & 7))
    ymw(ch, 0xA0, fnum & 0xFF)
    ym0(0x28, 0xF0 | (ch if ch < 3 else (ch + 1)))   # all 4 slots on

def key_off(ch):
    ym0(0x28, 0x00 | (ch if ch < 3 else (ch + 1)))

# ── PSG (SN76489) ─────────────────────────────────────────────────────
def psg_note(chan, name, vol=6):
    f = freq_of(name)
    n = max(1, min(1023, int(round(3579545.0 / (32.0 * f)))))
    psg(0x80 | (chan << 5) | (n & 0x0F))
    psg((n >> 4) & 0x3F)
    psg(0x90 | (chan << 5) | (vol & 0x0F))

def psg_off(chan):
    psg(0x9F | (chan << 5))

# ── The composition ───────────────────────────────────────────────────
# D natural minor, 6/8. Phrases are 6 eighth-notes long.
E = BEAT // 2                                   # eighth note

MELODY = [  # (note or None, eighths)
    ('A4',3), ('F4',1), ('G4',1), ('A4',1),
    ('D5',3), ('C5',1), ('A4',2),
    ('F4',3), ('G4',1), ('A4',1), ('C5',1),
    ('D5',4), (None,2),
    ('E5',3), ('D5',1), ('C5',1), ('A4',1),
    ('G4',3), ('F4',1), ('D4',2),
    ('F4',2), ('G4',2), ('A4',2),
    ('D4',6),
]
BASSLINE = ['D2','D2','F2','F2','A2','A2','G2','G2',
            'F2','F2','C2','C2','A2','A2','D2','D2']
ARPS = ['D4','F4','A4','F4', 'C4','F4','A4','F4',
        'A3','D4','F4','D4', 'G3','B3','D4','B3']
PSG_SPARKLE = ['A5','D6','F6','D6','C6','A5','G5','F5']

def init_chips():
    ym0(0x22, 0x00)          # LFO off
    ym0(0x27, 0x00)          # normal ch3 mode
    ym0(0x2B, 0x00)          # DAC off
    for ch in range(6):
        key_off(ch)
    for c in range(4):
        psg(0x9F | (c << 5))  # all PSG silent
    set_patch(0, 5, 4, BELL)   # FM1 lead
    set_patch(1, 4, 3, GLASS)  # FM2 high pad
    set_patch(2, 4, 2, WARM)   # FM3 low pad
    set_patch(3, 5, 5, PLUCK)  # FM4 arpeggio
    set_patch(5, 2, 6, BASS)   # FM6 bass

def render(loop_bars=2):
    """Emit the theme. Returns the sample offset where the loop begins."""
    init_chips()
    wait(BEAT)                               # breath before the first note

    loop_point = len(cmds)
    mi = ai = si = 0
    for bar in range(16):
        # pads move once per bar
        key_on(2, BASSLINE[bar])                       # warm low pad
        key_on(1, ['A4','A4','C5','C5','D5','D5','C5','C5',
                   'A4','A4','G4','G4','F4','F4','A4','A4'][bar])
        key_on(5, BASSLINE[bar])                       # bass
        for e in range(6):                             # 6 eighths per bar
            # melody: consume the phrase list in eighth-note units
            if mi < len(MELODY):
                n, dur = MELODY[mi]
                if not hasattr(render, "_held"):
                    render._held = 0
                if render._held == 0:
                    if n: key_on(0, n)
                    else: key_off(0)
                render._held += 1
                if render._held >= dur:
                    render._held = 0
                    mi += 1
                    if mi >= len(MELODY):
                        mi = 0
            # arpeggio every eighth
            key_on(3, ARPS[(ai := (ai + 1)) % len(ARPS)])
            # sparkle on the off-beats only — restraint reads as space
            if e in (1, 4):
                psg_note(0, PSG_SPARKLE[(si := (si + 1)) % len(PSG_SPARKLE)],
                         vol=9)
            elif e in (2, 5):
                psg_off(0)
            wait(E)
        key_off(3)
    for ch in range(6):
        key_off(ch)
    for c in range(4):
        psg(0x9F | (c << 5))
    return loop_point

loop_off = render()
total_samples = 0
i = 0
while i < len(cmds):                      # count wait samples for the header
    c = cmds[i]
    if c == 0x61:
        total_samples += struct.unpack("<H", cmds[i+1:i+3])[0]; i += 3
    elif c in (0x52, 0x53): i += 3
    elif c == 0x50: i += 2
    else: i += 1
cmds.append(0x66)                         # end of stream

# ── VGM 1.61 header (0x100 bytes) ─────────────────────────────────────
hdr = bytearray(0x100)
hdr[0x00:0x04] = b"Vgm "
hdr[0x08:0x0C] = struct.pack("<I", 0x161)
hdr[0x0C:0x10] = struct.pack("<I", 3579545)          # SN76489 clock
hdr[0x18:0x1C] = struct.pack("<I", total_samples)    # total samples
hdr[0x1C:0x20] = struct.pack("<I", 0x100 + loop_off - 0x1C)  # loop offset
hdr[0x20:0x24] = struct.pack("<I", total_samples)    # loop # samples
hdr[0x24:0x28] = struct.pack("<I", 60)               # rate
hdr[0x28:0x2A] = struct.pack("<H", 0x0009)           # SN feedback
hdr[0x2A] = 16                                       # SN shift width
hdr[0x2C:0x30] = struct.pack("<I", 7670453)          # YM2612 clock
hdr[0x34:0x38] = struct.pack("<I", 0x100 - 0x34)     # VGM data offset
hdr[0x04:0x08] = struct.pack("<I", 0x100 + len(cmds) - 4)   # EOF offset

OUT.write_bytes(bytes(hdr) + bytes(cmds))
print(f"{OUT}: {len(cmds):,} bytes of commands, "
      f"{total_samples/44100:.1f}s, loop at {loop_off}")
