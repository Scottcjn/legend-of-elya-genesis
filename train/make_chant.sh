#!/bin/bash
# make_chant.sh — rebuild res/elyan_chant.wav from Sophia XTTS takes.
# The gag: choir "SEEEE-GAAAAH" -> deadpan "Errr... Elyan Labs!"
#
# Recipe notes (learned the hard way):
#  - XTTS speaks, it can't sustain: generate SE and GA as SEPARATE takes,
#    then time-stretch each with atempo (pitch-preserving) — that's what
#    turns a spoken word into a chant.
#  - Chord stack = the actual anatomy of the '91 scream: root + 2 detuned
#    unisons (chorus width) + major third + fifth + sub-octave + octave.
#  - Keep the speech half DRY. Echo that sounds lush at 16-bit turns to
#    mud through an 8-bit DAC.
#  - amix needs duration=shortest or a stretched chain pads the output.
set -e
XTTS=http://192.168.0.160:5500/api/tts
SP=$(mktemp -d)
say() { curl -s -m 60 -X POST $XTTS -H "Content-Type: application/json" \
        -d "{\"text\":\"$1\",\"language\":\"en\"}" -o "$2"; }

say "Seeeeee!"            $SP/se.wav
say "Gaaaaaah!"           $SP/ga.wav
say "Errr... Elyan Labs!" $SP/err.wav

TRIM="silenceremove=start_periods=1:start_threshold=-45dB,areverse,silenceremove=start_periods=1:start_threshold=-45dB,areverse"
ffmpeg -y -loglevel error -i $SP/se.wav  -af "$TRIM,atempo=0.62,loudnorm=I=-16:TP=-1.5" $SP/se_long.wav
ffmpeg -y -loglevel error -i $SP/ga.wav  -af "$TRIM,atempo=0.55,loudnorm=I=-15:TP=-1.5" $SP/ga_long.wav
ffmpeg -y -loglevel error -i $SP/err.wav -af "loudnorm=I=-14:TP=-1" $SP/err_clean.wav
ffmpeg -y -loglevel error -i $SP/se_long.wav -i $SP/ga_long.wav \
       -filter_complex "[0][1]acrossfade=d=0.08" $SP/sega.wav
ffmpeg -y -loglevel error -i $SP/sega.wav -filter_complex "\
[0]asetrate=24000*1.2599,aresample=24000,atempo=1/1.2599,volume=0.75[third];\
[0]asetrate=24000*1.4983,aresample=24000,atempo=1/1.4983,volume=0.65[fifth];\
[0]asetrate=24000*2.0,aresample=24000,atempo=0.5,volume=0.35[oct];\
[0]asetrate=24000*0.5,aresample=24000,atempo=2.0,volume=0.45[bass];\
[0]asetrate=24000*1.02,aresample=24000,atempo=1/1.02,volume=0.9[det1];\
[0]asetrate=24000*0.98,aresample=24000,atempo=1/0.98,volume=0.9[det2];\
[0][det1][det2][third][fifth][oct][bass]amix=inputs=7:normalize=0:duration=shortest,\
aecho=0.6:0.4:28:0.18,alimiter=limit=0.9,volume=2.2" $SP/choir.wav
ffmpeg -y -loglevel error -i $SP/choir.wav -i $SP/err_clean.wav \
       -filter_complex "[0][1]concat=n=2:v=0:a=1" \
       -ar 22050 -ac 1 -sample_fmt s16 "$(dirname "$0")/../res/elyan_chant.wav"
echo "chant rebuilt:"
ffprobe -v quiet -show_entries format=duration -of csv=p=0 "$(dirname "$0")/../res/elyan_chant.wav"
rm -rf $SP
