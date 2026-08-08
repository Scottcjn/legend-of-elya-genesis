#!/bin/bash
# Clean A/B of two model blobs against a byte-identical engine binary.
#
# Every run starts from `make clean`. That is not paranoia: SGDK tracks
# neither res/model.bin nor EXTRA_FLAGS as dependencies, and BOTH have
# already produced a ROM here that did not contain what the log claimed.
# A full rebuild costs ~15 s; a false null result costs a day.
#
# Usage: ./ab.sh [EXTRA_FLAGS] -- blob.bin [blob.bin ...]
set -euo pipefail
cd "$(dirname "$0")"

FLAGS=""
if [ "${1:-}" != "--" ]; then FLAGS="$1"; shift; fi
[ "${1:-}" = "--" ] && shift

REPEAT="${REPEAT:-3}"
for blob in "$@"; do
    cp "$blob" res/model.bin
    make clean >/dev/null 2>&1
    echo "=== $(basename "$blob")  EXTRA_FLAGS='$FLAGS' ==="
    make build EXTRA_FLAGS="$FLAGS" 2>&1 | grep "bench model:"
    md5sum out/rom.bin
    ( cd .. && python3 tools/mame/bench_run.py bench \
        --label "$(basename "$blob" .bin)" --repeat "$REPEAT" ) \
        | grep -E "run 1\]|REPRODUCIBLE"
done
