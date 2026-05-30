#!/bin/bash
# $1 = agent label, rest = command
export PATH=$HOME/.navi/bin:$PATH
LAB="$LABDIR"
LABEL="$1"; shift
OUT="$LAB/votes/${LABEL}.ideate.md"
ERR="$LAB/votes/${LABEL}.err"
echo "[START $(date -u +%H:%M:%S)] $LABEL" >> "$LAB/votes/_progress.log"
"$@" > "$OUT" 2> "$ERR"
echo "[DONE  $(date -u +%H:%M:%S)] $LABEL exit=$? bytes=$(wc -c < "$OUT")" >> "$LAB/votes/_progress.log"
