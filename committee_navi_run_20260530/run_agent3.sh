#!/bin/bash
export PATH=$HOME/.navi/bin:$PATH
LAB="$LABDIR"; LABEL="$1"; shift
OUT="$LAB/votes3/${LABEL}.vote.md"; ERR="$LAB/votes3/${LABEL}.err"
echo "[START $(date -u +%H:%M:%S)] $LABEL" >> "$LAB/votes3/_progress.log"
"$@" > "$OUT" 2> "$ERR"
echo "[DONE  $(date -u +%H:%M:%S)] $LABEL exit=$? bytes=$(wc -c < "$OUT")" >> "$LAB/votes3/_progress.log"
