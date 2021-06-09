#!/usr/bin/env bash

BASE=~/mrcs
OUTPUT=plots_optimal

for D in $BASE/*/ ; do
    for F in $D/*.csv ; do
        FILENAME=$(basename "$F" .csv)
        O="$OUTPUT/$FILENAME.pdf"
        python optimal_knee.py -i $F -a knee
    done
done