#!/usr/bin/env bash

BASE=~/mrcs
OUTPUT=plots

for D in $BASE/*/ ; do
    for F in $D/*.csv ; do
        for R in .95 .99; do
            for C in average; do
                for T in .02; do
                    FILENAME=$(basename "$F" .csv)
                    O="$OUTPUT/$FILENAME-$R-$C-$T.pdf"
                    echo -e "R2 = $R C = $C T = $T F = $FILENAME O = $O"
                    python test_rdp.py -i $F -o $O -r $R -c $C -t $T 
                done
            done
        done
    done
done
