#!/usr/bin/env bash

BASE=~/mrcs
OUTPUT=plots


for R in .9 .95 .99; do
    for C in single complete average; do
        for T in .01 .02 .05; do
            for D in $BASE/*/ ; do
                for F in $D/*.csv ; do
                    FILENAME=$(basename "$F" .csv)
                    O="$OUTPUT/$FILENAME-$R-$C-$T.pdf"
                    echo -e "R2 = $R C = $C T = $T F = $FILENAME O = $O"
                    python test_rdp.py -i $F -o $O -r $R -c $C -t $T 
                done
            done
        done
    done
done


#