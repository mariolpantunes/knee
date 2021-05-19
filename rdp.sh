#!/usr/bin/env bash

BASE=~/mrcs
OUTPUT=plots

for D in $BASE/*/ ; do
    for F in $D/*.csv ; do
        for R in .9 .95 .99; do
            for C in average single complete; do
                for T in .01 .02 .05; do
                    FILENAME=$(basename "$F" .csv)
                    O="$OUTPUT/$FILENAME-$R-$C-$T.pdf"
                    echo -e "R2 = $R C = $C T = $T F = $FILENAME O = $O"
                    python eval_rdp.py -i $F -o $O -r $R -c $C -t $T 
                done
            done
        done
    done
done

pdfunite $OUTPUT/*.pdf report.pdf
