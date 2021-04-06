#!/usr/bin/env bash

BASE=~/mrcs

for D in $BASE/*/ ; do
    echo $D
    for F in $D/*.csv ; do
        python main.py -i $F
    done
done