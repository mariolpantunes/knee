#!/usr/bin/env bash

# load all the traces in the mrc folder
for f in $(ls /home/mantunes/mrcs | egrep -i 'w[0-9]*-(lru|arc).csv' ); do
    input='/home/mantunes/mrcs/'$f
    output='/home/mantunes/mrcs/'${f%.*}'_reduced.csv'
    #echo -e $input $output
    python -m examples.reduce_trace -i $input -o $output
done
