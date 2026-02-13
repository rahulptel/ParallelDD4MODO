#!/bin/bash

for i in `seq 3 7` ; do
    rm multiobj_$i
    make clean
    make -j NUM_OBJS=$i
    cp multiobj multiobj$i
done
