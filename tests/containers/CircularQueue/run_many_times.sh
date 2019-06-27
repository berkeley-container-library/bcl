#!/usr/bin/env bash
for i in 1 .. 30
do
    mpirun -n 4 ./CircularQueue02
done