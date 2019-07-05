#!/usr/bin/env bash
for i in {1..10}
do
    echo run ${i}
    mpirun -n 4 ./CircularQueue03
    echo
done
