#!/usr/bin/env bash
for i in {1..10}
do
    echo run ${i}
    mpirun -n 4 ./simplify_mpi_issue
    echo
done