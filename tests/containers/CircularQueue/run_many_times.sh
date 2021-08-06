#!/usr/bin/env bash

// SPDX-FileCopyrightText: 2021 Benjamin Brock
//
// SPDX-License-Identifier: BSD-3-Clause

for i in {1..10}
do
    echo run ${i}
    mpirun -n 4 ./CircularQueue03
    echo
done
