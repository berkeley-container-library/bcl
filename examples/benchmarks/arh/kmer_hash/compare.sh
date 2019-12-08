#!/bin/bash -l
##SBATCH -A mp215
##SBATCH -t 10:00
##SBATCH -J kmer_hash
##SBATCH -o kmer_hash.o%j
##SBATCH -p debug
##SBATCH -C haswell
##SBATCH -N 32

targets1=(kmer_hash_agg)
targets2=(kmer_hash_stl kmer_hash_upcxx)

node_num=32
dataset=$SCRATCH/my_datasets/human-chr14-synthetic.txt

for target in ${targets1[*]}
do
    echo "srun -N $node_num -n $((2*node_num)) $target ${*:2}"
	srun -N $node_num -n $((node_num*2)) $target $dataset
done

for target in ${targets2[*]}
do
    echo "srun -N $node_num -n $((32*node_num)) $target ${*:2}"
	srun -N $node_num -n $((node_num*32)) $target $dataset
done

# -N 2
# large
# agg: 0.9 31.4
# naive: 33.5 69.6
# stl: 4.6 10.3
# upcxx: 2.9 7.6

# chr14
# agg: 2.9 102.1
# stl: 17.7 34.6
# upcxx: 15.2 30.6

# -N 32
# large
# agg: 0.058 4.60
# stl:
# upcxx: 0.17 0.74

# chr14
# agg: 0.26 11.6
# stl:
# upcxx: 0.62 2.0