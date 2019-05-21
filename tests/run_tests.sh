#!/usr/bin/env bash

base_dir=$PWD

user_backends=${BASH_ARGV[*]}

if hash gmake 2> /dev/null
then
  make_cmd='gmake'
else
  make_cmd='make'
fi

if [ "$user_backends" == 'all' ] || [ "$user_backends" == '' ]
then
  backends='mpi shmem gasnet_ex'
else
  backends=$user_backends
fi

echo "Testing backends $backends"

for backend in $backends
do
  for i in $(find . -iname "Makefile")
  do
    cd $(dirname $i)
    BCL_BACKEND=$backend $make_cmd clean test
    cd $base_dir
  done
done
