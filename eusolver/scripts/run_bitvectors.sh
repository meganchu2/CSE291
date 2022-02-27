#!/bin/bash

trap "exit" INT

rm -f log_bitvectors
for file in benchmarks-master/comp/2018/PBE_BV_Track/*.sl; do
	echo "===================="
	echo "====================" >> log_bitvectors
	echo "START BENCHMARK: $file"
	echo "START BENCHMARK: $file" >> log_bitvectors
	#PYTHONPATH=$PYTHONPATH:../thirdparty/libeusolver/build/:../thirdparty/z3/build/python:../thirdparty/bitstring-3.1.3/ timeout "$1" python3 core/solvers.py icfp $file log:wq
	start=`date +%s%N`
	timeout 30m ./eusolver "$file" >> log_bitvectors
        end=`date +%s%N`
	echo `expr $end - $start`
	echo `expr $end - $start` >> log_bitvectors
	echo "END BENCHMARK: $file"
	echo "END BENCHMARK: $file" >> log_bitvectors
	echo "===================="
	echo "====================" >> log_bitvectors
done
