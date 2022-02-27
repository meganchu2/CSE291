#!/bin/bash

trap "exit" INT

rm -f log_bitvectors
for file in benchmarks-master/comp/2018/PBE_BV_Track/*.sl; do
	echo "====================" >> log_bitvectors
	echo "START BENCHMARK: $file" >> log_bitvectors
	#PYTHONPATH=$PYTHONPATH:../thirdparty/libeusolver/build/:../thirdparty/z3/build/python:../thirdparty/bitstring-3.1.3/ timeout "$1" python3 core/solvers.py icfp $file log
	timeout 30m time ./eusolver "$file" >> log_bitvectors
	echo "END BENCHMARK: $file" >> log_bitvectors
	echo "====================" >> log_bitvectors
done
