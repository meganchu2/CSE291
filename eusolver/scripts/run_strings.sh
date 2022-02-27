#!/bin/bash

trap "exit" INT

rm -f log_strings
for file in benchmarks-master/comp/2018/PBE_Strings_Track/*.sl; do
	echo "====================" >> log_strings
	echo "START BENCHMARK: $file" >> log_strings
	#PYTHONPATH=$PYTHONPATH:../thirdparty/libeusolver/build/:../thirdparty/z3/build/python:../thirdparty/bitstring-3.1.3/ timeout "$1" python3 core/solvers.py icfp $file log
	timeout 30m time ./eusolver "$file" >> log_strings
	echo "END BENCHMARK: $file" >> log_strings
	echo "====================" >> log_strings
done
