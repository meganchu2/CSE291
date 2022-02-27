#!/bin/bash

trap "exit" INT

rm -f log_strings
for file in benchmarks-master/comp/2018/PBE_Strings_Track/*.sl; do
	echo "===================="
	echo "====================" >> log_strings
	echo "START BENCHMARK: $file"
	echo "START BENCHMARK: $file" >> log_strings
	#PYTHONPATH=$PYTHONPATH:../thirdparty/libeusolver/build/:../thirdparty/z3/build/python:../thirdparty/bitstring-3.1.3/ timeout "$1" python3 core/solvers.py icfp $file log:wq
	start=`date +%s%N`
	timeout 30m ./eusolver "$file" >> log_strings
        end=`date +%s%N`
	echo `expr $end - $start`
	echo `expr $end - $start` >> log_strings
	echo "END BENCHMARK: $file"
	echo "END BENCHMARK: $file" >> log_strings
	echo "===================="
	echo "====================" >> log_strings
done
