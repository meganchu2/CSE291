#!/bin/bash

trap "exit" INT

rm -f log_gp_strings
for file in ../benchmarks-selected/*.sl; do
	echo "===================="
	echo "====================" >> log_gp_strings
	echo "START BENCHMARK: $file"
	echo "START BENCHMARK: $file" >> log_gp_strings
	start=`date +%s%N`
	timeout 30m python gp.py "$file" >> log_gp_strings
        end=`date +%s%N`
	echo `expr $end - $start`
	echo `expr $end - $start` >> log_gp_strings
	echo "END BENCHMARK: $file"
	echo "END BENCHMARK: $file" >> log_gp_strings
	echo "===================="
	echo "====================" >> log_gp_strings
done
