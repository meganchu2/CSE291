# CSE291
Program Synthesis Project

## How to run the GPSolver
1. Please make sure that you are using python3


2. `cd` to the `~gpsolver/src/` directory to run the file `run_exp.py` (this executes the GPSolver)

3. You must specify the benchmarks to run


For our experiments, we made the following specifications:
```
python run_exp.py --benchmarks ../benchmarks-master/comp/2018/PBE_Strings_Track --fitness levenshtein --select lexicase --breed union --num_generation 1000 --selection_ratio 0.25 --offspring_ratio 1.5 --seed 1729 --log_file log1729.txt
```



