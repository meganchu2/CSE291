import argparse
import logging
import math
import random
from os import listdir
from os.path import basename, isdir, isfile, join

from gp import genetic_programming
from grammar import Grammar
from program import FuncNode, print_ast
from sexp import sexp
from utils import set_logger

logger = logging.getLogger()

def parse_args():
    parser = argparse.ArgumentParser()

    # data
    parser.add_argument(
        "--benchmarks",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--benchmark_list",
        action="store_true"
    )

    # logging
    parser.add_argument(
        "--log_file",
        default="log.txt",
        type=str,
    )
    parser.add_argument(
        "--debug",
        action="store_true"
    )

    # methods
    parser.add_argument(
        "--fitness",
        default="match",
        type=str,
    )
    parser.add_argument(
        "--select",
        default="best",
        type=str,
    )
    parser.add_argument(
        "--breed",
        default="random",
        type=str,
    )

    # hyperparameters
    parser.add_argument(
        "--max_depth",
        default=4,
        type=int,
    )
    parser.add_argument(
        "--num_generation",
        default=8,
        type=int,
    )
    parser.add_argument(
        "--pop_size",
        default=2000,
        type=int,
    )
    parser.add_argument(
        "--selection_ratio",
        default=0.5,
        type=float,
    )
    parser.add_argument(
        "--offspring_ratio",
        default=0.8,
        type=float,
    )
    parser.add_argument(
        "--mutation_prob",
        default=0.5,
        type=float,
    )
    parser.add_argument(
        "--crossover_prob",
        default=0.5,
        type=float,
    )
    parser.add_argument(
        "--seed",
        default=42,
        type=int,
    )
    parser.add_argument(
        "--timeout",
        default=1800,
        type=int,
    )

    args = parser.parse_args()
    return args


def get_bmfiles(bms, bm_list):
    if bm_list:
        with open(bms, "r") as f:
            bmfiles = [line.strip() for line in f]
        return bmfiles
    return [join(bms, f) for f in listdir(bms) if isfile(join(bms, f))]


def parse_benchmark(bm):
    no_comments = "("
    with open(bm, "r") as f:
        for line in f:
            no_comments += line.split(";", 1)[0]
    no_comments += ")"

    bm_parsed = sexp.parse_string(no_comments, parse_all=True).asList()[0]
    synth = [i for i in bm_parsed if i[0] == "synth-fun"][0]
    variables = [i[0] for i in synth[2]]
    grammar = synth[4]
    constraints = [i for i in bm_parsed if i[0] == "constraint"]
    ins, outs = [], []
    for c in constraints:
        i = [a[1] for a in c[1][1][1:]]
        ins.append({k: v for (k, v) in zip(variables, i)})
        o = c[1][2][1]
        outs.append(o)
    return grammar, {"in": ins, "out": outs}


def get_hyperparameters(grammar, args):
    num_selection = math.floor(args.pop_size * args.selection_ratio)
    num_offspring = math.floor(args.pop_size * args.offspring_ratio)
    return args.pop_size, num_selection, num_offspring


def solve(bm, args):
    bm_name = basename(bm).split(".")[0]
    logger.info(f"benchmark: {bm_name}")
    try:
        parsed_bm = parse_benchmark(bm)
    except Exception as _:
        logger.debug("Incompatible benchmark format")
        return None
    grammar = Grammar(parsed_bm[0])
    examples = parsed_bm[1]
    pop_size, num_selection, num_offspring = get_hyperparameters(grammar, args)
    logger.debug(f"population_size: {pop_size}, num_selection: {num_selection}, num_offspring: {num_offspring}")

    logger.debug("Start GP")
    result = genetic_programming(
        grammar,
        args,
        (pop_size, num_selection, num_offspring),
        examples,
    )
    if result:
        solution = max(zip(result, [p.size() for p in result]), key=lambda x: x[1])[0]
        logger.info(f"Final solution: {print_ast(solution)}")
        logger.info(f"Solution size: {solution.size()}")
    else:
        logger.info("Unable to find a solution")


if __name__ == "__main__":
    args = parse_args()
    random.seed(args.seed)
    set_logger(args.log_file, args.debug)
    logger.info(vars(args))

    if args.benchmark_list:
        assert isfile(args.benchmarks)
    else:
        assert isdir(args.benchmarks)
    bmfiles = get_bmfiles(args.benchmarks, args.benchmark_list)
    logger.debug(f"Evaluate on {len(bmfiles)} benchmarks")

    for bm in bmfiles:
        solve(bm, args)
