import argparse
import logging
import math
import random
from os import listdir
from os.path import basename, isdir, isfile, join

from grammar import Grammar
from program import FuncNode
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
        "--benchmark_dir",
        default=True,
        type=bool
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

    # hyperparameters
    parser.add_argument(
        "--init_max_depth",
        default=4,
        type=int,
    )
    parser.add_argument(
        "--num_generation",
        default=8,
        type=int,
    )
    parser.add_argument(
        "--weight_productions",
        default=8,
        type=int,
    )
    parser.add_argument(
        "--exp_functions",
        default=2,
        type=int,
    )
    parser.add_argument(
        "--exp_max_arity",
        default=2,
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

    args = parser.parse_args()
    return args


def get_bmfiles(bms, dir):
    if dir:
        return [join(bms, f) for f in listdir(bms) if isfile(join(bms, f))]
    with open(bms, "r") as f:
        bmfiles = [line.strip() for line in f]
    return bmfiles


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
    examples = []
    for c in constraints:
        ins = [i[1] for i in c[1][1][1:]]
        out = c[1][2][1]
        examples.append((ins, out))
    return grammar, variables, examples


def get_hyperparameters(grammar, args):
    num_prod = len(grammar.all_rules)
    all_rhs = [j.rhs for i in grammar.all_rules.values() for j in i]
    num_func = sum([isinstance(i, FuncNode) for i in all_rhs])
    max_arity = max([len(i.children) for i in all_rhs])
    pop_size = math.floor(
        num_prod * args.weight_productions * (num_func ** args.exp_functions) * (max_arity ** args.exp_max_arity)
    )
    num_selection = math.floor(pop_size * args.selection_ratio)
    num_offspring = math.floor(pop_size * args.offspring_ratio)
    return pop_size, num_selection, num_offspring


def solve(bm, args):
    bm_name = basename(bm).split(".")[0]
    logger.info(f"benchmark: {bm_name}")
    try:
        parsed_bm = parse_benchmark(bm)
    except Exception as _:
        logger.debug("Incompatible benchmark format")
        return None
    grammar = Grammar(parsed_bm[0])
    variables = parsed_bm[1]
    examples = parsed_bm[2]
    pop_size, num_selection, num_offspring = get_hyperparameters(grammar, args)
    logger.debug(f"population_size: {pop_size}, num_selection: {num_selection}, num_offspring: {num_offspring}")


if __name__ == "__main__":
    args = parse_args()
    random.seed(args.seed)
    set_logger(args.log_file, args.debug)
    logger.info(vars(args))

    if args.benchmark_dir:
        assert isdir(args.benchmarks)
    else:
        assert isfile(args.benchmarks)
    bmfiles = get_bmfiles(args.benchmarks, args.benchmark_dir)
    logger.debug(f"Evaluate on {len(bmfiles)} benchmarks")

    for bm in bmfiles:
        solve(bm, args)
