import argparse
import logging
import math
import random
from os import listdir
from os.path import isdir, isfile, join

from program import FuncNode

logger = logging.getLogger(__name__)

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
    parser.add_argument(
        "--log_file",
        default=None,
        type=str,
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


def get_logger(logger, log_file):
    logger.setLevel(logging.DEBUG)
    msg_fmt = "%(asctime)s - %(levelname)-5s - %(name)s -   %(message)s"
    date_fmt = "%m/%d/%Y %H:%M:%S"

    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)

    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)

    formatter = logging.Formatter(fmt=msg_fmt, datefmt=date_fmt)
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


def logging_setup(args):
    global logger
    logger = get_logger(
        logger,
        args.log_file if args.log_file else "log.txt"
    )


def get_bmfiles(bms, dir):
    if dir:
        return [join(bms, f) for f in listdir(bms) if isfile(join(bms, f))]
    with open(bms, "r") as f:
        bmfiles = [line.strip() for line in f]
    return bmfiles


def get_hyperparameters(grammar, args):
    num_prod = len(grammar.all_rules)
    all_rhs = [j.rhs for i in grammar.all_rules for j in i]
    num_func = sum([isinstance(i, FuncNode) for i in all_rhs])
    max_arity = max([len(i.children) for i in all_rhs])
    pop_size = math.floor(
        num_prod * args.weight_productions * (num_func ** args.exp_functions) * (max_arity ** args.exp_max_arity)
    )
    num_selection = math.floor(pop_size * args.selection_ratio)
    num_offspring = math.floor(pop_size * args.offspring_ratio)
    return pop_size, num_selection, num_offspring


def solve(bm, args):
    pass


def main():
    args = parse_args()
    random.seed(args.seed)
    logging_setup(args)
    logger.info(vars(args))

    if args.benchmark_dir:
        assert isdir(args.benchmarks)
    else:
        assert isfile(args.benchmarks)
    bmfiles = get_bmfiles(args.benchmarks, args.benchmark_dir)
    logger.debug(f"Evaluate on {len(bmfiles)} benchmarks")

    for bm in bmfiles:
        solve(bm, args)

if __name__ == "__main__":
    main()
