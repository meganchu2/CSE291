import copy
import random
import numpy as np
from typing import Callable, Dict, List, Optional

from program import Node, VarNode, FuncNode, ConstNode
from operation import generate_program, mutate, crossover
from utils import logger
from difflib import SequenceMatcher

from datetime import datetime


func_dict = {
    "str.++": lambda a, b: a + b,
    "str.replace": lambda a, b, c: str.replace(a, b, c, 1),
    "str.at": lambda a, b: a[b] if 0 <= b < len(a) else "",
    "int.to.str": lambda a: str(a) if a >= 0 else "",
    "str.substr": lambda a, b, c: a[b:(c+b)] if 0 <= b and len(a) >= (c + b) >= b else "",
    "str.len": lambda a: len(a),
    "str.to.int": lambda a: int(a) if a.isnumeric() else -1,
    "str.indexof": lambda a, b, c: str.find(a, b, c),
    "str.prefixof": lambda a, b: str.startswith(a, b),
    "str.suffixof": lambda a, b: str.endswith(a, b),
    "str.contains": lambda a, b: str.find(a, b) != -1,
    "-": lambda a, b: a - b,
    "+": lambda a, b: a + b,
    "ite": lambda a, b, c: b if a else c,
}


def best_indices(scores):
    mx = max(scores)
    return [i for i, x in enumerate(scores) if x == mx]


def select(programs, num_selection, constraints, algorithm, metric):
    assert len(programs) >= num_selection
    if algorithm == "best":
        scores = [fitness_all(p, constraints, metric) for p in programs]
        selection = sorted(range(len(programs)), key=lambda x: scores[x])[-num_selection:]
        return [programs[i] for i in selection]

    # lexicase selection
    variables, exs = constraints
    ex_indices = list(range(len(exs)))
    survivors = []
    while len(survivors) < num_selection:
        random.shuffle(ex_indices)
        current = list(filter(lambda i: i not in survivors, range(len(programs))))
        selected = False
        for ex_idx in ex_indices:
            scores = [fitness(programs[i], variables, exs[ex_idx], metric) for i in current]
            new = [programs[current[i]] for i in best_indices(scores)]
            if len(new) == 1:
                survivors.append(new[0])
                selected = True
                break
            current = new
        if not selected:
            survivors.append(random.choice(current))
    return [programs[i] for i in survivors]


def breed(grammar, population, pop_size, mutation_prob, crossover_prob):
    children = []
    for _ in range(pop_size):
        parents = random.sample(population, 2)
        for i, p in enumerate(parents):
            if random.random() < mutation_prob:
                new = mutate(p, grammar)
                if new:
                    parents[i] = new
        child = crossover(parents) if random.random() < crossover_prob else None
        child = child if child else parents[0]
        children.append(child)
    return children


def prog_size(prog_ast):

    child_size_sum = 0

    for child in prog_ast.children:
        child_size_sum += prog_size(child)

    return child_size_sum + 1


def print_prog(prog_ast):

    child_progs = []

    for child in prog_ast.children:
        child_progs.append(print_prog(child))

    if isinstance(prog_ast, ConstNode):
        return prog_ast.value
    elif isinstance(prog_ast, VarNode):
        return prog_ast.name
    elif isinstance(prog_ast, FuncNode):
        return [prog_ast.func_name, child_progs]


def execute(prog, var_dict):
    child_progs = [execute(child, var_dict) for child in prog.children]
    if isinstance(prog, ConstNode):
        return prog.value
    elif isinstance(prog, VarNode):
        return var_dict[prog.name]
    elif isinstance(prog, FuncNode):
        fn = func_dict.get(prog.func_name)
        if fn is None:
            logger.info(f"Unknown function: {prog.to_dict()}")
            return None
        return fn(*child_progs)
    return None


def fitness(prog, eg_info_dict, example_idx, jaccard):
    # change boolean if we want to include jaccard similarity in fitness
    jaccard = True

    correct = 0
    examples_dict = eg_info_dict['examples_dict']
    var_names = eg_info_dict['var_names']
    ex_params = examples_dict[example_idx]['params']
    ex_out = examples_dict[example_idx]['output']
    var_dict = {}
    var_idx = 0
    for var_name in var_names:
        var_dict[var_name] = ex_params[var_idx]
        var_idx += 1
    is_correct = True
    try:
        prog_out = execute(prog, var_dict)
        s = SequenceMatcher(None, prog_out, ex_out)
        m = s.find_longest_match(0,len(prog_out),0,len(ex_out))
        if jaccard:
            correct += m.size/max(len(prog_out),len(ex_out))
            correct += m.size/(len(prog_out) + len(ex_out) + m.size)
        else:
            correct += m.size/max(len(prog_out),len(ex_out))
    except Exception as e:
        return 0.0

    return correct

def fitness_all(prog, eg_info_dict):
    # change boolean if we want to include jaccard similarity in fitness
    jaccard = True

    # Loop over examples
    examples_idx_arr = copy.deepcopy(eg_info_dict['examples_idx_arr'])
    correct = 0
    for example_idx in examples_idx_arr:
        correct += fitness(prog, eg_info_dict, example_idx, jaccard)
    return correct/len(examples_idx_arr)


def verify_single(p: Node, ex: Dict, var_names: List[str]) -> bool:
    params = ex["params"]
    output = ex["output"]
    var_dict = {k: v for (k, v) in zip(var_names, params)}
    try:
        prog_out = execute(p, var_dict)
        return prog_out == output
    except Exception as _:
        return False


def verify(prog, eg_info_dict):

    """
    Verify the given program on the set of examples.
    Return empty list if verified
    Return a counterexample if not verified
    """

    # Loop over examples

    examples_idx_arr = copy.deepcopy(eg_info_dict['examples_idx_arr'])
    random.shuffle(examples_idx_arr)

    examples_dict = eg_info_dict['examples_dict']
    var_names = eg_info_dict['var_names']

    for example_idx in examples_idx_arr:
        ex_params = examples_dict[example_idx]['params']
        ex_out = examples_dict[example_idx]['output']

        var_dict = {}
        var_idx = 0

        for var_name in var_names:
            var_dict[var_name] = ex_params[var_idx]
            var_idx += 1

        is_correct = True

        try:
            prog_out = execute(prog, var_dict)
            is_correct = prog_out == ex_out
        except Exception as e:
            is_correct = False
            print("Program is faulty")

        if not is_correct:
            return [examples_dict[example_idx]]

    return []


def get_out_str(prog, eg_info_dict):

    examples_idx_arr = copy.deepcopy(eg_info_dict['examples_idx_arr'])

    examples_dict = eg_info_dict['examples_dict']
    var_names = eg_info_dict['var_names']

    out_str = ""

    for example_idx in examples_idx_arr:
        ex_params = examples_dict[example_idx]['params']
        ex_out = examples_dict[example_idx]['output']

        var_dict = {}
        var_idx = 0

        for var_name in var_names:
            var_dict[var_name] = ex_params[var_idx]
            var_idx += 1

        try:
            prog_out = execute(prog, var_dict)
            out_str = out_str + prog_out
        except Exception as e:
            print("Program is faulty")
            out_str = out_str + "<error>"

    return out_str


def initialize_population(grammar, constraints, population_size, max_depth):
    num_gen = 0
    gen_prog_dict = {}
    while num_gen < population_size:
        new_prog = generate_program(grammar, max_depth)
        prog_key = get_out_str(new_prog, constraints)
        old_prog = gen_prog_dict.get(prog_key)

        if old_prog is None:
            gen_prog_dict[prog_key] = new_prog
            num_gen += 1
        elif prog_size(old_prog) > prog_size(new_prog):
            gen_prog_dict[prog_key] = new_prog
    return list(gen_prog_dict.values())


def genetic_programming(grammar, args, other_hps, constraints):
    pop_size, num_selection, num_offspring = other_hps

    t_start = datetime.now()
    population = initialize_population(grammar, constraints, pop_size, args.init_max_depth)
    t_init = datetime.now()
    logger.info(f"Initialization took {t_init - t_start}")

    result = []
    for gen in range(args.num_generation):
        logger.debug(f"Generation {gen + 1}")
        for p in population:
            if len(verify(p, constraints)) == 0:
                result.append(p)
        if len(result) > 0:
            break

        selection = select(population, num_selection, constraints, args.select, args.fitness)
        children = breed(grammar, selection, num_offspring, args.mutation_prob, args.crossover_prob)
        population = children + selection
        scores = [fitness_all(p, constraints) for p in population]
        population = sorted(range(len(population)), key=lambda x: scores[x], reverse=True)[:pop_size]

    t_end = datetime.now()
    logger.info(f"GP took {t_end - t_init}")
    logger.info(f"Total time {t_end - t_start}")
    return result


def printDistances(distances, token1Length, token2Length):

    for t1 in range(token1Length + 1):
        for t2 in range(token2Length + 1):
            print(int(distances[t1][t2]), end=" ")
        print()


def levenshteinDistanceDP(token1, token2):

    distances = np.zeros((len(token1) + 1, len(token2) + 1))

    for t1 in range(len(token1) + 1):
        distances[t1][0] = t1

    for t2 in range(len(token2) + 1):
        distances[0][t2] = t2

    a = 0
    b = 0
    c = 0

    for t1 in range(1, len(token1) + 1):
        for t2 in range(1, len(token2) + 1):
            if (token1[t1-1] == token2[t2-1]):
                distances[t1][t2] = distances[t1 - 1][t2 - 1]
            else:
                a = distances[t1][t2 - 1]
                b = distances[t1 - 1][t2]
                c = distances[t1 - 1][t2 - 1]

                if (a <= b and a <= c):
                    distances[t1][t2] = a + 1
                elif (b <= a and b <= c):
                    distances[t1][t2] = b + 1
                else:
                    distances[t1][t2] = c + 1

    return distances[len(token1)][len(token2)]
