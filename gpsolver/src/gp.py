import copy
import random
import numpy as np
from typing import Callable, Dict, List, Optional

from program import Node, execute_batch
from operation import generate_program, mutate, crossover
from utils import logger
from difflib import SequenceMatcher

from datetime import datetime


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
        prog_out = prog.execute(var_dict)
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
        prog_out = p.execute(var_dict)
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
            prog_out = prog.execute(var_dict)
            is_correct = prog_out == ex_out
        except Exception as e:
            is_correct = False
            print("Program is faulty")

        if not is_correct:
            return [examples_dict[example_idx]]

    return []


def initialize_population(grammar, constraints, population_size, max_depth):
    num_gen = 0
    gen_prog_dict = {}
    while num_gen < population_size:
        new_prog = generate_program(grammar, max_depth)
        prog_key = str(execute_batch(new_prog, constraints))
        old_prog = gen_prog_dict.get(prog_key)

        if old_prog is None:
            gen_prog_dict[prog_key] = new_prog
            num_gen += 1
        elif old_prog.size() > new_prog.size():
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
