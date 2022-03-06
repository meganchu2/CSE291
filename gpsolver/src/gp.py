import copy
import random
import numpy as np
from typing import Callable, Dict, List, Optional

from program import Node, execute, execute_batch
from operation import generate_program, mutate, crossover
from utils import logger
from difflib import SequenceMatcher

from datetime import datetime


def best_indices(scores):
    mx = max(scores)
    return [i for i, x in enumerate(scores) if x == mx]


def select(programs, num_selection, examples, algorithm, metric):
    assert len(programs) >= num_selection
    ins = examples["in"]
    outs = examples["out"]
    if algorithm == "best":
        scores = [fitness_all(outs, execute_batch(p, ins), metric) for p in programs]
        selection = sorted(range(len(programs)), key=lambda x: scores[x])[-num_selection:]
        return [programs[i] for i in selection]

    # lexicase selection
    indices = list(range(len(examples)))
    survivors = []
    while len(survivors) < num_selection:
        random.shuffle(indices)
        current = [i for i in range(len(programs)) if i not in survivors]
        selected = False
        for i in indices:
            scores = [fitness(outs[i], execute(programs[j], ins[i]), metric) for j in current]
            new = [current[j] for j in best_indices(scores)]
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


def fitness(ans, out, metric):
    if metric == "binary":
        return 1.0 if ans == out else 0.0
    if metric == "match":
        try:
            s = SequenceMatcher(None, out, ans)
            m = s.find_longest_match(0, len(out), 0, len(ans))
            return m.size / max(len(ans), len(out))
        except Exception:
            return 0.0
    if metric == "jaccard":
        intersection = len(set(ans).intersection(out))
        union = len(set(ans)) + len(set(out)) - intersection
        return float(intersection) / union
    if metric == "levenshtein":
        length = max(len(ans), len(out))
        return (length - levenshteinDistanceDP(ans, out)) / float(length)
    return 0.0


def fitness_all(answers, outs, metric):
    return sum([fitness(answers[i], outs[i], metric) for i in range(len(outs))]) / len(outs)


def initialize_population(grammar, examples, population_size, max_depth):
    num_gen = 0
    gen_prog_dict = {}
    while num_gen < population_size:
        new_prog = generate_program(grammar, max_depth)
        prog_key = str(execute_batch(new_prog, examples["in"]))
        old_prog = gen_prog_dict.get(prog_key)

        if old_prog is None:
            gen_prog_dict[prog_key] = new_prog
            num_gen += 1
        elif old_prog.size() > new_prog.size():
            gen_prog_dict[prog_key] = new_prog
    return list(gen_prog_dict.values())


def verify(prog, examples):
    return execute_batch(prog, examples["in"]) == examples["out"]


def genetic_programming(grammar, args, other_hps, examples):
    pop_size, num_selection, num_offspring = other_hps

    t_start = datetime.now()
    population = initialize_population(grammar, examples, pop_size, args.init_max_depth)
    t_init = datetime.now()
    logger.info(f"Initialization took {t_init - t_start}")

    result = []
    for gen in range(args.num_generation):
        logger.debug(f"Generation {gen + 1}")
        result = [p for p in population if verify(p, examples)]
        if len(result) > 0:
            break

        selection = select(population, num_selection, examples, args.select, args.fitness)
        children = breed(grammar, selection, num_offspring, args.mutation_prob, args.crossover_prob)
        population = children + selection
        scores = [fitness_all(examples["out"], execute_batch(p, examples["in"]), args.fitness) for p in population]
        indices = sorted(range(len(population)), key=lambda x: scores[x], reverse=True)[:pop_size]
        population = [population[i] for i in indices]

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
