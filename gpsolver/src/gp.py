import copy
import random
from typing import Callable, List, Optional

from program import Node
from grammar import Grammar
from operation import generate_program, mutate, crossover


def select(programs: List[Node], scores: List[float], num_selection: int) -> List[Node]:
    selection = sorted(range(len(programs)), key=lambda x: scores[x])[-num_selection:]
    return [programs[i] for i in selection]


def breed(g: Grammar, population: List[Node], population_size: int,
          mutation_prob: float, crossover_prob: float
         ) -> List[Node]:
    children = []
    for _ in range(population_size):
        parents = random.sample(population, 2)
        for i, p in enumerate(parents):
            if random.random() < mutation_prob:
                new = mutate(p, g)
                if new:
                    parents[i] = new
        child = crossover(parents) if random.random() < crossover_prob else None
        child = child if child else parents[0]
        children.append(child)
    return children


def check(gen_prog, example):

    example_out = example[-1]
    prog_out = gen_prog(*tuple(example[:-1]))

    return prog_out == example_out


def verify(prog: Node):

    """
    Verify the given program on the set of examples.
    Return empty list if verified
    Return a counterexample if not verified
    """
    
    # Convert the node into a python program

    # Loop over examples

    example_arr = copy.deepcopy(all_examples)

    random.shuffle(example_arr)

    for example in example_arr:
        is_correct = check(gen_prog, example)

        if not is_correct:
            return [example]

    return []


def genetic_programming(g: Grammar, population_size: int, max_generation: int, num_selection: int,
                        fitness: Callable[[Node], float],
                        select: Callable[[List[Node], List[float], int], List[Node]],
                        breed: Callable[[List[Node]], List[Node]],
                        verify: Callable[[Node], bool]
                       ) -> Optional[Node]:
    
    population = [generate_program(g) for _ in range(population_size)]
    
    for _ in range(max_generation):

        scores = [fitness(p) for p in population]

        for i in range(population_size):
            if scores[i] == 1.0 and verify(population[i]):
                return population[i]

        selection = select(population, scores, num_selection)
        population = breed(selection)

    scores = [fitness(p) for p in population]
    
    return population[scores.index(max(scores))]

if __name__ == '__main__':
    g = Grammar("../benchmarks-master/comp/2018/PBE_Strings_Track/bikes.sl")
    print(g.start)
    print(g.all_rules)
    print(g.terminal_rules)

    a = generate_program(g)
    print(a)