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
