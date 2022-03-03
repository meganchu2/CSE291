import copy
import random
import numpy as np
from typing import Callable, List, Optional

from program import Node, VarNode, FuncNode, ConstNode
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


def get_py_function(prog_ast, var_dict):

    child_progs = []

    for child in prog_ast.children:
        child_progs.append(get_py_function(child, var_dict))

    if isinstance(prog_ast, ConstNode):
        return prog_ast.value

    elif isinstance(prog_ast, VarNode):
        return var_dict[prog_ast.name]

    elif isinstance(prog_ast, FuncNode):

        if prog_ast.func_name == 'str.++':
            return (lambda a, b : a + b)(child_progs[0], child_progs[1])

        elif prog_ast.func_name == 'str.replace':
            return (lambda a,b,c: str.replace(a, b, c, 1))(child_progs[0], child_progs[1], child_progs[2])

        elif prog_ast.func_name == 'str.at':
            return (lambda a,b: a[b] if 0 <= b < len(a) else '')(child_progs[0], child_progs[1])

        elif prog_ast.func_name == 'int.to.str':
            return (lambda a : str(a) if a >= 0 else '')(child_progs[0])

        elif prog_ast.func_name == 'str.substr':
            return (lambda a,b,c: a[b:(c+b)] if 0 <= b and len(a) >= (c+b) >= b else '')(child_progs[0], child_progs[1], child_progs[2])

        elif prog_ast.func_name == 'str.len':
            return (lambda a: len(a))(child_progs[0])

        elif prog_ast.func_name == 'str.to.int':

            def eval_c(a):
                try:
                    if all(map(lambda x: '0' <= x <= '9', a)):
                        return int(a)
                    else:
                        return -1
                except ValueError:
                    return -1

            return (lambda a: eval_c(a))(child_progs[0])

        elif prog_ast.func_name == 'str.indexof':
            return (lambda a, b, c: str.find(a, b, c))(child_progs[0], child_progs[1], child_progs[2])

        elif prog_ast.func_name == 'str.prefixof':
            return (lambda a, b: str.startswith(a, b))(child_progs[0], child_progs[1])

        elif prog_ast.func_name == 'str.suffixof':
            return (lambda a, b: str.endswith(a, b))(child_progs[0], child_progs[1])

        elif prog_ast.func_name == 'str.contains':
            return (lambda a,b: str.find(a,b) != -1)(child_progs[0], child_progs[1])

        elif prog_ast.func_name == '-':
            return (lambda a,b: a - b)(child_progs[0], child_progs[1])

        elif prog_ast.func_name == '+':
            return (lambda a,b: a + b)(child_progs[0], child_progs[1])

        else:
            print("Unknown function", prog_ast.to_dict())
            return None

        return [prog_ast.func_name, child_progs]


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
            prog_out = get_py_function(prog, var_dict)
            is_correct = prog_out == ex_out
        except Exception as e:
            is_correct = False
            print("Program is faulty")

        if not is_correct:
            return [examples_dict[example_idx]]

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

    benchmark_file = "../benchmarks-master/comp/2018/PBE_Strings_Track/name-combine-2.sl"
    # benchmark_file = "../benchmarks-master/comp/2018/PBE_Strings_Track/firstname_small.sl"
    # benchmark_file = "../benchmarks-master/comp/2018/PBE_Strings_Track/bikes.sl"

    # Load the grammar from the file

    g = Grammar(benchmark_file)

    # Load the examples

    f = open(benchmark_file, 'r')

    examples_dict = {}
    ex_idx_arr = []
    example_index = 0

    var_names = []

    for line in f:

        if line.startswith("(declare-var"):
            var_names.append(line.split(" ")[1])

        if line.startswith("(constraint"):
            ex_str = line.strip().split("(")[3]
            ex_components = ex_str.split(")")[:-2]

            ex_in_list = ex_components[0].split("\"")

            num_params = int((len(ex_in_list) - 1) / 2)
            ex_params = [ex_in_list[int(1 + 2 * i)] for i in range(num_params)]

            ex_out = ex_components[-1][2:-1]

            examples_dict[example_index] = {
                'params': ex_params,
                'output': ex_out
            }

            ex_idx_arr.append(example_index)

            example_index += 1

    f.close()

    # print(examples_dict)
    # print(var_names)

    eg_info_dict = {
        'var_names': var_names,
        'examples_dict': examples_dict,
        'examples_idx_arr': ex_idx_arr,
    }


    # Generate a random program for testing

    a = generate_program(g)

    # print(print_prog(a))

    # Call verify on the program

    print(verify(a, eg_info_dict))



