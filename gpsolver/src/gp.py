import copy
import random
import numpy as np
from typing import Callable, Dict, List, Optional

from program import Node, VarNode, FuncNode, ConstNode, print_ast, count_nodes
from grammar import Grammar
from operation import generate_program, mutate, crossover
from difflib import SequenceMatcher

from datetime import datetime


def select(programs: List[Node], scores: List[float], num_selection: int) -> List[Node]:
    selection = sorted(range(len(programs)), key=lambda x: scores[x])[-num_selection:]
    return [programs[i] for i in selection]

def verify_single(p: Node, ex: Dict, var_names: List[str]) -> bool:
    params = ex["params"]
    output = ex["output"]
    var_dict = {k: v for (k, v) in zip(var_names, params)}
    try:
        prog_out = get_py_function(p, var_dict)
        return prog_out == output
    except Exception as _:
        return False

def lexicase_select(programs: List[Node], num_selection: int, examples_info_dict: Dict) -> List[Node]:
    assert len(programs) >= num_selection
    var_names = examples_info_dict["var_names"]
    exs = list(examples_info_dict["examples_dict"].values())
    survivors = []
    while len(survivors) < num_selection:
        random.shuffle(exs)
        current = [i for i in range(len(programs)) if i not in survivors]
        selected = False
        for ex in exs:
            new = list(filter(lambda p: verify_single(p, ex, var_names), current))
            if not new:
                survivors.append(random.choice(current))
                selected = True
                break
            if len(new) == 1:
                survivors.append(new[0])
                selected = True
                break
            current = new
        if not selected:
            survivors.append(random.choice(current))
    return [programs[i] for i in survivors]

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
            
        elif prog_ast.func_name == 'ite':
            return (lambda a, b, c: (b if a else c))(child_progs[0], child_progs[1], child_progs[2])
            
        else:
            print("Unknown function", prog_ast.to_dict())
            return None

        return [prog_ast.func_name, child_progs]

def fitness(prog, eg_info_dict):
    # change boolean if we want to include jaccard similarity in fitness
    jaccard = true

    # Loop over examples    
    examples_idx_arr = copy.deepcopy(eg_info_dict['examples_idx_arr'])
    examples_dict = eg_info_dict['examples_dict']
    var_names = eg_info_dict['var_names']
    correct = 0
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
            s = SequenceMatcher(None, prog_out, ex_out)
            m = s.find_longest_match(0,len(prog_out),0,len(ex_out))
            if jaccard:
                correct += m.size/max(len(prog_out),len(ex_out))
                correct += m.size/(len(prog_out) + len(ex_out) + m.size)
            else:
                correct += m.size/max(len(prog_out),len(ex_out))
        except Exception as e:
            return 0.0
    return correct/len(examples_idx_arr)

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

# return smallest program if there is one
def smallest_prog(programs: List[Node]):
    minSize = None
    minInd = -1
    for i in range(len(programs)):
        size = count_nodes(programs[i])
        if not minSize or size < minSize:
            minSize = size
            minInd = i
        #print(print_ast(programs[i]))
        #print(size)
    if minInd == -1:
        return None
    return programs[minInd]



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
            prog_out = get_py_function(prog, var_dict)
            out_str = out_str + prog_out
        except Exception as e:
            print("Program is faulty")
            out_str = out_str + "<error>"

    return out_str


def genetic_programming(g: Grammar, population_size: int, max_generation: int, num_selection: int,
                        #fitness: Callable[[Node], float],
                        #select: Callable[[List[Node], List[float], int], List[Node]],
                        breed: Callable[[Grammar, List[Node], int, float, float], List[Node]],
                        verify: Callable[[Node, Dict], List],
                        examples_info_dict: Dict
                       ) -> List[Node]:

    t_start = datetime.now()

    num_gen = 0
    gen_prog_dict = {}

    while num_gen < population_size:

        new_prog = generate_program(g)
        prog_key = get_out_str(new_prog, examples_info_dict)
        old_prog = gen_prog_dict.get(prog_key)

        if old_prog is None:

            gen_prog_dict[prog_key] = new_prog
            num_gen += 1

            # print("Count", num_gen, "Added program", print_ast(new_prog))

        elif prog_size(old_prog) > prog_size(new_prog):

            gen_prog_dict[prog_key] = new_prog

            # print("Count", num_gen, "Replaced program", print_ast(old_prog), "\n", print_ast(new_prog))

    # population = [generate_program(g) for _ in range(population_size)]
    population = list(gen_prog_dict.values())

    result = []

    t_1 = datetime.now()

    print("Initial population generation took", t_1 - t_start)

    # Generate the initial population while checking for equivalence

    for idx in range(max_generation):

        print("-------------------------------Starting Generation", idx + 1, "-------------------------------")

        #scores = [fitness(p, examples_info_dict) for p in population]


        for i in range(population_size):
            # print("Verifying", print_ast(population[i]))
            if len(verify(population[i], examples_info_dict)) == 0:
                result.append(population[i])

        if len(result) > 0: # at least one solution found in this generation, stop
            break

        #selection = select(population, scores, num_selection)
        selection = lexicase_select(population, num_selection, examples_info_dict)
        population = breed(g, selection, population_size, 0.0, 1.0)

    return result


def load_examples(benchmark_file):
    f = open(benchmark_file, 'r')

    examples_dict = {}
    ex_idx_arr = []
    example_index = 0

    var_names = []
    lines = f.readlines()
    i = 0
    while i < len(lines):
        line = lines[i]

        if line.startswith("(declare-var"):
            var_names.append(line.split(" ")[1])

        if line.startswith("(constraint"):
            ex_str = line.strip().split("(")[3]
            ex_components = ex_str.split(")")[:-2]

            while len(ex_components) == 0: # constraint is on multiple lines in spec
                i += 1
                line = lines[i]
                ex_str += line.strip()
                ex_components = ex_str.split(")")[:-2]

            ex_in_list = ex_components[0].split("\"")

            num_params = int((len(ex_in_list) - 1) / 2)
            ex_params = [ex_in_list[int(1 + 2 * i)] for i in range(num_params)]

            ex_out = ex_components[-1][0:-1].lstrip().lstrip("\"")

            examples_dict[example_index] = {
                'params': ex_params,
                'output': ex_out
            }

            ex_idx_arr.append(example_index)

            example_index += 1
        i += 1

    f.close()

    # print(examples_dict)
    # print(var_names)

    eg_info_dict = {
        'var_names': var_names,
        'examples_dict': examples_dict,
        'examples_idx_arr': ex_idx_arr,
    }

    return eg_info_dict

if __name__ == '__main__':


    #benchmark_file = "../benchmarks-master/comp/2018/PBE_Strings_Track/firstname_small.sl"    # easy
    # benchmark_file = "../benchmarks-master/comp/2018/PBE_Strings_Track/name-combine_short.sl" # easy
    benchmark_file = "../benchmarks-master/comp/2018/PBE_Strings_Track/univ_2.sl"
    # benchmark_file = "../benchmarks-master/comp/2018/PBE_Strings_Track/name-combine-2.sl"
    # benchmark_file = "../benchmarks-master/comp/2018/PBE_Strings_Track/bikes.sl"
    # benchmark_file = "../benchmarks-master/comp/2018/PBE_Strings_Track/phone-5.sl"

    # Load the grammar from the file

    g = Grammar(benchmark_file)

    # Load the examples
    eg_info_dict = load_examples(benchmark_file)

    # get hyperparameters
    (population_size, max_generation, num_selection) = g.get_hyperparameters(len(eg_info_dict))
    print((population_size, max_generation, num_selection))

    print("starting genetic_programming")

    result = genetic_programming(g, population_size, max_generation, num_selection, breed, verify, eg_info_dict)

    solution = smallest_prog(result)

    if result is not None:
        print("Final solution", print_ast(solution))
        print("Solution size", prog_size(solution))
    else:
        print("Unable to find a solution")

