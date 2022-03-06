import random
from copy import copy, deepcopy

from program import is_terminal


def expand(root, g, max_depth):
    node = root
    queue = [root]
    while queue:
        node = queue.pop(0)
        rules = g.all_rules[node.nt_name] if node.depth < max_depth - 1 else g.terminal_rules[node.nt_name]
        if not rules:
            continue
        node = node.apply(random.choice(rules))
        queue += node.children
    while node.parent:
        node = node.parent

    if not is_terminal(node):
        return generate_program(g, max_depth)
    return node


def get_nodes(root):
    return root.children + [n for c in root.children for n in get_nodes(c)]


def get_revertible_nodes(root):
    return [x for x in root.children if x.revert] + [n for c in root.children for n in get_nodes(c) if n.revert]


def get_same_type_nodes(root, expr_type):
    return [x for x in root.children if x.expr_type == expr_type] + [n for c in root.children for n in get_nodes(c) if n.expr_type == expr_type]


def generate_program(g, max_depth):
    assert max_depth > 0
    root = copy(g.start)
    return expand(root, g, max_depth)


def mutate(program, g, max_depth):
    new_program = deepcopy(program)
    nodes = get_revertible_nodes(new_program)
    if not nodes:
        return None
    new_node = random.choice(nodes).revert
    new_node = expand(new_node, g, max_depth)
    if new_node.parent:
        new_node.parent.children[int(new_node.id[-1])] = new_node
    return new_program


def crossover(parents):
    new_program = deepcopy(parents[0])
    new_program2 = deepcopy(parents[1])
    nodes = get_nodes(new_program)
    if not nodes:
        return None
    node = random.choice(nodes)
    # print("Base expression", print_ast(node), node.expr_type)

    matches = get_same_type_nodes(new_program2, node.expr_type)

    # i = 1
    # for match in matches:
    #     print("Match", i, print_ast(match), match.expr_type)
    #     i += 1

    if not matches:
        return None
    node2 = deepcopy(random.choice(matches))

    parent, parent2 = node.parent, node2.parent
    if parent:
        parent.children[int(node.id[-1])] = node2
    if parent2:
        parent2.children[int(node2.id[-1])] = node
    node.id, node2.id = node2.id, node.id
    node.parent, node2.parent = node2.parent, node.parent
    return new_program
