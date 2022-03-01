import random
from copy import copy, deepcopy
from typing import List, Optional

from grammar import Grammar
from program import NTNode, Node, is_terminal

MAX_DEPTH = 32

def expand(root: NTNode, g: Grammar, max_depth: int = MAX_DEPTH) -> Node:
    node = root
    queue = [root]
    while queue:
        node = queue.pop(0)
        # print(node.to_dict())
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

# note that we do not want to pick the root node
def get_nodes(root: Node) -> List[Node]:
    return [] # TODO

def get_revertible_nodes(root: Node) -> List[Node]:
    return [] # TODO

def get_same_type_nodes(root: Node, expr_type: int) -> List[Node]:
    return [] # TODO

def revert_node(node: Node) -> NTNode:
    return NTNode("", 0) # TODO

def generate_program(g: Grammar, max_depth: int = MAX_DEPTH) -> Node:
    assert max_depth > 0
    root = copy(g.start)
    return expand(root, g, max_depth)

def mutate(program: Node, g: Grammar) -> Optional[Node]:
    new_program = deepcopy(program)
    nodes = get_revertible_nodes(new_program)
    if not nodes:
        return None
    new_node = revert_node(random.choice(nodes))
    new_node = expand(new_node, g)
    if new_node.parent:
        new_node.parent.children[int(new_node.id[-1])] = new_node
    return new_program

def crossover(parents: List[Node]) -> Optional[Node]:
    new_program = deepcopy(parents[0])
    new_program2 = deepcopy(parents[1])
    nodes = get_nodes(new_program)
    if not nodes:
        return None
    node = random.choice(nodes)
    matches = get_same_type_nodes(new_program2, node.expr_type)
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
