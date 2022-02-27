from typing import List
from collections import defaultdict

from program import *

class Production:
    def __init__(self, lhs: NTNode, rhs: Node) -> None:
        self.lhs = lhs
        self.rhs = rhs

class Grammar:
    def __init__(self, non_terminals: List[NTNode], rules: List[Production], start: NTNode) -> None:
        self.nts = non_terminals
        self.start = start
        self.all_rules = defaultdict(list)
        self.terminal_rules = defaultdict(list)

        for r in rules:
            self.all_rules[r.lhs.nt_name].append(r.rhs)
            if not is_terminal(r.rhs):
                self.terminal_rules[r.lhs.nt_name].append(r.rhs)
