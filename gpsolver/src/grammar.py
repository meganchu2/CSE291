from program import *
from typing import List, Any
from collections import defaultdict


class Production:
    def __init__(self, lhs: NTNode, rhs: Node) -> None:
        self.lhs = lhs
        self.rhs = rhs

class Grammar:
    def __init__(self, li: List[Any]) -> None:
        start = li[0]
        self.start = NTNode(nt_name=start[2][0], expr_type=start[1])
        self.all_rules = defaultdict(list)
        self.terminal_rules = defaultdict(list)
        type_dict = dict({i[0]: i[1] for i in li})
        rules = li[1:]
        for r in rules:
            lhs = NTNode(nt_name=r[0], expr_type=r[1])
            for i in r[2]:
                if isinstance(i, str):
                    rhs = VarNode(var_name=i, expr_type=lhs.nt_name)
                    self.terminal_rules[lhs.nt_name].append(Production(lhs, rhs))
                elif isinstance(i, tuple):
                    rhs = ConstNode(expr_type=i[0], value=i[1])
                    self.terminal_rules[lhs.nt_name].append(Production(lhs, rhs))
                else: # list
                    rhs = FuncNode(func_name=i[0], expr_type=lhs.nt_name)
                    for j in i[1:]:
                        rhs.children.append(NTNode(nt_name=j, expr_type=type_dict[j]))
                self.all_rules[lhs.nt_name].append(Production(lhs, rhs))
