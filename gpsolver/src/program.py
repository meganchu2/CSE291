from __future__ import annotations
from copy import deepcopy
from typing import Any


class Production:
    def __init__(self, lhs: NTNode, rhs: Node) -> None:
        self.lhs = lhs
        self.rhs = rhs


class Node:
    def __init__(self, expr_type: str) -> None:
        self.parent = None
        self.children = []
        self.node_type = 0
        self.expr_type = expr_type
        self.depth = 0
        self.id = "0"
        self.revert = None

    def __eq__(self, o: object) -> bool:
        return isinstance(o, Node) and self.expr_type == o.expr_type

    def to_dict(self) -> dict[str, Any]:
        return {"node_type": self.node_type, "expr_type": self.expr_type, "depth": self.depth, "id": self.id}

    def apply(self, rule: grammar.Production) -> Node:
        assert self.__eq__(rule.lhs)
        new = deepcopy(rule.rhs)
        new.id = self.id
        new.parent = self.parent
        new.revert = self.to_dict()
        new.depth = self.depth
        for i, c in enumerate(new.children):
            c.parent = new
            c.id = new.id + str(i)
            c.depth = self.depth + 1
        if self.parent:
            self.parent.children[int(self.id[-1])] = new
        return new

class VarNode(Node):
    def __init__(self, var_name: str, expr_type: str) -> None:
        super().__init__(expr_type)
        self.node_type = 1
        self.name = var_name

    def __eq__(self, o: object) -> bool:
        return super().__eq__(o) and isinstance(o, VarNode) and self.name == o.name

    def to_dict(self) -> dict[str, Any]:
        d = super().to_dict()
        d["name"] = self.name
        return d

class ConstNode(Node):
    def __init__(self, expr_type: str, value: Any) -> None:
        super().__init__(expr_type)
        self.node_type = 2
        self.value = value

    def __eq__(self, o: object) -> bool:
        return super().__eq__(o) and isinstance(o, ConstNode) and self.value == o.value

    def to_dict(self) -> dict[str, str]:
        d = super().to_dict()
        d["value"] = self.value
        return d

class FuncNode(Node):
    def __init__(self, func_name: str, expr_type: str) -> None:
        super().__init__(expr_type)
        self.node_type = 3
        self.func_name = func_name

    def __eq__(self, o: object) -> bool:
        return super().__eq__(o) and isinstance(o, FuncNode) and self.func_name == o.func_name

    def to_dict(self) -> dict[str, str]:
        d = super().to_dict()
        d["func_name"] = self.func_name
        return d

class NTNode(Node): # non-terminal
    def __init__(self, nt_name: str, expr_type: str) -> None:
        super().__init__(expr_type)
        self.node_type = 4
        self.nt_name = nt_name

    def __eq__(self, o: object) -> bool:
        return super().__eq__(o) and isinstance(o, NTNode) and self.nt_name == o.nt_name

    def to_dict(self) -> dict[str, str]:
        d = super().to_dict()
        d["nt_name"] = self.nt_name
        return d

def is_terminal(n: Node) -> bool:
    if isinstance(n, NTNode):
        return False
    if isinstance(n, FuncNode):
        return all([is_terminal(c) for c in n.children])
    return True
