from __future__ import annotations
from copy import deepcopy
from typing import Any

from utils import logger


func_dict = {
    "str.++": lambda a, b: a + b,
    "str.replace": lambda a, b, c: str.replace(a, b, c, 1),
    "str.at": lambda a, b: a[b] if 0 <= b < len(a) else "",
    "int.to.str": lambda a: str(a) if a >= 0 else "",
    "str.substr": lambda a, b, c: a[b:(c+b)] if 0 <= b and len(a) >= (c + b) >= b else "",
    "str.len": lambda a: len(a),
    "str.to.int": lambda a: int(a) if a.isnumeric() else -1,
    "str.indexof": lambda a, b, c: str.find(a, b, c),
    "str.prefixof": lambda a, b: str.startswith(a, b),
    "str.suffixof": lambda a, b: str.endswith(a, b),
    "str.contains": lambda a, b: str.find(a, b) != -1,
    "-": lambda a, b: a - b,
    "+": lambda a, b: a + b,
    "ite": lambda a, b, c: b if a else c,
}


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

    def is_ground(self):
        if isinstance(self, NTNode):
            return False
        if isinstance(self, FuncNode):
            return all([c.is_ground for c in self.children])
        return True

    def to_dict(self) -> dict[str, Any]:
        return {"node_type": self.node_type, "expr_type": self.expr_type, "depth": self.depth, "id": self.id}

    def apply(self, rule: Production) -> Node:
        assert self.__eq__(rule.lhs)
        new = deepcopy(rule.rhs)
        new.id = self.id
        new.parent = self.parent
        new.revert = self
        new.depth = self.depth
        for i, c in enumerate(new.children):
            c.parent = new
            c.id = new.id + str(i)
            c.depth = self.depth + 1
        if self.parent:
            self.parent.children[int(self.id[-1])] = new
        return new

    def get_name(self) -> str:
        return ""

    def size(self):
        return 1 + sum([c.size() for c in self.children])


class VarNode(Node):
    def __init__(self, var_name: str, expr_type: str) -> None:
        super().__init__(expr_type)
        self.node_type = 1
        self.name = var_name

    def __eq__(self, o: object) -> bool:
        return super().__eq__(o) and isinstance(o, VarNode) and self.name == o.name

    def __hash__(self) -> int:
        return hash(str(self.to_dict()))

    def to_dict(self) -> dict[str, Any]:
        d = super().to_dict()
        d["name"] = self.name
        return d

    def get_name(self) -> str:
        return self.name

class ConstNode(Node):
    def __init__(self, expr_type: str, value: Any) -> None:
        super().__init__(expr_type)
        self.node_type = 2
        self.value = value

    def __eq__(self, o: object) -> bool:
        return super().__eq__(o) and isinstance(o, ConstNode) and self.value == o.value

    def __hash__(self) -> int:
        return hash(str(self.to_dict()))

    def to_dict(self) -> dict[str, str]:
        d = super().to_dict()
        d["value"] = self.value
        return d

    def get_name(self) -> str:
        return str(self.value)

class FuncNode(Node):
    def __init__(self, func_name: str, expr_type: str) -> None:
        super().__init__(expr_type)
        self.node_type = 3
        self.func_name = func_name

    def __eq__(self, o: object) -> bool:
        return super().__eq__(o) and isinstance(o, FuncNode) and self.func_name == o.func_name

    def __hash__(self) -> int:
        return hash(str(self.to_dict()))

    def to_dict(self) -> dict[str, str]:
        d = super().to_dict()
        d["func_name"] = self.func_name
        return d

    def get_name(self) -> str:
        return self.func_name

class NTNode(Node): # non-terminal
    def __init__(self, nt_name: str, expr_type: str) -> None:
        super().__init__(expr_type)
        self.node_type = 4
        self.nt_name = nt_name

    def __eq__(self, o: object) -> bool:
        return super().__eq__(o) and isinstance(o, NTNode) and self.nt_name == o.nt_name

    def __hash__(self) -> int:
        return hash(str(self.to_dict()))

    def to_dict(self) -> dict[str, str]:
        d = super().to_dict()
        d["nt_name"] = self.nt_name
        return d

    def get_name(self) -> str:
        return self.nt_name

def is_terminal(n: Node) -> bool:
    if isinstance(n, NTNode):
        return False
    if isinstance(n, FuncNode):
        return all([is_terminal(c) for c in n.children])
    return True

def print_ast(n: Node) -> str:
    if isinstance(n, ConstNode):
        if n.value == "":
            return "\'\'"
        if n.value == " ":
            return "\' \'"
        return str(n.value)
    if isinstance(n, FuncNode):
        s = n.get_name() + "("
        s += ", ".join(print_ast(child) for child in n.children)
        s += ")"
        return s
    return n.get_name()


def execute(prog, var_dict):
    assert prog.is_ground()
    child_progs = [execute(child, var_dict) for child in prog.children]
    if isinstance(prog, ConstNode):
        return prog.value
    elif isinstance(prog, VarNode):
        return var_dict[prog.name]
    elif isinstance(prog, FuncNode):
        fn = func_dict.get(prog.func_name)
        if fn is None:
            logger.info(f"Unknown function: {prog.to_dict()}")
            return None
        return fn(*child_progs)
    return None


def execute_batch(prog, ins):
    outs = []
    for i in ins:
        try:
            o = execute(prog, i)
        except Exception:
            logger.debug("Program is faulty")
            o = "<error>"
        outs.append(o)
    return outs
