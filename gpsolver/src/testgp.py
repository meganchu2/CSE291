from typing import List
from grammar import Grammar
from program import *
from gp import select, breed, genetic_programming



# x = VarNode("x", 0) #returns an int
# y = VarNode("y", 0) #returns an int
# const0 = ConstNode(0, 0) # returns an int
# const1 = ConstNode(1, 0) # returns an int
x = NTNode("x", 0) #returns an int
y = NTNode("y", 0) #returns an int
const0 = NTNode("0", 0) # returns an int
const1 = NTNode("1", 0) # returns an int
ite = NTNode("ite",0)# returns an int 
leq = NTNode("leq",1)# returns a bool

ntnodes = [ite, leq]

# need NTNode lhs and Node rhs
prods = [(ite, leq),(ite, x),(ite, y),(ite, const0),(ite,const1),(leq, x),(leq,y),(leq,const0),(leq,y)]

rules = []
for pair in prods:
    print(pair[1])
    #print(pair[1])
    rules.append(Production(pair[0],pair[1]))


# need NTNode
start = ite

# need list of NT nodes, list of rules, start node
g = Grammar(ntnodes, rules, start)

# (g: Grammar, population_size: int, max_generation: int, num_selection: int,
                        ## fitness: Callable[[Node], float],
                        # select: Callable[[List[Node], List[float], int], List[Node]],
                        # breed: Callable[[List[Node]], List[Node]],
                        # verify: Callable[[Node], bool]
                       # ) -> Optional[Node]:
result = genetic_programming(g, 5, 2, 2, select, breed)
print(result)



