from parsers.sexp import sexp
from program import *
from typing import List, Any
from collections import defaultdict
import math

class Grammar:
    def __init__(self, bmfile: str) -> None:

        li = self.__load_bm_grammar(bmfile)
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
                    rhs = VarNode(var_name=i, expr_type=lhs.expr_type)
                    self.terminal_rules[lhs.nt_name].append(Production(lhs, rhs))
                elif isinstance(i, tuple):
                    rhs = ConstNode(expr_type=i[0], value=i[1])
                    self.terminal_rules[lhs.nt_name].append(Production(lhs, rhs))
                else: # list
                    rhs = FuncNode(func_name=i[0], expr_type=lhs.expr_type)
                    for j in i[1:]:
                        rhs.children.append(NTNode(nt_name=j, expr_type=type_dict[j]))
                self.all_rules[lhs.nt_name].append(Production(lhs, rhs))

    def __load_bm_grammar(self, bmfile: str) -> List[Any]:
        def stripComments(bmf):
            noComments = "("
            for line in bmf:
                line = line.split(";", 1)[0]
                noComments += line
            return noComments + ")"

        with open(bmfile) as f:
            bm = stripComments(f)
            bme = sexp.parse_string(bm, parse_all=True).asList()[0]
        return bme[1][4]



    def get_hyperparameters(self, constraints, experiment, selection):
        types = len(self.all_rules)
        total = sum([len(self.all_rules[k]) for k in self.all_rules])
        term = sum([len(self.terminal_rules[k]) for k in self.terminal_rules])
        func = total - term
        
        max_arity = 0
        for k in self.all_rules:
            getMax = max([len(prod.rhs.children) if isinstance(prod.rhs, FuncNode) else 0 for prod in self.all_rules[k]])
            if getMax > max_arity:
                max_arity = getMax
        
        if experiment == 'min':
            population_size = 100        
        elif experiment == 'max':
            population_size = 3000        
        elif experiment == 'types': # exponents chosen to get population size within desired range
            population_size = types**6        
        elif experiment == 'arity':
            population_size = arity**6            
        elif experiment == 'func':
            population_size = func**3            
        elif experiment == 'term':
            population_size = term**3            
        elif experiment == 'mix':
            population_size = types**5 + arity**5 + func**2 + term**2 
        else: # explicit size given
            population_size = int(experiment)    
        
        if population_size > 3000:
            population_size = 3000
            
        if selection == 'min':
            num_selection = 2        
        elif selection == 'max':
            num_selection = population_size        
        elif selection == 'most':
            num_selection = math.floor(population_size*.95)# 95% of population size
        elif selection == 'best':
            num_selection = math.floor(population_size*.05)# 5% of population size
        else: # explicit percentage given
            num_selection = math.floor(population_size*int(selection)/100)
        
        max_generation = 1 + math.floor(total/2) + math.floor(constraints/total)*2 # if many constraints then generations will increase
        max_generation = max_generation*math.floor(3000/population_size/5) # if population_size is small, we need more generations        
            
        return (population_size, max_generation, num_selection)