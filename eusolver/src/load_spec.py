import sys

from exprs import exprs
from utils import utils
from parsers import parser
from verifiers import verifiers
from core import specifications
from exprs import expr_transforms


def get_pbe_valuations(constraints, synth_fun):

    valuations = []

    for constraint in constraints:

        if not exprs.is_application_of(constraint, 'eq') and \
                not exprs.is_application_of(constraint, '='):
            return None

        if len(exprs.get_all_variables(constraint)) > 0:
            return None

        arg_func, arg_other = None, None

        for a in constraint.children:
            if exprs.is_application_of(a, synth_fun):
                arg_func = a
            else:
                arg_other = a

        if arg_func is None or arg_other is None:
            return None

        valuations.append((arg_func.children, arg_other))

    return valuations


def make_specification(synth_funs, theory, syn_ctx, constraints):
 
    if not expr_transforms.is_single_invocation(constraints, theory, syn_ctx):
 
        specification = specifications.MultiPointSpec(syn_ctx.make_function_expr('and', *constraints),
                syn_ctx, synth_funs)
        syn_ctx.set_synth_funs(synth_funs)
        verifier = verifiers.MultiPointVerifier(syn_ctx, specification)
 
    elif len(synth_funs) == 1 and get_pbe_valuations(constraints, synth_funs[0]) is not None:
 
        synth_fun = synth_funs[0]
        valuations = get_pbe_valuations(constraints, synth_fun)
        specification = specifications.PBESpec(valuations, synth_fun, theory)
        syn_ctx.set_synth_funs(synth_funs)
        verifier = verifiers.PBEVerifier(syn_ctx, specification)
 
    else:

        spec_expr = constraints[0] if len(constraints) == 1 \
                else syn_ctx.make_function_expr('and', *constraints)

        specification = specifications.StandardSpec(spec_expr, syn_ctx, synth_funs, theory)
        syn_ctx.set_synth_funs(synth_funs)
        verifier = verifiers.StdVerifier(syn_ctx, specification)

    return specification, verifier


def massage_constraints(syn_ctx, macro_instantiator, uf_instantiator, theory, constraints):
    # Instantiate all macro functions

    constraints = [ macro_instantiator.instantiate_all(c)
            for c in constraints ]

    constraints = expr_transforms.AckermannReduction.apply(constraints, uf_instantiator, syn_ctx)

    constraints = expr_transforms.LetFlattener.apply(constraints, syn_ctx)

    constraints = expr_transforms.RewriteITE.apply(constraints, syn_ctx)

    return constraints


if __name__ == '__main__':
    
    benchmark_files = sys.argv[1:]

    for benchmark_file in benchmark_files:

        print("Processing benchmark file start", benchmark_file)
        file_sexp = parser.sexpFromFile(benchmark_file)

        # Print the file string expression to see the spec

        for a in file_sexp:
            print(a)

        # Extract the benchmark details from the file expression list
        benchmark_tuple = parser.extract_benchmark(file_sexp)
        theories, syn_ctx, synth_instantiator, macro_instantiator, uf_instantiator, constraints, grammar_map, forall_vars_map = benchmark_tuple

        theory = theories[0]

        # Re write constraints if feasible
        rewritten_constraints = utils.timeout(massage_constraints, (syn_ctx, macro_instantiator, uf_instantiator, theory, constraints), {}, timeout_duration=120, default=None)

        # Make the specification for the benchmark
        synth_funs = list(synth_instantiator.get_functions().values())
        specification, verifier = make_specification(synth_funs, theory, syn_ctx, constraints)

        solver_params = (theory, syn_ctx, synth_funs, grammar_map, specification, verifier)

        print("Processing benchmark file end", benchmark_file)
