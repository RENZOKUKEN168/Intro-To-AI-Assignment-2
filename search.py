"""Run: python search.py <filename> <method>."""
import argparse
from importlib import import_module
import sys
from route_problem import read_problem

# Other algorithms can be added to their own files using these function names.
METHODS = {
    'DFS': ('basic_search', 'depth_first_search'),
    'BFS': ('basic_search', 'breadth_first_search'),
    'GBFS': ('inform_search', 'greedy_best_first_search'),
    'AS': ('inform_search', 'astar_search'),
    'CUS1': ('custom_search', 'iterative_lengthening_search'),
    'CUS2': ('custom_search', 'ida_star_moves'),
}


def run_search(problem, method):
    """Import only the selected method, so unfinished methods do not block it."""
    module_name, function_name = METHODS[method]
    try:
        module = import_module(module_name)
    except ModuleNotFoundError as error:
        if error.name != module_name:
            raise
        raise NotImplementedError(
            f'{method} is not implemented in {module_name}.py yet.') from error
    algorithm = getattr(module, function_name, None)
    if not callable(algorithm):
        raise NotImplementedError(
            f'Add {function_name}(problem) to {module_name}.py first.')
    return algorithm(problem)


def print_result(filename, method, result):
    """Keep successful standard output in the brief's three-part format."""
    print(filename, method)
    if result.goal is None:
        # The brief specifies successful output only; this is our failure format.
        print('NONE', result.nodes_created)
        print('NO PATH')
        return 1
    print(result.goal, result.nodes_created)
    print('; '.join(map(str, result.path)))
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description='Tree Based Search: route finding')
    parser.add_argument('filename', help='Map text file in the assignment format')
    parser.add_argument('method', type=str.upper, choices=METHODS)
    args = parser.parse_args(argv)
    try:
        problem = read_problem(args.filename)
        result = run_search(problem, args.method)
    except (OSError, UnicodeError, ValueError, NotImplementedError) as error:
        print(f'Error: {error}', file=sys.stderr)
        return 2
    return print_result(args.filename, args.method, result)


if __name__ == '__main__':
    sys.exit(main())
