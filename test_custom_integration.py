"""Run the group's actual RouteProblem parser against Member 3's algorithms.

Keep this file alongside route_problem.py, custom_search.py, and the five
Stage 3 graph input files, then run: python test_custom_integration.py
"""
from collections import deque
from pathlib import Path

from route_problem import read_problem
from custom_search import iterative_lengthening_search, ida_star_moves

TEST_FILES = (
    'test_11_large_graph.txt',
    'test_12_long_route.txt',
    'test_13_many_branches.txt',
    'test_14_merging_paths.txt',
    'test_15_difficult_combo.txt',
)


def shortest_moves(problem):
    """Independent BFS oracle used only by the test runner."""
    queue = deque([(problem.initial, 0)])
    seen = {problem.initial}
    while queue:
        state, moves = queue.popleft()
        if problem.goal_test(state):
            return moves
        for action in problem.actions(state):
            next_state = problem.result(state, action)
            if next_state not in seen:
                seen.add(next_state)
                queue.append((next_state, moves + 1))
    return None


def valid_path(problem, result, expected_moves):
    if expected_moves is None:
        return result.goal is None and not result.path
    path = result.path
    if not path or path[0] != problem.initial or path[-1] != result.goal:
        return False
    if not problem.goal_test(result.goal) or len(path) - 1 != expected_moves:
        return False
    return all(
        any(problem.result(src, action) == dst for action in problem.actions(src))
        for src, dst in zip(path, path[1:])
    )


def main():
    passed = 0
    for name in TEST_FILES:
        problem = read_problem(str(Path(__file__).parent / name))
        expected = shortest_moves(problem)
        for label, algorithm in (
            ('CUS1', iterative_lengthening_search),
            ('CUS2', ida_star_moves),
        ):
            actual = algorithm(problem)
            ok = valid_path(problem, actual, expected)
            passed += int(ok)
            print(
                f'{name} {label}: {"PASS" if ok else "FAIL"} '
                f'| expected moves={expected} '
                f'| created={actual.nodes_created} '
                f'| path={actual.path}'
            )
    print(f'Results: {passed}/{2 * len(TEST_FILES)} checks passed')


if __name__ == '__main__':
    main()
