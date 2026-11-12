"""End-to-end verification of the COS30019 assignment algorithms.

Run inside the folder containing the group's Python files and test graphs:
    python verify_project.py

For each problem, compare route validity against the input graph, BFS hop-count
oracle, and Dijkstra weighted-cost oracle. DFS and GBFS are only required to
return valid paths, not necessarily shortest ones.
"""
from collections import deque
import heapq
from pathlib import Path

from route_problem import RouteProblem, read_problem
from search import METHODS, run_search

TEST_FILES = [
    'test_case/PathFinder-test-1.txt',
    *(f'test_case/testcase{number}.txt' for number in range(1, 17)),
]


def min_moves(problem):
    q = deque([(problem.initial, 0)])
    seen = {problem.initial}
    while q:
        state, steps = q.popleft()
        if problem.goal_test(state):
            return steps
        for nxt in problem.actions(state):
            if nxt not in seen:
                seen.add(nxt)
                q.append((nxt, steps + 1))
    return None


def min_weighted_cost(problem):
    q = [(0, problem.initial)]
    best = {problem.initial: 0}
    while q:
        cost, state = heapq.heappop(q)
        if cost > best[state]:
            continue
        if problem.goal_test(state):
            return cost
        for nxt in problem.actions(state):
            trial = cost + problem.graph[state][nxt]
            if trial < best.get(nxt, float('inf')):
                best[nxt] = trial
                heapq.heappush(q, (trial, nxt))
    return None


def verify_path(problem, result):
    if result.goal is None:
        return result.path == [] and result.path_cost is None
    path = result.path
    if not path or path[0] != problem.initial or path[-1] != result.goal:
        return False
    if not problem.goal_test(result.goal):
        return False
    if any(v not in problem.graph[u] for u, v in zip(path, path[1:])):
        return False
    cost = sum(problem.graph[u][v] for u, v in zip(path, path[1:]))
    return abs(result.path_cost - cost) < 1e-7 and result.nodes_created >= 1


def verify_problem(problem, name):
    moves = min_moves(problem)
    cost = min_weighted_cost(problem)
    checked = 0
    print(f'\n{name}: min moves={moves}, min cost={cost}')
    for method in METHODS:
        result = run_search(problem, method)
        ok = verify_path(problem, result)
        if moves is None:
            ok = ok and result.goal is None
        else:
            ok = ok and result.goal is not None
        if method in ('BFS', 'CUS1', 'CUS2') and moves is not None:
            ok = ok and result.moves == moves
        if method == 'AS' and cost is not None:
            ok = ok and abs(result.path_cost - cost) < 1e-7
        checked += int(ok)
        print(f'  {method:4} {"PASS" if ok else "FAIL"} | '
              f'moves={result.moves} cost={result.path_cost} '
              f'created={result.nodes_created} path={result.path}')
    return checked


def main():
    successes = 0
    for filename in TEST_FILES:
        problem = read_problem(Path(__file__).parent / filename)
        successes += verify_problem(problem, filename)
    # Extra scenarios not dependent on external files.
    scenarios = [
        ('Unreachable goal', RouteProblem({1: {2: 1}, 2: {1: 1}, 3: {}}, 1, {3},
                                          {1: (0, 0), 2: (1, 0), 3: (5, 5)})),
        ('Origin is goal', RouteProblem({1: {2: 3}, 2: {}}, 1, {1},
                                       {1: (0, 0), 2: (1, 0)})),
    ]
    for name, problem in scenarios:
        successes += verify_problem(problem, name)
    total = (len(TEST_FILES) + len(scenarios)) * len(METHODS)
    print(f'\nResults: {successes}/{total} checks passed')
    if successes != total:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
