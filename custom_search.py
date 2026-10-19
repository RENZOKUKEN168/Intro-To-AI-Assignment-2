"""Custom search algorithms for COS30019 Assignment 2.

CUS1: Iterative Deepening Depth-First Search (IDDFS), uninformed.
CUS2: Iterative Deepening A* (IDA*), informed; minimizes the number of moves.

Both methods receive RouteProblem and return the shared SearchResult via
route_problem.make_result(). The method aliases at the bottom preserve the
current search.py dispatcher's existing function names.
"""

from collections import deque
from math import ceil, inf

from route_problem import Node, make_result


def iddfs_search(problem, trace=False):
    """CUS1: IDDFS, exploring paths in ascending successor order.

    A fresh search-tree root is created at every depth limit. A search path
    cannot revisit a state, but the same state may occur on another branch.
    This is tree search with path-cycle checking, not graph-wide pruning.
    """
    created = 0
    expanded = 0
    history = []

    def visit(node, states_on_path, depth, depth_limit):
        nonlocal created, expanded

        is_goal = problem.goal_test(node.state)
        if trace:
            history.append({
                'step': len(history) + 1,
                'state': node.state,
                'depth': depth,
                'limit': depth_limit,
                'status': 'goal' if is_goal else 'visited',
            })
        if is_goal:
            return node, False

        if depth == depth_limit:
            # Continue deepening only if there is an unvisited path successor.
            cutoff = any(
                problem.result(node.state, action) not in states_on_path
                for action in problem.actions(node.state)
            )
            return None, cutoff

        expanded += 1
        saw_cutoff = False
        for action in problem.actions(node.state):
            next_state = problem.result(node.state, action)
            if next_state in states_on_path:
                continue
            child = node.child_node(problem, action)
            created += 1
            found, child_cutoff = visit(
                child, states_on_path | {next_state}, depth + 1, depth_limit
            )
            if found is not None:
                return found, False
            saw_cutoff = saw_cutoff or child_cutoff

        return None, saw_cutoff

    depth_limit = 0
    while True:
        root = Node(problem.initial)
        created += 1
        found, cutoff = visit(root, {root.state}, 0, depth_limit)
        if found is not None:
            return make_result(found, created, expanded, trace=history)
        if not cutoff:
            return make_result(None, created, expanded, trace=history)
        depth_limit += 1


def _move_heuristic(problem):
    """Build an admissible hop-count heuristic from geometric distances.

    RouteProblem.heuristics('greedy') supplies h_geo(s), the straight-line
    distance to the nearest destination. Let D be the largest single-edge
    *decrease* in h_geo among states reachable from the origin. Any path of
    k edges to a goal can lower h_geo by at most k*D, so:

        h_moves(s) = ceil(h_geo(s)/D) <= true remaining moves.

    If D=0, use the safe zero heuristic. Only states (not Node search-tree
    records) are explored during this one-time heuristic preprocessing.
    """
    geo = problem.heuristics('greedy')
    seen = {problem.initial}
    frontier = deque([problem.initial])
    max_decrease = 0.0

    while frontier:
        state = frontier.popleft()
        for action in problem.actions(state):
            nxt = problem.result(state, action)
            max_decrease = max(max_decrease, float(geo[state] - geo[nxt]))
            if nxt not in seen:
                seen.add(nxt)
                frontier.append(nxt)

    if max_decrease <= 0:
        return lambda state: 0

    def heuristic(state):
        estimate = float(geo[state]) / max_decrease
        return max(0, ceil(estimate - 1e-12))

    return heuristic


def ida_star_moves(problem, trace=False):
    """CUS2: IDA* with an admissible move-count heuristic.

    Each directed edge counts as one move, irrespective of its weight.
    Thus this method minimizes number of edges rather than weighted cost.
    """
    heuristic = _move_heuristic(problem)
    created = 0
    expanded = 0
    history = []
    bound = heuristic(problem.initial)

    def visit(node, states_on_path, moves, limit):
        nonlocal created, expanded
        score = moves + heuristic(node.state)
        if score > limit:
            return None, score

        is_goal = problem.goal_test(node.state)
        if trace:
            history.append({
                'step': len(history) + 1,
                'state': node.state,
                'moves': moves,
                'priority': score,
                'limit': limit,
                'status': 'goal' if is_goal else 'visited',
            })
        if is_goal:
            return node, None

        expanded += 1
        next_bound = inf
        for action in problem.actions(node.state):
            next_state = problem.result(node.state, action)
            if next_state in states_on_path:
                continue
            child = node.child_node(problem, action)
            created += 1
            found, excess = visit(
                child, states_on_path | {next_state}, moves + 1, limit
            )
            if found is not None:
                return found, None
            if excess is not None:
                next_bound = min(next_bound, excess)

        return None, next_bound

    while True:
        root = Node(problem.initial)
        created += 1
        found, new_bound = visit(root, {root.state}, 0, bound)
        if found is not None:
            return make_result(found, created, expanded, trace=history)
        if new_bound == inf:
            return make_result(None, created, expanded, trace=history)
        bound = new_bound


# Convenient descriptive names for new code.
cus1_search = iddfs_search
cus2_search = ida_star_moves

# Compatibility with Member 1's existing search.py dispatcher.
# The dispatcher labels CUS1 "iterative_lengthening_search", but this
# implementation is IDDFS (depth-limited, not cumulative-cost lengthening).
iterative_lengthening_search = iddfs_search
