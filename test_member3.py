"""Run the custom-search checks on renamed Stage 3 test cases.

Run with: python test_member3.py
"""
from pathlib import Path

from custom_search import cus1_search, cus2_search
from route_problem import read_problem
from verify_project import min_moves, verify_path

TEST_CASE_DIR = Path(__file__).parent / 'test_case'
TEST_FILES = tuple(f'testcase{number}.txt' for number in range(12, 17))


def main():
    passed = 0
    total = 0
    for filename in TEST_FILES:
        problem = read_problem(TEST_CASE_DIR / filename)
        expected_moves = min_moves(problem)
        for method, algorithm in (('CUS1', cus1_search), ('CUS2', cus2_search)):
            result = algorithm(problem)
            ok = verify_path(problem, result)
            if expected_moves is None:
                ok = ok and result.goal is None
            else:
                ok = ok and result.goal is not None and result.moves == expected_moves
            total += 1
            passed += int(ok)
            print(
                f'{filename} {method}: {"PASS" if ok else "FAIL"} '
                f'| expected moves={expected_moves} | created={result.nodes_created} '
                f'| path={result.path}'
            )
    print(f'Results: {passed}/{total} checks passed')
    if passed != total:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
