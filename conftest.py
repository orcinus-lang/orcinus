from orcinus.syntax import SyntaxToken


def pytest_assertrepr_compare(op, left, right):
    if isinstance(left, SyntaxToken) and isinstance(right, SyntaxToken) and op == "==":
        return [
            'Comparing Foo instances:',
            f'   values: {left.id.name} != {right.id.name}',
            f'   left location: {left.location}',
            f'   right location: {right.location}',
        ]
