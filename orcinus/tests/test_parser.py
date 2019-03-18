import io

from orcinus.parser import Parser
from orcinus.scanner import Scanner
from orcinus.syntax import SyntaxTree, SyntaxContext, FunctionNode, ParameterNode, NamedTypeNode


def parse_string(content: str) -> SyntaxTree:
    context = SyntaxContext()
    with Scanner('test', io.StringIO(content), diagnostics=context.diagnostics) as scanner:
        parser = Parser('test', scanner, context)
        return parser.parse()


def test_function():
    parse_string("""
def main() -> int: ...
    """)


def test_incorrect_function():
    tree = parse_string("""
def main(a: , b: int) -> :
""")

    assert len(tree.members) == 1

    func = tree.members[0]
    assert isinstance(func, FunctionNode)
    assert func.name == 'main'
    assert len(func.parameters) == 2

    arg1 = func.parameters[0]
    assert isinstance(arg1, ParameterNode)
    assert arg1.name == 'a'
    assert isinstance(arg1.type, NamedTypeNode)
    assert arg1.type.name == ''

    arg2 = func.parameters[1]
    assert isinstance(arg2, ParameterNode)
    assert arg2.name == 'b'
