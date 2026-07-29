"""math.calculate — safe arithmetic via a restricted AST walker (never eval/exec).

Supported: + - * / **, unary +/-, parentheses, int/float literals.
Everything else (names, calls, attributes, subscripts, collections,
comprehensions, lambdas, boolean/compare ops, strings, ...) is rejected with a
controlled INVALID_ARGUMENTS error.

Documented limits (explicit, to bound cost and prevent abuse):
  - MAX_EXPR_LEN     = 200   max characters in the expression string
  - MAX_AST_DEPTH    = 20    max nesting depth of the parsed expression
  - MAX_EXPONENT     = 100   max |exponent| for ** (prevents huge computations)
  - MAX_MAGNITUDE    = 1e15  max |value| for any literal or intermediate result
"""

import ast
import operator

from tools.base import BaseTool, ToolValidationError
from tools.models import ToolPermission

MAX_EXPR_LEN = 200
MAX_AST_DEPTH = 20
MAX_EXPONENT = 100
MAX_MAGNITUDE = 1e15

_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
}
_UNARY_OPS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


class CalculatorError(ToolValidationError):
    """Any rejected/invalid expression -> INVALID_ARGUMENTS."""


def _check_magnitude(value):
    if isinstance(value, complex):
        raise CalculatorError("Complex results are not supported.")
    if abs(value) > MAX_MAGNITUDE:
        raise CalculatorError(f"Value exceeds the maximum magnitude of {MAX_MAGNITUDE:g}.")
    return value


def _eval_node(node, depth):
    if depth > MAX_AST_DEPTH:
        raise CalculatorError(f"Expression nesting exceeds the maximum depth of {MAX_AST_DEPTH}.")

    if isinstance(node, ast.Expression):
        return _eval_node(node.body, depth)

    # Numeric literal (Python 3.8+: ast.Constant). Reject str/bool/None/bytes/complex.
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise CalculatorError("Only numeric literals are allowed.")
        return _check_magnitude(node.value)

    if isinstance(node, ast.BinOp):
        op_type = type(node.op)
        if op_type not in _BIN_OPS:
            raise CalculatorError(f"Unsupported operator: {op_type.__name__}.")
        left = _eval_node(node.left, depth + 1)
        right = _eval_node(node.right, depth + 1)
        if op_type is ast.Div and right == 0:
            raise CalculatorError("Division by zero.")
        if op_type is ast.Pow:
            if abs(right) > MAX_EXPONENT:
                raise CalculatorError(f"Exponent exceeds the maximum of {MAX_EXPONENT}.")
        return _check_magnitude(_BIN_OPS[op_type](left, right))

    if isinstance(node, ast.UnaryOp):
        op_type = type(node.op)
        if op_type not in _UNARY_OPS:
            raise CalculatorError(f"Unsupported unary operator: {op_type.__name__}.")
        return _check_magnitude(_UNARY_OPS[op_type](_eval_node(node.operand, depth + 1)))

    # Anything else — names, calls, attributes, subscripts, lists/tuples/dicts/sets,
    # comprehensions, lambdas, bool/compare ops, strings, etc. — is rejected.
    raise CalculatorError(f"Disallowed expression element: {type(node).__name__}.")


def calculate(expression: str):
    if not isinstance(expression, str):
        raise CalculatorError("'expression' must be a string.")
    expression = expression.strip()
    if not expression:
        raise CalculatorError("'expression' must not be empty.")
    if len(expression) > MAX_EXPR_LEN:
        raise CalculatorError(f"'expression' exceeds {MAX_EXPR_LEN} characters.")
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError:
        raise CalculatorError("The expression could not be parsed.")
    result = _eval_node(tree, 0)
    # Normalize integral floats to int for a cleaner result (396.0 -> 396).
    if isinstance(result, float) and result.is_integer():
        result = int(result)
    return result


class CalculatorTool(BaseTool):
    name = "math.calculate"
    permission = ToolPermission.READ  # pure computation, no side effects
    description = (
        "Evaluate a basic arithmetic expression. Supports + - * / ** , parentheses, "
        "decimals, and unary minus. Example: (17 * 23) + 5"
    )
    input_schema = {
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": "The arithmetic expression to evaluate, e.g. '(17 * 23) + 5'.",
            }
        },
        "required": ["expression"],
    }
    timeout_seconds = 5.0

    def validate_arguments(self, arguments: dict) -> dict:
        arguments = super().validate_arguments(arguments)
        expr = arguments.get("expression")
        if not isinstance(expr, str):
            raise CalculatorError("'expression' must be a string.")
        return arguments

    def execute(self, arguments: dict) -> dict:
        expr = arguments["expression"]
        result = calculate(expr)
        return {"expression": expr, "result": result}
