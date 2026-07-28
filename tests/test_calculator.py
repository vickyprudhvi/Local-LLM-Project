"""math.calculate restricted-AST evaluator: arithmetic + every rejection + limits."""

import pytest

from tools.calculator import (
    MAX_EXPONENT,
    MAX_EXPR_LEN,
    CalculatorError,
    calculate,
)


@pytest.mark.parametrize("expr,expected", [
    ("2 + 3", 5),
    ("10 - 4", 6),
    ("6 * 7", 42),
    ("20 / 4", 5),
    ("(17 * 23) + 5", 396),
    ("2.5 + 1.5", 4),
    ("1.5 * 2", 3),
    ("-5 + 2", -3),
    ("+7", 7),
    ("2 ** 10", 1024),
    ("(1 + 2) * (3 + 4)", 21),
    ("7 / 2", 3.5),
])
def test_valid_expressions(expr, expected):
    assert calculate(expr) == expected


def test_integral_float_normalized_to_int():
    result = calculate("10 / 2")
    assert result == 5
    assert isinstance(result, int)


@pytest.mark.parametrize("expr", [
    "1/0",           # division by zero
    "2 +",           # invalid syntax
    "x + 1",         # variable / name
    "abs(-5)",       # function call
    "(2).real",      # attribute access
    "[1, 2, 3]",     # list
    "(1, 2)",        # tuple
    "{1: 2}",        # dict
    "{1, 2}",        # set
    "[i for i in range(3)]",  # comprehension
    "lambda: 1",     # lambda
    "1 and 2",       # boolean op
    "1 < 2",         # comparison
    "'hello'",       # string literal
    "True + 1",      # bool literal rejected
    "1 % 2",         # unsupported operator (mod)
    "5 << 1",        # unsupported operator (shift)
])
def test_rejected_expressions(expr):
    with pytest.raises(CalculatorError):
        calculate(expr)


def test_excessive_length_rejected():
    expr = "+".join(["1"] * (MAX_EXPR_LEN))  # far exceeds MAX_EXPR_LEN chars
    with pytest.raises(CalculatorError):
        calculate(expr)


def test_excessive_exponent_rejected():
    with pytest.raises(CalculatorError):
        calculate(f"2 ** {MAX_EXPONENT + 1}")


def test_excessive_depth_rejected():
    # A long left-associative chain nests BinOp nodes deeper than MAX_AST_DEPTH.
    # (Parentheses alone add no AST depth, so real operators are used.)
    expr = "+".join(["1"] * 30)
    with pytest.raises(CalculatorError):
        calculate(expr)


def test_huge_literal_rejected():
    with pytest.raises(CalculatorError):
        calculate("1e400")  # inf / beyond max magnitude


def test_empty_expression_rejected():
    with pytest.raises(CalculatorError):
        calculate("   ")
