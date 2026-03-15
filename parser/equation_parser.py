import sympy as sp
import re


def normalize_expression(expr: str):
    # transforma "(x+2)(x+3)" em "(x+2)*(x+3)"
    expr = re.sub(r"\)\(", ")*(", expr)
    return expr


def parse_equation(equation: str):

    left, right = equation.split("=")

    left = normalize_expression(left)
    right = normalize_expression(right)

    left_expr = sp.sympify(left)
    right_expr = sp.sympify(right)

    return left_expr, right_expr