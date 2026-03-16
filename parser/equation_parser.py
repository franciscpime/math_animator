import sympy as sp
import re

def normalize_expression(expr):
    expr = expr.replace("^", "**")

    # 2x → 2*x
    expr = re.sub(r'(\d)([a-zA-Z])', r'\1*\2', expr)

    return expr


def parse_equation(equation: str):

    left, right = equation.split("=")

    left = normalize_expression(left)
    right = normalize_expression(right)

    left_expr = sp.sympify(left)
    right_expr = sp.sympify(right)

    return left_expr, right_expr