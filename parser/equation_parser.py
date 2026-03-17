import sympy as sp
import re

def normalize_expression(expr):
    expr = expr.replace("^", "**")

    # 2x → 2*x
    expr = re.sub(r'(\d)([a-zA-Z])', r'\1*\2', expr)

    # x( → x*(
    expr = re.sub(r'([a-zA-Z])\(', r'\1*(', expr)

    # )x ou )2 → )*x / )*2
    expr = re.sub(r'\)([a-zA-Z0-9])', r')*\1', expr)

    # )( → )*(
    expr = re.sub(r'\)\(', ')*(', expr)

    return expr


def parse_equation(equation: str):

    left, right = equation.split("=")

    left = normalize_expression(left)
    right = normalize_expression(right)

    left_expr = sp.sympify(left)
    right_expr = sp.sympify(right)

    return left_expr, right_expr