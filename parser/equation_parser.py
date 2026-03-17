import sympy as sp
import re

def normalize_expression(expr):
    expr = expr.replace("^", "**")

    expr = re.sub(r'(\d)([a-zA-Z])', r'\1*\2', expr)
    expr = re.sub(r'([a-zA-Z])\(', r'\1*(', expr)
    expr = re.sub(r'\)([a-zA-Z0-9])', r')*\1', expr)
    expr = re.sub(r'\)\(', ')*(', expr)

    return expr


def parse_equation(equation: str):

    left, right = equation.split("=")
    left = left.strip()
    right = right.strip()

    left = normalize_expression(left)
    right = normalize_expression(right)

    return left, right