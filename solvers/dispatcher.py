import sympy as sp
from solvers.linear_solver import solve_linear
from solvers.quadratic_solver import solve_quadratic
from parser.equation_parser import normalize_expression

x = sp.symbols("x")


def dispatch_solver(equation: str):

    left, right = equation.split("=")

    right_norm = normalize_expression(right.strip().replace(" ", ""))
    left_norm = normalize_expression(left.strip().replace(" ", ""))

    expr = sp.expand(sp.sympify(f"{left_norm}-({right_norm})"))
    poly = sp.Poly(expr, x)

    degree = poly.degree()

    if degree == 1:
        return solve_linear(equation)

    if degree == 2:
        return solve_quadratic(equation)

    raise ValueError("Unsupported equation type")