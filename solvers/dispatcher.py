import sympy as sp

from solvers.linear_solver import solve_linear
from solvers.quadratic_solver import solve_quadratic
from solvers.polynomial_solver import solve_polynomial
from solvers.notable_products_solver import solve_notable_product


x = sp.symbols("x")


def dispatch_solver(expr):

    poly = sp.Poly(expr, x)

    degree = poly.degree()

    if is_notable_product(expr):
        return solve_notable_product(expr)

    if degree == 1:
        return solve_linear(expr)

    if degree == 2:
        return solve_quadratic(expr)

    if degree >= 3:
        return solve_polynomial(expr)

    raise ValueError("Unsupported equation")