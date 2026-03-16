import sympy as sp
from models.step import Step

x = sp.symbols("x")


def solve_polynomial(expr):

    steps = []

    solutions = sp.solve(expr, x)

    steps.append(
        Step(
            before=expr,
            after=f"x={sp.latex(solutions)}"
        )
    )

    return steps