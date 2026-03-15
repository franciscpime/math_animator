import sympy as sp
from models.step import EquationStep

x = sp.symbols("x")


def solve_polynomial(expr):

    steps = []

    solutions = sp.solve(expr, x)

    steps.append(
        EquationStep(
            before=expr,
            after=f"x={sp.latex(solutions)}"
        )
    )

    return steps