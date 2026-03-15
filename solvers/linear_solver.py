import sympy as sp
from models.step import EquationStep
from math_utils.mmc import compute_mmc

x = sp.symbols("x")


def solve_linear(expr):

    steps = []

    solution = sp.solve(expr, x)

    steps.append(
        EquationStep(
            before=expr,
            after=f"x={sp.latex(solution[0])}"
        )
    )

    return steps