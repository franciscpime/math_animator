from sympy import sp
from models.step import EquationStep

def solve_notable_product(expr):

    steps = []

    expanded = sp.expand(expr)

    steps.append(
        EquationStep(
            before=expr,
            after=expanded,
            explanation="Expand the product"
        )
    )

    from solvers.dispatcher import dispatch_solver

    next_steps = dispatch_solver(expanded)

    return steps + next_steps