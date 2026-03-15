import sympy as sp
from models.step import EquationStep

def solve_notable_product(expr):
    from solvers.dispatcher import dispatch_solver
    
    steps = []

    expanded = sp.expand(expr)

    steps.append(
        EquationStep(
            before=expr,
            after=expanded,
            explanation="Expand the product"
        )
    )

    next_steps = dispatch_solver(expanded)

    return steps + next_steps