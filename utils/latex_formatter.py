import sympy as sp


def to_latex(expr):

    if isinstance(expr, str):
        return expr

    if isinstance(expr, sp.Basic):
        return sp.latex(expr)

    return str(expr)