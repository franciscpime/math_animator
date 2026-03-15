import sympy as sp


def extract_factor_product(expr):

    if not isinstance(expr, sp.Mul):
        return None

    factors = expr.args

    if len(factors) != 2:
        return None

    f1, f2 = factors

    if isinstance(f1, sp.Add) and isinstance(f2, sp.Add):
        return f1, f2

    return None