from sympy import lcm
from functools import reduce
import sympy as sp

def compute_mmc(expressions):
    """
    Calcula o mínimo múltiplo comum de vários denominadores.
    """

    if not expressions:
        return None

    return reduce(lcm, expressions)


def apply_mmc(expr):
    """Scale expression by MMC and return (mmc, scaled_sum_of_numerators)."""
    expr = sp.sympify(expr)

    terms = expr.as_ordered_terms()
    denominators = [sp.fraction(t)[1] for t in terms]

    mmc = compute_mmc(denominators)

    new_terms = []
    for t in terms:
        num, den = sp.fraction(t)
        factor = sp.Rational(mmc, den)
        new_terms.append(num * factor)

    result = sum(new_terms)

    return mmc, result