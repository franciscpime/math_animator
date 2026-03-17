from sympy import expand
from analyzers.notable_products import notable_products
from models.step import Step


def solve(expr):

    steps = []
    result = notable_products(expr, steps, Step)

    if result is None:
        return None

    return result, steps
