from sympy import lcm
from functools import reduce

def compute_mmc(expressions):
    """
    Calcula o mínimo múltiplo comum de vários denominadores.
    """

    if not expressions:
        return None

    return reduce(lcm, expressions)