from sympy import expand
from analyzers.notable_products import detect_notable


def solve_notable(expr):

    kind, match = detect_notable(expr)

    if kind is None:
        return None

    result = expand(expr)

    return {
        "type": kind,
        "original": expr,
        "result": result
    }