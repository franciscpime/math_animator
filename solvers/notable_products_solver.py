from sympy import expand
from analyzers.notable_products import detect_notable
from models.step import Step


def solve(expr):

    notable_type, match = detect_notable(expr)

    if notable_type is None:
        return None

    expanded = expand(expr)

    steps = []

    steps.append(
        Step(
            description="Aplicar caso notável",
            before=expr,
            after=expanded
        )
    )

    return expanded, steps