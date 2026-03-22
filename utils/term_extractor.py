import sympy as sp
import re

x = sp.symbols("x")


def extract_terms(expression: str):

    expression = expression.replace(" ", "")

    terms = re.findall(r'[+-]?[^+-]+', expression)

    sympy_terms = []

    for term in terms:

        if term in ["+", "-"]:
            continue

        sympy_terms.append(sp.parse_expr(term))

    return sympy_terms


def detailed_multiplication(expr):
    """Return a list of intermediate SymPy expressions that show how a product
    of rationals/nums is combined step by step. If expr is not a Mul, returns []
    """
    steps = []

    if not isinstance(expr, sp.Mul):
        return steps

    args = expr.args

    nums = []
    dens = []

    for a in args:
        if isinstance(a, sp.Rational):
            nums.append(a.p)
            dens.append(a.q)
        else:
            nums.append(a)
            dens.append(1)

    num_expr = sp.Mul(*nums)
    den_expr = sp.Mul(*dens)

    step1 = sp.Mul(num_expr, sp.Pow(den_expr, -1))
    steps.append(step1)

    step2 = sp.together(step1)
    steps.append(step2)

    step3 = sp.simplify(step2)
    steps.append(step3)

    return steps

