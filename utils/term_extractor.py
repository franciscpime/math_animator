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