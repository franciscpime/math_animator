import sympy as sp
import re

x = sp.symbols("x")

"""
This function extracts terms from a mathematical expression.
So that they can be processed individually, because
it allows for easier manipulation and analysis of each term.

Example:
Expression: 2*x + 3 - 4*x
Result: [2*x, 3, -4*x]
"""
def extract_terms(expression: str):
    # Remove whitespace
    expression = expression.replace(" ", "")

    # Find all terms that are either numbers (with optional signs) 
    # or the variable x
    terms = re.findall(r'[+-]?[^+-]+', expression)

    # Convert each term into a SymPy expression
    sympy_terms = []

    for term in terms:
        if term in ["+", "-"]:
            continue

        sympy_terms.append(sp.parse_expr(term))

    return sympy_terms

