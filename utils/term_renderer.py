import sympy as sp

def render_terms(terms):

    parts = []

    for term in terms:
        parts.append(sp.latex(term))

    expression = " + ".join(parts)

    expression = expression.replace("+ -", "- ")

    return expression