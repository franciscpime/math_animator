import sympy as sp


def render_terms(terms):

    # Protege contra listas vazias — devolve "0" em vez de string vazia
    if not terms:
        return "0"

    parts = [sp.latex(t) for t in terms]
    expr = " + ".join(parts)
    expr = expr.replace("+ -", "- ")

    return expr


def build_equation(left_terms, right_terms):
    left_side = render_terms(left_terms)
    right_side = render_terms(right_terms)
    return f"{left_side} = {right_side}"