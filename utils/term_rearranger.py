import sympy as sp

from utils.term_extractor import extract_terms

x = sp.symbols("x")


def rearrange_terms(left, right):
    """
    Split both sides of the equation into variable (x) terms and constant
    terms, then move everything to the canonical sides:
      - All x-terms   >> left side  (right-side x-terms are negated)
      - All constants >> right side (left-side constants are negated)

    Parameters
    ----------
    left  : str -- the left-hand side expression string (already normalized)
    right : str -- the right-hand side expression string (already normalized)

    Returns
    -------
    variable_terms : list[sp.Expr] -- x-terms to place on the left
    constant_terms : list[sp.Expr] -- constants to place on the right
    """
    left_terms  = extract_terms(left)
    right_terms = extract_terms(right)

    left_x     = []
    left_const = []

    for term in left_terms:
        if term.has(x):
            left_x.append(term)
        else:
            left_const.append(term)

    right_x     = []
    right_const = []

    for term in right_terms:
        if term.has(x):
            right_x.append(term)
        else:
            right_const.append(term)

    # Variable terms: keep left-side ones as-is; negate right-side ones
    # (moving across the equals sign flips the sign).
    variable_terms = []

    for term in left_x:
        variable_terms.append(term)

    for term in right_x:
        variable_terms.append(-term)

    # Constant terms: keep right-side ones as-is; negate left-side ones.
    constant_terms = []

    for term in right_const:
        constant_terms.append(term)

    for term in left_const:
        constant_terms.append(-term)

    return variable_terms, constant_terms

