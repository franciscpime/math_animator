import sympy as sp

from utils.term_extractor import extract_terms

x = sp.symbols("x")

'''
Split both sides of the equation into variable (x) terms and constant
terms, then move everything to the canonical sides:
    - All x-terms   >> left side  (right-side x-terms are negated)
    - All constants >> right side (left-side constants are negated)

Example: 3x + 2 = 5x - 4
    left_x = [3x]
    left_const = [2]
    right_x = [5x]
    right_const = [-4]
Then:
    x_terms = [3x - 5x]
    constant_terms = [4 - 2]
'''
def rearrange_terms(left, right):
    
    left_terms  = extract_terms(left)
    right_terms = extract_terms(right)

    left_x = []
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
    x_terms = []

    for term in left_x:
        x_terms.append(term)

    for term in right_x:
        x_terms.append(-term)

    # Constant terms: keep right-side ones as-is; negate left-side ones.
    constant_terms = []

    for term in right_const:
        constant_terms.append(term)

    for term in left_const:
        constant_terms.append(-term)

    return x_terms, constant_terms

