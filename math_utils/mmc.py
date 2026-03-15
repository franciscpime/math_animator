import sympy as sp

def compute_mmc(polynomial):

    coeffs = polynomial.all_coeffs()

    denominators = []

    for c in coeffs:
        num, den = c.as_numer_denom()
        denominators.append(den)

    return sp.lcm(*denominators)