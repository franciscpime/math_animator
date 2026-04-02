import sympy as sp


def safe_gcd(a, b):
    """
    Return the greatest common divisor of two values, converting both to
    SymPy Rationals first so the function works correctly with fractional
    coefficients as well as plain integers.

    Parameters
    ----------
    a : numeric -- first value (int, float, or sp.Rational)
    b : numeric -- second value (int, float, or sp.Rational)

    Returns
    -------
    sp.Rational -- the GCD of a and b
    """
    a_rational = sp.Rational(a)
    b_rational = sp.Rational(b)

    return sp.gcd(a_rational, b_rational)

