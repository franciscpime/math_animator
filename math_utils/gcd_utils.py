import sympy as sp

'''
This function returns the greatest common divisor (GCD) of two values.

Both values are converted to SymPy Rationals first so the function works
correctly with fractions as well as plain integers.

Examples with integers:
    safe_gcd(4, 6)
    >> a_rational = Rational(4, 1)  -- 4/1, just 4
    >> b_rational = Rational(6, 1)  -- 6/1, just 6
    >> gcd(4/1, 6/1)  >>  2         -- largest number that divides both 4 and 6

Examples with fractions:
    safe_gcd(Rational(1, 2), Rational(3, 4))
    >> gcd(1/2, 3/4)  >>  1/4       -- largest piece that fits into both 1/2 and 3/4 exactly
                                    --   (1/2) / (1/4) = 2  (no remainder)
                                    --   (3/4) / (1/4) = 3  (no remainder)

    safe_gcd(Rational(1, 4), Rational(1, 6))
    >> gcd(1/4, 1/6)  >>  1/12      -- largest piece that fits into both 1/4 and 1/6 exactly
                                    --   (1/4) / (1/12) = 3  (no remainder)
                                    --   (1/6) / (1/12) = 2  (no remainder)
'''
def safe_gcd(a: sp.Rational | int, b: sp.Rational | int):

    a_rational = sp.Rational(a)
    b_rational = sp.Rational(b)

    return sp.gcd(a_rational, b_rational)

