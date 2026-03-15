import sympy as sp

from parser.equation_analyzer import analyze_equation
from math_animator.solvers.linear_solver import solve_linear
from math_animator.solvers.quadratic_solver import solve_quadratic
from math_animator.solvers.polynomial_solver import solve_polynomial

x = sp.symbols("x")


def detect_factorized(left: str):
    return ")(" in left or ")*(" in left


def extract_factor_terms(left):

    parts = left.split("*")

    first = parts[0].strip("()")
    second = parts[1].strip("()")

    first_terms = [t for t in first.replace("-", "+-").split("+") if t]
    second_terms = [t for t in second.replace("-", "+-").split("+") if t]

    m = sp.sympify(first_terms[0])
    n = sp.sympify(first_terms[1])
    o = sp.sympify(second_terms[0])
    p = sp.sympify(second_terms[1])

    return m, n, o, p


def solve_equation(equation: str):

    steps = []

    left, right = analyze_equation(equation)

    # Detect factorized form
    is_factorized = detect_factorized(left)

    m = n = o = p = None

    if is_factorized:
        m, n, o, p = extract_factor_terms(left)

    # Build symbolic expression
    expr = sp.sympify(left) - sp.sympify(right)

    if is_factorized:
        original_expression = expr
    else:
        original_expression = sp.expand(expr)

    polynomial = sp.Poly(sp.expand(expr), x)

    # Detect denominators
    coeffs = polynomial.all_coeffs()
    denominators = []

    for c in coeffs:
        num, den = c.as_numer_denom()
        denominators.append(den)

    mmc = sp.lcm(*denominators)

    # Remove denominators
    if mmc != 1:

        scaled_expression = sp.expand(original_expression * mmc)

        scaled_polynomial = sp.Poly(scaled_expression, x)

        polynomial = scaled_polynomial

    else:

        scaled_expression = original_expression

    # Determine polynomial degree
    degree = polynomial.degree()

    if degree == 1:

        steps += solve_linear(left, right, equation)

    elif degree == 2:

        steps += solve_quadratic(
            polynomial,
            equation,
            mmc,
            scaled_expression,
            is_factorized,
            m,
            n,
            o,
            p
        )

    else:

        steps += solve_polynomial(polynomial, equation)

    return steps
