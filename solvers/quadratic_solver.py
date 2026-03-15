import sympy as sp
from models.step import EquationStep
from math_utils.mmc import compute_mmc

x = sp.symbols("x")


def solve_quadratic(
    polynomial,
    equation,
    mmc,
    scaled_expression,
    is_factorized,
    m,
    n,
    o,
    p
):

    steps = []

    # coeficientes
    a, b, c = polynomial.all_coeffs()

    delta = b**2 - 4*a*c

    # mostrar equação após MMC (se houver)
    steps.append(
        EquationStep(
            before=equation,
            after=sp.latex(scaled_expression)
        )
    )

    # identificar coeficientes
    steps.append(
        EquationStep(
            before=sp.latex(scaled_expression),
            after=sp.latex(scaled_expression),
            explanation=f"a={sp.latex(a)},\\; b={sp.latex(b)},\\; c={sp.latex(c)}"
        )
    )

    # fórmula do delta
    steps.append(
        EquationStep(
            before=sp.latex(scaled_expression),
            after="\\Delta=b^2-4ac"
        )
    )

    # substituir valores
    steps.append(
        EquationStep(
            before="\\Delta=b^2-4ac",
            after=f"\\Delta=({sp.latex(b)})^2-4({sp.latex(a)})({sp.latex(c)})"
        )
    )

    # calcular delta
    steps.append(
        EquationStep(
            before=f"\\Delta=({sp.latex(b)})^2-4({sp.latex(a)})({sp.latex(c)})",
            after=f"\\Delta={sp.latex(delta)}"
        )
    )

    # fórmula resolvente
    steps.append(
        EquationStep(
            before=f"\\Delta={sp.latex(delta)}",
            after="x=\\frac{-b\\pm\\sqrt{\\Delta}}{2a}"
        )
    )

    # substituir valores
    steps.append(
        EquationStep(
            before="x=\\frac{-b\\pm\\sqrt{\\Delta}}{2a}",
            after=f"x=\\frac{{{-sp.latex(b)}\\pm\\sqrt{{{sp.latex(delta)}}}}}{{{sp.latex(2*a)}}}"
        )
    )

    # resolver
    solutions = sp.solve(polynomial, x)

    steps.append(
        EquationStep(
            before=f"x=\\frac{{{-sp.latex(b)}\\pm\\sqrt{{{sp.latex(delta)}}}}}{{{sp.latex(2*a)}}}",
            after=f"x={sp.latex(solutions)}"
        )
    )

    return steps