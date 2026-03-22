import sympy as sp
from models.step import Step
from math_utils.mmc import compute_mmc
 
x = sp.symbols("x")
 
 
def solve_quadratic(
    polynomial,
    equation,
    mmc,
    scaled_expression,
    is_factorized=False,
    m=None,
    n=None,
    o=None,
    p=None
):
 
    steps = []
 
    # coeficientes
    a, b, c = polynomial.all_coeffs()
 
    delta = b**2 - 4*a*c
 
    # FIX H: Step usa before/after em vez de expr (que não existe no modelo)
 
    # mostrar equação original
    steps.append(
        Step(
            before=equation,
            after=sp.latex(scaled_expression),
            explanation="Multiply both sides to clear fractions" if mmc != 1 else ""
        )
    )
 
    # identificar coeficientes
    steps.append(
        Step(
            before=sp.latex(scaled_expression),
            after=sp.latex(scaled_expression),
            explanation=f"$a={sp.latex(a)},\\; b={sp.latex(b)},\\; c={sp.latex(c)}$"
        )
    )
 
    # fórmula do delta
    steps.append(
        Step(
            before=sp.latex(scaled_expression),
            after="\\Delta=b^2-4ac",
            explanation="Apply the discriminant formula"
        )
    )
 
    # substituir valores no delta
    steps.append(
        Step(
            before="\\Delta=b^2-4ac",
            after=f"\\Delta=({sp.latex(b)})^2-4({sp.latex(a)})({sp.latex(c)})"
        )
    )
 
    # calcular delta
    steps.append(
        Step(
            before=f"\\Delta=({sp.latex(b)})^2-4({sp.latex(a)})({sp.latex(c)})",
            after=f"\\Delta={sp.latex(delta)}"
        )
    )
 
    # fórmula resolvente
    steps.append(
        Step(
            before=f"\\Delta={sp.latex(delta)}",
            after="x=\\frac{-b\\pm\\sqrt{\\Delta}}{2a}",
            explanation="Apply the quadratic formula"
        )
    )
 
    # substituir valores na fórmula
    steps.append(
        Step(
            before="x=\\frac{-b\\pm\\sqrt{\\Delta}}{2a}",
            after=f"x=\\frac{{{sp.latex(-b)}\\pm\\sqrt{{{sp.latex(delta)}}}}}{{{sp.latex(2*a)}}}"
        )
    )
 
    # resolver
    solutions = sp.solve(polynomial, x)
 
    steps.append(
        Step(
            before=f"x=\\frac{{{sp.latex(-b)}\\pm\\sqrt{{{sp.latex(delta)}}}}}{{{sp.latex(2*a)}}}",
            after=f"x={sp.latex(solutions)}"
        )
    )
 
    return steps