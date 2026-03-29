import sympy as sp
from solvers.linear_solver import solve_linear
from solvers.quadratic_solver import solve_quadratic
from parser.equation_parser import normalize_expression
 
x = sp.symbols("x")
 
 
def dispatch_solver(
    equation: str,
    polynomial=None,
    mmc=None,
    scaled_expression=None,
    is_factorized=False,
    m=None, n=None, o=None, p=None
):
    """
    Encaminha a equação para o solver correto consoante o grau.
 
    O AnimationController calcula polynomial, mmc e scaled_expression
    antes de chamar aqui, e passa-os como argumentos.
    Se chamado sem esses argumentos (ex: testes unitários),
    o dispatcher calcula-os internamente a partir da string.
    """
 
    left, right = equation.split("=")
 
    right_norm = normalize_expression(right.strip().replace(" ", ""))
    left_norm  = normalize_expression(left.strip().replace(" ", ""))
 
    expr = sp.expand(sp.sympify(f"{left_norm}-({right_norm})"))
 
    # Se o polynomial não foi passado, calcula aqui
    if polynomial is None:
        polynomial = sp.Poly(expr, x)
 
    if scaled_expression is None:
        scaled_expression = expr
 
    if mmc is None:
        mmc = sp.Integer(1)
 
    degree = polynomial.degree()
 
    if degree == 1:
        return solve_linear(equation)
 
    if degree == 2:
        # FIX C: passa todos os argumentos que solve_quadratic precisa
        return solve_quadratic(
            polynomial=polynomial,
            equation=equation,
            mmc=mmc,
            scaled_expression=scaled_expression,
            is_factorized=is_factorized,
            m=m, n=n, o=o, p=p
        )
 
    raise ValueError(f"Unsupported equation degree: {degree}")

