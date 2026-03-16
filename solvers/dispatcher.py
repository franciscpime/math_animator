from solvers.linear_solver import solve_linear
from solvers.quadratic_solver import solve_quadratic
from solvers.polynomial_solver import solve_polynomial
from solvers.notable_products_solver import solve as solve_notable


def dispatch_solver(
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

    degree = polynomial.degree()

    if degree == 1:

        return solve_linear(equation)

    if degree == 2:

        return solve_quadratic(
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
    result = solve_notable(expr)

    if result:
        return result
    
    return solve_polynomial(polynomial, equation)