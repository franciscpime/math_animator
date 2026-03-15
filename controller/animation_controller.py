import sympy as sp

from parser.equation_parser import parse_equation
from solvers.dispatcher import dispatch_solver
from animation.equation_renderer import EquationRenderer

x = sp.symbols("x")


class AnimationController:

    def __init__(self, scene):
        self.renderer = EquationRenderer(scene)

    def run(self, equation):

        left, right = parse_equation(equation)

        expr = sp.sympify(left) - sp.sympify(right)

        # detectar fatorização
        is_factorized = False
        m = n = o = p = None

        if isinstance(left, sp.Mul):

            factors = left.args

            if len(factors) == 2 and all(isinstance(f, sp.Add) for f in factors):

                is_factorized = True

                m, n = factors[0].args
                o, p = factors[1].args

        polynomial = sp.Poly(sp.expand(expr), x)

        coeffs = polynomial.all_coeffs()

        denominators = []

        for c in coeffs:
            num, den = c.as_numer_denom()
            denominators.append(den)

        mmc = sp.lcm(*denominators)

        if mmc != 1:

            scaled_expression = sp.expand(expr * mmc)
            polynomial = sp.Poly(scaled_expression, x)

        else:

            scaled_expression = expr

        steps = dispatch_solver(
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

        self.renderer.animate(steps)