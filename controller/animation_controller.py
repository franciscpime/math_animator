import sympy as sp
from functools import reduce
 
from parser.equation_parser import parse_equation, normalize_expression
from solvers.dispatcher import dispatch_solver
from animation.equation_renderer import EquationRenderer
 
x = sp.symbols("x")
 
 
class AnimationController:
 
    def __init__(self, scene):
        self.renderer = EquationRenderer(scene)
 
    def run(self, equation):
 
        # parse_equation devolve strings normalizadas
        left_str, right_str = parse_equation(equation)
 
        # FIX A: converter para SymPy DEPOIS do parse, não antes
        left_sym = sp.sympify(left_str, evaluate=False)
        right_sym = sp.sympify(right_str, evaluate=False)
 
        expr = sp.expand(left_sym - right_sym, evaluate=False)
 
        # FIX A: deteção de fatorização feita sobre objetos SymPy, não strings
        is_factorized = False
        m = n = o = p = None
 
        if isinstance(left_sym, sp.Mul):
            factors = left_sym.args
            if len(factors) == 2 and all(isinstance(f, sp.Add) for f in factors):
                is_factorized = True
                m, n = factors[0].args
                o, p = factors[1].args
 
        polynomial = sp.Poly(expr, x)
        coeffs = polynomial.all_coeffs()
 
        # FIX B: usa compute_mmc (reduce com lcm) em vez de sp.lcm(*lista)
        #         que só aceita exatamente 2 argumentos
        denominators = []
        for c in coeffs:
            _, den = c.as_numer_denom()
            denominators.append(den)
 
        mmc = reduce(sp.lcm, denominators) if denominators else sp.Integer(1)
 
        if mmc != 1:
            scaled_expression = sp.expand(expr * mmc, evaluate=False)
            polynomial = sp.Poly(scaled_expression, x)
        else:
            scaled_expression = expr
 
        # FIX C: dispatch_solver trata internamente do grau;
        #         os argumentos extra do quadratic são passados via keyword
        steps = dispatch_solver(
            equation=equation,
            polynomial=polynomial,
            mmc=mmc,
            scaled_expression=scaled_expression,
            is_factorized=is_factorized,
            m=m, n=n, o=o, p=p
        )

        for i, step in enumerate(steps):
            print(f"Step {i}: before={repr(step.before)} | after={repr(step.after)} | explanation={repr(step.explanation)}")
        
        self.renderer.animate(steps)

