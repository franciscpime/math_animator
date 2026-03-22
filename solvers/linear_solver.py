import sympy as sp
from sympy import lcm
import re
from functools import reduce
from models.step import Step
from parser.equation_parser import parse_equation, normalize_expression, safe_sympify, fix_implicit_mul
from math_utils.mmc import compute_mmc, apply_mmc
from utils.term_extractor import extract_terms, detailed_multiplication
from utils.equation_builder import build_equation, render_terms
 
# Define a variável simbólica x que será usada nas equações
x = sp.symbols("x")  

# FIX 1 + FIX 2: Combina termos passo a passo garantindo que só combina termos do mesmo tipo
def combine_terms_stepwise(terms):
    
    new_terms = terms.copy()
    steps = []
    
    i = 0
    while i < len(new_terms) - 1:
        t1 = new_terms[i]
        t2 = new_terms[i + 1]

        # Só combina se forem do mesmo tipo (ambos com x, ou ambos constantes)
        both_have_x = t1.has(x) and t2.has(x)
        both_are_const = not t1.has(x) and not t2.has(x)

        if both_have_x:
            coef = t1.coeff(x) + t2.coeff(x)
            new_term = coef * x
            new_terms = new_terms[:i] + [new_term] + new_terms[i + 2:]
            steps.append(new_terms.copy())
            # Não avança i — pode haver mais termos do mesmo tipo para combinar
        elif both_are_const:
            new_term = t1 + t2
            new_terms = new_terms[:i] + [new_term] + new_terms[i + 2:]
            steps.append(new_terms.copy())
        else:
            # Tipos diferentes — avança sem combinar
            i += 1

    return steps


def substitution_steps(expr, value):
    steps = []

    expr_sub = expr.subs(x, sp.Integer(value), evaluate=False)
    steps.append(expr_sub)

    expanded = sp.expand(expr_sub)
    if expanded != expr_sub:
        steps.append(expanded)

    final = sp.simplify(expanded)
    if final != expanded:
        steps.append(final)

    return steps


def common_divisor_of_constants(terms):
    """Return the largest rational divisor that divides all constant terms.

    For rational constants this returns gcd(numerators)/lcm(denominators) as a
    Rational so it can represent integer or fractional common divisors.
    """
    if not terms:
        return sp.Integer(1)

    # Convert all terms to Rational (works for ints and rationals)
    rats = [sp.Rational(t) for t in terms]

    # denominators and their least common multiple
    denoms = [r.q for r in rats]
    mmc = lcm(*denoms) if len(denoms) > 0 else 1

    # scale to common denominator and compute gcd of numerators
    nums = [int((r * mmc)) for r in rats]
    gcd_nums = abs(reduce(sp.gcd, [sp.Integer(n) for n in nums])) if nums else 1

    return sp.Rational(gcd_nums, mmc)


# helpers (safe_sympify, fix_implicit_mul, apply_mmc, detailed_multiplication)
# are provided by parser/math_utils/utils modules and imported above.


# Função principal que resolve equações lineares passo a passo
def solve_linear(equation: str):  
    
    left, right = parse_equation(equation)  

    steps = []

    steps.append(
        Step(
            before=equation,  
            after=equation,  
            explanation="Organizar termos"  
        )
    )

    left_terms = extract_terms(left)
    right_terms = extract_terms(right)

    left_x, left_const = [], []
    right_x, right_const = [], []

    for t in left_terms:
        if t.has(x):
            left_x.append(t)
        else:
            left_const.append(t)

    for t in right_terms:
        if t.has(x):
            right_x.append(t)
        else:
            right_const.append(t)

    variable_terms = left_x + [-t for t in right_x]
    constant_terms = right_const + [-t for t in left_const]

    new_eq = build_equation(variable_terms, constant_terms)

    if len(variable_terms) > 1:
        steps.append(
            Step(
                before=new_eq,
                after=new_eq,
                    explanation="Vamos resolver o lado das variáveis"
            )
        )

    # --- Simplify variables ---
    current_vars = variable_terms
    var_steps = combine_terms_stepwise(variable_terms)

    for new_vars in var_steps:
        steps.append(
            Step(
                before=build_equation(current_vars, constant_terms),
                after=build_equation(new_vars, constant_terms)
            )
        )
        current_vars = new_vars

    # --- Simplify constants ---
    current_consts = constant_terms

    if len(constant_terms) > 1:
        steps.append(
            Step(
                before=build_equation(current_vars, current_consts),
                after=build_equation(current_vars, current_consts),
                explanation="Agora vamos resolver o lado das constantes"
            )
        )

    const_steps = combine_terms_stepwise(constant_terms)

    for new_consts in const_steps:
        # compute the largest common divisor for the current constants
        divisor = common_divisor_of_constants(current_consts)

        # evitar mostrar 1 como divisor
        if divisor == 1:
            explanation = None
        else:
            explanation = f"Dividir ambos os lados por {sp.latex(divisor)}"

        steps.append(
            Step(
                before=build_equation(current_vars, current_consts),
                after=build_equation(current_vars, new_consts),
                explanation=explanation
            )
        )
        current_consts = new_consts  

    # Final solve
    final_left = current_vars[0]
    final_right = current_consts[0]

    coef = final_left.coeff(x)
    const = final_right

    # FIX 10: usa sp.Rational diretamente em vez de int() para o GCD, evitando truncagem
    def safe_gcd(a, b):
        return sp.gcd(sp.Rational(a), sp.Rational(b))

    def _check_solution(final_value, final_latex, equation, steps):
        """FIX 4: Verificação da solução extraída para função separada, chamada em ambos os casos."""

        # show a verification caption without replacing the current equation
        steps.append(
            Step(
                before=equation,
                after=equation,
                explanation="Vamos verificar!"
            )
        )

        steps.append(
            Step(
                before=equation,
                after=equation,
                explanation=f"Agora vamos substituir x por {final_latex}"
            )
        )
        
        # FIX 5: substituição via SymPy.
        # substituted usa str() — expressão Python válida — para que stepwise_string_eval
        # consiga fazer sp.sympify() sem erros.
        # sp.latex() é usado APENAS no Step de display.
        left_str, right_str = equation.split("=")
        left_sym  = sp.sympify(normalize_expression(left_str.strip()), evaluate=False)
        right_sym = sp.sympify(normalize_expression(right_str.strip()), evaluate=False)

        # keep the substituted expression unevaluated so we can show
        # step-by-step simplification (avoid jumping straight to the numeric result)
        left_evaled  = sp.sympify(left_sym.subs(x, final_value), evaluate=False)
        right_evaled = sp.sympify(right_sym.subs(x, final_value), evaluate=False)

        substituted = f"{str(left_evaled)} = {str(right_evaled)}"

        # Show the immediate substituted expression as a Step (LaTeX) so the
        # user sees the substitution before the simplification sequence.
        substituted_display = f"{sp.latex(left_evaled)} = {sp.latex(right_evaled)}"
        steps.append(
            Step(
                before=equation,
                after=substituted_display,
            )
        )

        def stepwise_string_eval(expr):
            """Avalia uma expressão passo-a-passo usando o AST do SymPy.

            Gera passos intermédios para multiplicações e somas numéricas,
            mostrando primeiro as multiplicações (ex: 2·3 -> 6) e depois as
            somas (ex: 6+1 -> 7). Retorna uma lista de passos (strings/LaTeX).
            """
            steps_local = []

            # Normalizar multiplicação implícita e obter expressão SymPy
            cur = fix_implicit_mul(expr)
            expr_sym = safe_sympify(cur)
            expr_sym = sp.sympify(expr_sym, evaluate=False)

            # Função auxiliar: encontrar e simplificar um subexpressão numérica
            def simplify_one(sym):
                # Ensure we are working with a SymPy expression. Sometimes callers
                # pass native Python numbers/strings which don't implement
                # preorder_traversal, causing an AttributeError. Coerce early.
                try:
                    sym = safe_sympify(sym)
                except Exception:
                    return None

                # 1) procurar um Mul com fatores numéricos
                for sub in sym.preorder_traversal():
                    if isinstance(sub, sp.Mul):
                        if all(a.is_number for a in sub.args):
                            steps_local.append(str(sub))   # ex: 2*3
                            prod = sp.nsimplify(sp.prod(sub.args))
                            new = sym.xreplace({sub: prod})
                            steps_local.append(str(prod))  # ex: 6
                            return new

                # 2) procurar Add com termos numéricos que possam ser somados
                for sub in sym.preorder_traversal():
                    if isinstance(sub, sp.Add):
                        numeric_terms = [t for t in sub.args if isinstance(t, (sp.Integer, sp.Rational, sp.Float))]
                        if len(numeric_terms) > 1:
                            # mostrar a soma dos termos numéricos
                            # construir representação antes da soma
                            before = sp.Add(*sub.args)
                            # calcular soma dos termos numéricos
                            s = sum(numeric_terms)
                            # construir novo subexpressão substituindo os termos numéricos pela soma
                            other_terms = [t for t in sub.args if t not in numeric_terms]
                            new_sub = sp.Add(*(other_terms + [s])) if other_terms else s
                            steps_local.append(str(before))
                            steps_local.append(str(new_sub))
                            new = sym.xreplace({sub: new_sub})
                            return new

                return None

            # Iterativamente simplificar multiplicações e somas até ficar estável
            current = expr_sym
            while True:
                nxt = simplify_one(current)
                if nxt is None:
                    break
                current = nxt

            # Se ainda houver Rationals a tratar com mmc
            try:
                ordered = list(current.as_ordered_terms())
            except Exception:
                ordered = [current]

            if any(isinstance(t, sp.Rational) for t in ordered):
                mmc, new_num = apply_mmc(current)

                # só mostrar fração se realmente necessário
                if mmc != 1:
                    step_mmc = f"{sp.latex(current)} = \\frac{{{sp.latex(new_num)}}}{{{mmc}}}"
                    steps_local.append(step_mmc)

                    result_rational = sp.Rational(new_num, mmc)
                    current = result_rational
                    steps_local.append(str(current))

            # Finalmente simplificar
            simplified = sp.simplify(current)

            # só adiciona se ainda NÃO for número final simples
            if isinstance(current, (sp.Add, sp.Mul)):
                if str(simplified) != str(current):
                    steps_local.append(str(simplified))

            return steps_local

        left_steps = stepwise_string_eval(substituted.split("=")[0])
        right_steps = stepwise_string_eval(substituted.split("=")[1])

        if len(left_steps) == 1:
            left_steps = [substituted.split("=")[0].strip()] + left_steps

        if len(right_steps) == 1:
            right_steps = [substituted.split("=")[1].strip()] + right_steps

        current_left = substituted.split("=")[0].strip()
        current_right = substituted.split("=")[1].strip()

        for i in range(max(len(left_steps), len(right_steps))):

            # sanitize sides to avoid chaining '=' that may appear in LaTeX/display steps
            before_left = current_left.split("=")[0].strip()
            before_right = current_right.split("=")[-1].strip()
            before = f"{before_left} = {before_right}"

            if i < len(left_steps):
                current_left = left_steps[i]

            if i < len(right_steps):
                current_right = right_steps[i]

            after_left = current_left.split("=")[0].strip()
            after_right = current_right.split("=")[-1].strip()
            after = f"{after_left} = {after_right}"

            steps.append(
                Step(
                    before=before,
                    after=after
                )
            )

        # Extrai apenas o lado relevante (ignora "=" residuais na string)
        left_for_check = current_left.split("=")[0].strip()
        right_for_check = current_right.split("=")[-1].strip()

        # safe_sympify lida com LaTeX (\frac, etc.) e com expressões Python normais
        left_final = safe_sympify(left_for_check)
        right_final = safe_sympify(right_for_check)

        is_true = sp.simplify(left_final - right_final) == 0

        # FIX 8: mensagem de verificação em inglês, consistente com o resto do código
        explanation = "The solution is correct!" if is_true else "The solution does not satisfy the equation."

        steps.append(
            Step(
                before=f"{current_left} = {current_right}",
                after=f"{current_left} = {current_right}",
                explanation=explanation
            )
        )

    # CASO 1: coeficiente já é 1 (ex: x = 3)
    if coef == 1:
        steps.append(
            Step(
                before=f"{sp.latex(final_left)} = {sp.latex(final_right)}",
                after=f"x = {sp.latex(const)}"
            )
        )

        # FIX 4: verificação também no caso 1
        final_value = const
        final_latex = sp.latex(final_value)
        _check_solution(final_value, final_latex, equation, steps)

    # CASO 2: precisa dividir
    else:
        gcd = safe_gcd(abs(const), abs(coef))  # FIX 10

        if gcd > 1:
            new_left = final_left / gcd
            new_right = final_right / gcd

            # só mostra se NÃO ficar x isolado
            if new_left.coeff(x) != 1:
                steps.append(
                    Step(
                        before=f"{sp.latex(final_left)} = {sp.latex(final_right)}",
                        after=f"{sp.latex(new_left)} = {sp.latex(new_right)}",
                        explanation="Now the term on the left moves to the right, dividing it"
                    )
                )

            final_left = new_left
            final_right = new_right

        coef = final_left.coeff(x)
        const = final_right

        solution = sp.Rational(const, coef)

        # FIX 3: bloco de divisão final sem duplicação — apenas um Step
        steps.append(
            Step(
                before=f"{sp.latex(final_left)} = {sp.latex(final_right)}",
                after=f"x = {sp.latex(solution)}"
            )
        )

        final_value = solution
        final_latex = sp.latex(final_value)

        # FIX 4: verificação chamada via função partilhada
        _check_solution(final_value, final_latex, equation, steps)

    return steps
