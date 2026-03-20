import sympy as sp  
from sympy import lcm
import re  
from models.step import Step  
from parser.equation_parser import parse_equation  
from math_utils.mmc import compute_mmc
 
# Define a variável simbólica x que será usada nas equações
x = sp.symbols("x")  
 
# Função que recebe uma expressão (string) e devolve os termos separados
def extract_terms(expr: str):  
    expr = expr.replace(" ", "")  
    terms = re.findall(r'[+-]?[^+-]+', expr)  
    
    sympy_terms = []
    
    for term in terms:  
        if term in ["+", "-"]:
            continue
 
        sympy_terms.append(sp.sympify(term))
 
    return sympy_terms
 
 
# Função que transforma uma lista de termos numa string LaTeX
def render_terms(terms):
 
    # FIX 9: Protege contra listas vazias — devolve "0" em vez de string vazia
    if not terms:
        return "0"
  
    parts = [sp.latex(t) for t in terms]  
    expr = " + ".join(parts)  
    expr = expr.replace("+ -", "- ")  
 
    return expr
 
 
# Constrói uma equação completa a partir de duas listas de termos
def build_equation(left_terms, right_terms):  
 
    left = render_terms(left_terms)
    right = render_terms(right_terms)
 
    return f"{left} = {right}"
 
 
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
 
 
# FIX 7: detailed_multiplication agora é chamada em stepwise_string_eval quando necessário
def detailed_multiplication(expr):
    steps = []
 
    if isinstance(expr, sp.Mul):
        args = expr.args
 
        nums = []
        dens = []
 
        for a in args:
            if isinstance(a, sp.Rational):
                nums.append(a.p)
                dens.append(a.q)
            else:
                nums.append(a)
                dens.append(1)
 
        num_expr = sp.Mul(*nums)
        den_expr = sp.Mul(*dens)
 
        step1 = sp.Mul(num_expr, sp.Pow(den_expr, -1))
        steps.append(step1)
 
        step2 = sp.together(step1)
        steps.append(step2)
 
        step3 = sp.simplify(step2)
        steps.append(step3)
 
    return steps
 
 
def fix_implicit_mul(expr: str):
    # 10(3/4) → 10*(3/4)
    expr = re.sub(r'(\d)\(', r'\1*(', expr)
    return expr
 
 
# FIX 6: apply_mmc usa divisão simbólica (sp.Rational) em vez de // para evitar erros com frações
def apply_mmc(expr):
    expr = sp.sympify(expr)
 
    terms = expr.as_ordered_terms()
    denominators = [sp.fraction(t)[1] for t in terms]
 
    mmc = compute_mmc(denominators)
 
    new_terms = []
    for t in terms:
        num, den = sp.fraction(t)
        factor = sp.Rational(mmc, den)   # divisão simbólica em vez de //
        new_terms.append(num * factor)
 
    result = sum(new_terms)
 
    return mmc, result
 
 
def latex_to_sympy(expr: str) -> str:
    """Converte notação LaTeX básica para sintaxe que o sympify consegue parsear."""
    # \frac{a}{b} → (a)/(b)
    expr = re.sub(r'\\frac\{([^}]+)\}\{([^}]+)\}', r'(\1)/(\2)', expr)
    # Remove backslashes soltos que o sympify não entende
    expr = expr.replace("\\", "")
    # Remove espaços extra
    expr = expr.strip()
    return expr
 
 
def safe_sympify(expr: str) -> sp.Basic:
    """
    Tenta parsear uma expressão que pode estar em formato LaTeX ou Python.
    Estratégia:
      1. Tenta parse_latex (lida com \frac, expoentes, etc.)
      2. Fallback: limpa com latex_to_sympy e usa sympify normal
    Nunca lança excepção — devolve sp.nan em caso de falha total.
    """
    from sympy.parsing.latex import parse_latex as _parse_latex
 
    # Limpa espaços
    expr = expr.strip()
 
    # Tenta parse_latex primeiro (mais robusto para strings LaTeX)
    try:
        return _parse_latex(expr)
    except Exception:
        pass
 
    # Fallback: converte manualmente e usa sympify
    try:
        cleaned = latex_to_sympy(expr)
        cleaned = fix_implicit_mul(cleaned)
        return sp.sympify(cleaned)
    except Exception:
        return sp.nan
 
 
# Função principal que resolve equações lineares passo a passo
def solve_linear(equation: str):  
    
    left, right = parse_equation(equation)  
 
    steps = []
 
    steps.append(
        Step(
            before=equation,  
            after=equation,  
            explanation="Organise terms"  
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
                explanation="Let's solve the left side"
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
                explanation="Now let's solve the right side"
            )
        )
 
    const_steps = combine_terms_stepwise(constant_terms)
 
    for new_consts in const_steps:
        steps.append(
            Step(
                before=build_equation(current_vars, current_consts),
                after=build_equation(current_vars, new_consts)
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
 
        steps.append(
            Step(
                before="",
                after="",
                explanation="Let's check!"
            )
        )
 
        steps.append(
            Step(
                before="",
                after=equation,
                explanation=f"Now let's replace x by ${final_latex}$"
            )
        )
 
        # FIX 5: substituição via SymPy em vez de str.replace(), evitando erros com coeficientes
        left_str, right_str = equation.split("=")
        left_sym = sp.sympify(left_str.replace("x", f"*({final_value})").lstrip("*"))
        right_sym = sp.sympify(right_str.replace("x", f"*({final_value})").lstrip("*"))
 
        substituted_left = sp.latex(left_sym.subs(x, final_value))
        substituted_right = sp.latex(right_sym.subs(x, final_value))
        substituted = f"{substituted_left} = {substituted_right}"
 
        steps.append(
            Step(
                before=equation,
                after=substituted
            )
        )
 
        def stepwise_string_eval(expr):
            steps_local = []
            current = expr
 
            current = fix_implicit_mul(current)
 
            mult_pattern = r'(\d+)\(([^)]+)\)'
 
            while re.search(mult_pattern, current):
                new = re.sub(
                    mult_pattern,
                    lambda m: str(sp.sympify(m.group(1)) * sp.sympify(m.group(2))),
                    current
                )
                steps_local.append(new)
 
                # FIX 7: chama detailed_multiplication para mostrar passos intermédios de multiplicação
                sym_expr = sp.sympify(new)
                if isinstance(sym_expr, sp.Mul):
                    mul_steps = detailed_multiplication(sym_expr)
                    for ms in mul_steps:
                        steps_local.append(sp.latex(ms))
 
                current = new
 
            expr_sym = sp.sympify(current)
 
            if any(isinstance(t, sp.Rational) for t in expr_sym.as_ordered_terms()):
 
                mmc, new_num = apply_mmc(expr_sym)
 
                # step para mostrar ao utilizador (LaTeX) — só para display
                step_mmc = f"{sp.latex(expr_sym)} = \\frac{{{sp.latex(new_num)}}}{{{mmc}}}"
                steps_local.append(step_mmc)
 
                # current mantém-se como expressão Python válida (não LaTeX)
                result_rational = sp.Rational(new_num, mmc)
                current = str(result_rational)  # ex: "55/2" — sympify consegue parsear
 
            result = str(sp.simplify(current))
 
            if result != current:
                steps_local.append(result)
 
            return steps_local
 
        left_steps = stepwise_string_eval(substituted.split("=")[0])
        right_steps = stepwise_string_eval(substituted.split("=")[1])
 
        current_left = substituted.split("=")[0]
        current_right = substituted.split("=")[1]
 
        for i in range(max(len(left_steps), len(right_steps))):
 
            before = f"{current_left} = {current_right}"
 
            if i < len(left_steps):
                current_left = left_steps[i]
 
            if i < len(right_steps):
                current_right = right_steps[i]
 
            after = f"{current_left} = {current_right}"
 
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
 
            steps.append(
                Step(
                    before=f"{sp.latex(final_left)} = {sp.latex(final_right)}",
                    after=f"{sp.latex(new_left)} = {sp.latex(new_right)}",
                    explanation=f"Simplify both sides by {gcd}"
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
                after=f"x = {sp.latex(solution)}",
                explanation=f"Divide both sides by {sp.latex(coef)}"
            )
        )
 
        final_value = solution
        final_latex = sp.latex(final_value)
 
        # FIX 4: verificação chamada via função partilhada
        _check_solution(final_value, final_latex, equation, steps)
 
    return steps

