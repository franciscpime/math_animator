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

        both_have_x = t1.has(x) and t2.has(x)
        both_are_const = not t1.has(x) and not t2.has(x)

        if both_have_x:
            coef = t1.coeff(x) + t2.coeff(x)
            new_term = coef * x
            new_terms = new_terms[:i] + [new_term] + new_terms[i + 2:]
            steps.append(new_terms.copy())
        elif both_are_const:
            new_term = t1 + t2
            new_terms = new_terms[:i] + [new_term] + new_terms[i + 2:]
            steps.append(new_terms.copy())
        else:
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

    rats = [sp.Rational(t) for t in terms]
    denoms = [r.q for r in rats]
    mmc = lcm(*denoms) if len(denoms) > 0 else 1
    nums = [int((r * mmc)) for r in rats]
    gcd_nums = abs(reduce(sp.gcd, [sp.Integer(n) for n in nums])) if nums else 1

    return sp.Rational(gcd_nums, mmc)


# ---------------------------------------------------------------------------
# FIX: stepwise_string_eval — walk the unevaluated AST manually.
#
# O problema antigo: evaluate=False NÃO impede o SymPy de fazer constant
# folding em expressões puramente numéricas (ex: "2*3+1" → 7 imediatamente).
# Portanto preorder_traversal não encontrava Mul/Add nenhum para reduzir,
# e a animação saltava directamente da expressão substituída para a igualdade.
#
# A solução: construir a árvore AST manualmente com evaluate=False e percorrê-
# la em post-order, registando cada redução elementar como um par (antes, depois).
# ---------------------------------------------------------------------------

def _unwrap(expr):
    """Unwrap UnevaluatedExpr recursivamente para que is_number e isinstance
    funcionem correctamente durante a redução do AST."""
    if isinstance(expr, sp.UnevaluatedExpr):
        return _unwrap(expr.args[0])
    if expr.args:
        new_args = [_unwrap(a) for a in expr.args]
        if any(na is not a for na, a in zip(new_args, expr.args)):
            return expr.func(*new_args, evaluate=False)
    return expr


def _collect_mul_add_steps(sym):
    """
    Percorre o AST unevaluated de sym e devolve uma lista de pares
    (before_sym, after_sym), um por cada redução aritmética elementar:
      - 1.ª passagem: todos os Mul cujos args são números → produto
      - 2.ª passagem: todos os Add cujos args são números  → soma
    Post-order: sub-expressões internas são reduzidas antes das externas.
    UnevaluatedExpr é unwrapped antes de processar para que is_number
    funcione correctamente (UnevaluatedExpr(3).is_number é True mas
    impede a avaliação do Mul pai).
    """
    pairs = []

    # Unwrap primeiro para garantir que não há UnevaluatedExpr a bloquear
    sym = _unwrap(sym)

    def _reduce_muls(expr):
        if isinstance(expr, sp.Mul) and all(a.is_number for a in expr.args):
            result = sp.Mul(*expr.args)   # avalia intencionalmente aqui
            pairs.append((expr, result))
            return result
        new_args = []
        changed = False
        for a in expr.args:
            na = _reduce_muls(a)
            if na is not a:
                changed = True
                new_args.append(na)
            else:
                new_args.append(a)
        if changed:
            return expr.func(*new_args, evaluate=False)
        return expr

    def _reduce_adds(expr):
        if isinstance(expr, sp.Add) and all(a.is_number for a in expr.args):
            result = sp.Add(*expr.args)   # avalia intencionalmente aqui
            pairs.append((expr, result))
            return result
        new_args = []
        changed = False
        for a in expr.args:
            na = _reduce_adds(a)
            if na is not a:
                changed = True
                new_args.append(na)
            else:
                new_args.append(a)
        if changed:
            return expr.func(*new_args, evaluate=False)
        return expr

    current = sym
    while True:
        before = current
        current = _reduce_muls(current)
        if current == before:
            break

    while True:
        before = current
        current = _reduce_adds(current)
        if current == before:
            break

    return pairs, current


def stepwise_string_eval(expr_str: str):
    """
    Dada uma string de expressão numérica (após substituição), devolve uma
    lista ordenada de strings LaTeX mostrando cada redução aritmética elementar.
    Primeiro elemento = expressão tal como está; último = valor completamente
    simplificado. Entradas adjacentes diferem por exactamente uma operação.
    """
    expr_str = expr_str.strip()

    cleaned = fix_implicit_mul(expr_str)
    sym = safe_sympify(cleaned)

    # Se já for um número simples, não há passos intermédios a mostrar
    if sym.is_number and not isinstance(sym, (sp.Add, sp.Mul)):
        return [sp.latex(sym)]

    steps_latex = [sp.latex(sym)]

    pairs, final = _collect_mul_add_steps(sym)

    seen = set()
    for before, after in pairs:
        key = (str(before), str(after))
        if key in seen or before == after:
            continue
        seen.add(key)
        steps_latex.append(sp.latex(after))

    # Garantir que o valor final simplificado é o último elemento
    final_latex = sp.latex(sp.simplify(final))
    if steps_latex[-1] != final_latex:
        steps_latex.append(final_latex)

    # Remover entradas consecutivas idênticas
    deduped = [steps_latex[0]]
    for s in steps_latex[1:]:
        if s != deduped[-1]:
            deduped.append(s)

    return deduped


# ---------------------------------------------------------------------------
# FIX: merge_side_steps — substitui o antigo for-loop que emitia Steps mesmo
# quando before == after (sem mudança visível), e que podia saltar transições
# quando os dois lados tinham números de passos diferentes.
# ---------------------------------------------------------------------------

def merge_side_steps(left_steps, right_steps, initial_left, initial_right, steps):
    """
    Mostra primeiro todos os passos do lado esquerdo (mantendo o direito fixo),
    depois todos os passos do lado direito (mantendo o esquerdo fixo).
    Assim cada Step corresponde a exactamente uma operação num único lado,
    evitando que operações de lados diferentes apareçam na mesma transição.
    Só emite Steps quando before != after.
    Devolve (cur_left, cur_right) — as strings LaTeX finais de cada lado.
    """
    cur_left  = initial_left
    cur_right = initial_right

    # Primeiro: todos os passos do lado esquerdo
    for l_after in left_steps:
        before = f"{cur_left} = {cur_right}"
        after  = f"{l_after} = {cur_right}"
        if before != after:
            steps.append(Step(before=before, after=after))
        cur_left = l_after

    # Depois: todos os passos do lado direito
    for r_after in right_steps:
        before = f"{cur_left} = {cur_right}"
        after  = f"{cur_left} = {r_after}"
        if before != after:
            steps.append(Step(before=before, after=after))
        cur_right = r_after

    return cur_left, cur_right


# Função principal que resolve equações lineares passo a passo
def solve_linear(equation: str):

    left, right = parse_equation(equation)

    steps = []

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

    # Mostrar a explicação ANTES da transição — o renderer executa
    # sempre transição→explicação, por isso a explicação tem de vir
    # num Step próprio antes da mudança visual
    steps.append(
        Step(
            before=equation,
            after=equation,
            explanation="Organizar termos"
        )
    )
    steps.append(
        Step(
            before=equation,
            after=new_eq,
        )
    )

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
        # Mostrar apenas a transição — sem mensagem de divisão aqui,
        # pois estamos apenas a somar constantes (ex: 8-20 → -12),
        # não a dividir. A mensagem de divisão fica reservada para o
        # CASO 2 onde se divide ambos os lados pelo coeficiente de x.
        steps.append(
            Step(
                before=build_equation(current_vars, current_consts),
                after=build_equation(current_vars, new_consts),
            )
        )
        current_consts = new_consts

    # Final solve
    final_left = current_vars[0]
    final_right = current_consts[0]

    coef = final_left.coeff(x)
    const = final_right

    # FIX 10: usa sp.Rational directamente em vez de int() para o GCD
    def safe_gcd(a, b):
        return sp.gcd(sp.Rational(a), sp.Rational(b))

    def _sympy_stepwise(sym):
        """
        Recebe um objecto SymPy (unevaluated) e devolve lista de strings LaTeX,
        uma por cada redução aritmética elementar.
        A cada passo substitui o sub-nó reduzido na expressão completa (xreplace)
        para que o LaTeX mostre sempre a expressão inteira e não só o sub-nó.
        """
        sym = _unwrap(sym)

        # Se já é um número simples, nada a mostrar
        if sym.is_number and not isinstance(sym, (sp.Add, sp.Mul)):
            return [sp.latex(sym)]

        result = [sp.latex(sym, order='none')]
        current = sym
        _frac_steps = []  # passos intermédios de int×Rational

        def _step_one_mul(expr):
            """Encontra e reduz o primeiro Mul numérico, devolve (expr_nova, before, after).
            Caso especial: Mul(inteiro, Rational) gera passo intermédio em _frac_steps:
              10*(-3/7) -> frac{10*(-3)}{7} -> -30/7
            """
            if isinstance(expr, sp.Mul) and all(a.is_number for a in expr.args):
                inteiros  = [a for a in expr.args if isinstance(a, sp.Integer)]
                racionais = [a for a in expr.args if isinstance(a, sp.Rational) and a.q != 1]
                if len(expr.args) == 2 and inteiros and racionais:
                    inteiro  = inteiros[0]
                    racional = racionais[0]
                    num_expr = sp.Mul(inteiro, sp.Integer(racional.p), evaluate=False)
                    intermed = r'\frac{' + sp.latex(num_expr) + r'}{' + str(racional.q) + r'}'
                    resultado = sp.Rational(int(inteiro) * racional.p, racional.q)
                    _frac_steps.append((sp.latex(expr, order='none'), intermed, sp.latex(resultado)))
                    return resultado, expr, resultado
                return sp.Mul(*expr.args), expr, sp.Mul(*expr.args)
            for i, a in enumerate(expr.args):
                result_inner = _step_one_mul(a)
                if result_inner is not None:
                    reduced, b, af = result_inner
                    new_args = list(expr.args)
                    new_args[i] = reduced
                    return expr.func(*new_args, evaluate=False), b, af
            return None

        def _step_one_add(expr):
            """Encontra e reduz o primeiro Add numérico, devolve (expr_nova, before, after)."""
            if isinstance(expr, sp.Add) and all(a.is_number for a in expr.args):
                return sp.Add(*expr.args), expr, sp.Add(*expr.args)
            for i, a in enumerate(expr.args):
                result_inner = _step_one_add(a)
                if result_inner is not None:
                    reduced, b, af = result_inner
                    new_args = list(expr.args)
                    new_args[i] = reduced
                    return expr.func(*new_args, evaluate=False), b, af
            return None

        def _fix_plus_minus(s):
            """Corrigir sinais duplos após substituição de frações.
            '+ -' -> '-'  (positivo + negativo = negativo)
            '- -' -> '+'  (negativo + negativo = positivo)
            """
            import re as _re
            s = _re.sub(r'-\s*-\s*', '+ ', s)   # -- -> +
            s = _re.sub(r'\+\s*-\s*', '- ', s)  # +- -> -
            return s

        # Reduzir Muls um a um — para int×Rational usar os passos de _frac_steps
        while True:
            _frac_steps.clear()
            r = _step_one_mul(current)
            if r is None:
                break
            new_expr, sub_antes, sub_depois = r
            if _frac_steps:
                orig_l, intermed_l, final_l = _frac_steps[0]
                s1 = _fix_plus_minus(result[-1].replace(orig_l, intermed_l, 1))
                if s1 != result[-1]:
                    result.append(s1)
                s2 = _fix_plus_minus(s1.replace(intermed_l, final_l, 1))
                if s2 != s1:
                    result.append(s2)
            else:
                latex_antes  = sp.latex(sub_antes,  order='none')
                latex_depois = sp.latex(sub_depois, order='none')
                nova_str = _fix_plus_minus(result[-1].replace(latex_antes, latex_depois, 1))
                if nova_str != result[-1]:
                    result.append(nova_str)
            current = new_expr

        # Reduzir Adds um a um
        # Caso especial: Add(inteiro, Rational não inteiro) → passo MMC intermédio
        while True:
            r = _step_one_add(current)
            if r is None:
                break
            new_expr, sub_antes, sub_depois = r
            # Verificar se o Add tem inteiro + racional: gerar passo MMC
            if isinstance(sub_antes, sp.Add):
                inteiros_add  = [a for a in sub_antes.args if isinstance(a, sp.Integer)]
                racionais_add = [a for a in sub_antes.args
                                 if isinstance(a, sp.Rational) and not isinstance(a, sp.Integer) and a.q != 1]
                if inteiros_add and racionais_add:
                    den = racionais_add[0].q
                    inteiro_val = int(inteiros_add[0])
                    # Construir rac{n*den}{den} sem simplificar (sp.Rational simplificaria)
                    frac_str = r'\frac{' + str(inteiro_val * den) + r'}{' + str(den) + r'}'
                    latex_inteiro = sp.latex(sp.Integer(inteiro_val))
                    # Substituir apenas o inteiro isolado (não parte de outro número)
                    import re as _re
                    intermed_str = _re.sub(
                        r'(?<![0-9])' + _re.escape(latex_inteiro) + r'(?![0-9])',
                        lambda m: frac_str, result[-1], count=1
                    )
                    intermed_str = _fix_plus_minus(intermed_str)
                    if intermed_str != result[-1]:
                        result.append(intermed_str)
            latex_antes  = sp.latex(sub_antes,  order='none')
            latex_depois = sp.latex(sub_depois, order='none')
            nova_str = _fix_plus_minus(result[-1].replace(latex_antes, latex_depois, 1))
            current = new_expr
            if nova_str != result[-1]:
                result.append(nova_str)

        # Garantir valor final simplificado
        final_latex = sp.latex(sp.simplify(current), order='none')
        if result[-1] != final_latex:
            result.append(final_latex)

        # Deduplicar entradas consecutivas iguais
        deduped = [result[0]]
        for s in result[1:]:
            if s != deduped[-1]:
                deduped.append(s)
        return deduped

    def _check_solution(final_value, final_latex, equation, steps):
        """Verificação da solução com passos intermédios completos."""

        left_str, right_str = equation.split("=")
        left_sym  = sp.sympify(normalize_expression(left_str.strip()), evaluate=False)
        right_sym = sp.sympify(normalize_expression(right_str.strip()), evaluate=False)

        # Equação original em LaTeX — preservar ordem original dos termos
        # usando sp.printing.latex com order='none' para não reordenar
        equation_latex = f"{sp.latex(left_sym, order='none')} = {sp.latex(right_sym, order='none')}"

        steps.append(
            Step(
                before=equation_latex,
                after=equation_latex,
                explanation="Vamos verificar!"
            )
        )
        steps.append(
            Step(
                before=equation_latex,
                after=equation_latex,
                explanation=f"Agora vamos substituir x por {final_latex}"
            )
        )

        # FIX 4: substituir x na string LaTeX para preservar a ordem original
        # xreplace reordena os args do Add após substituição numérica
        left_latex_orig  = sp.latex(left_sym, order='none')
        right_latex_orig = sp.latex(right_sym, order='none')
        val_latex = sp.latex(final_value)

        def _sub_x_in_latex(latex_str, val):
            import re as _re
            return _re.sub(r'(?<![a-zA-Z])x(?![a-zA-Z])', lambda m: '(' + val + ')', latex_str)

        left_subst_str  = _sub_x_in_latex(left_latex_orig, val_latex)
        right_subst_str = _sub_x_in_latex(right_latex_orig, val_latex)

        substituted_display = f"{left_subst_str} = {right_subst_str}"
        steps.append(
            Step(
                before=equation_latex,
                after=substituted_display,
            )
        )

        # Para os passos aritméticos usamos o AST SymPy
        left_evaled  = left_sym.xreplace({x: sp.UnevaluatedExpr(final_value)})
        right_evaled = right_sym.xreplace({x: sp.UnevaluatedExpr(final_value)})

        left_steps  = _sympy_stepwise(left_evaled)
        right_steps = _sympy_stepwise(right_evaled)

        # Substituir o primeiro elemento pela string de ordem original
        if left_steps:
            left_steps[0] = left_subst_str
        if right_steps:
            right_steps[0] = right_subst_str

        # Intercalar os dois lados em Steps, sem emitir transições nulas
        cur_left, cur_right = merge_side_steps(
            left_steps,
            right_steps,
            initial_left=left_subst_str,
            initial_right=right_subst_str,
            steps=steps,
        )

        # Verificação final — usar final_value directamente em vez de parsear
        # a string LaTeX de cur_left/cur_right, que pode conter \frac, \cdot, etc.
        # que o sp.sympify não consegue parsear.
        left_orig  = sp.sympify(normalize_expression(left_str.strip()), evaluate=False)
        right_orig = sp.sympify(normalize_expression(right_str.strip()), evaluate=False)
        left_final  = left_orig.subs(x, final_value)
        right_final = right_orig.subs(x, final_value)

        is_true = sp.simplify(left_final - right_final) == 0

        explanation = "The solution is correct!" if is_true else "The solution does not satisfy the equation."

        steps.append(
            Step(
                before=f"{cur_left} = {cur_right}",
                after=f"{cur_left} = {cur_right}",
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

        final_value = const
        final_latex = sp.latex(final_value)
        _check_solution(final_value, final_latex, equation, steps)

    # CASO 2: precisa dividir (ex: 30x=5 → 6x=1 → x=1/6)
    else:
        solution = sp.Rational(const, coef)

        # Calcular o MDC entre constante e coeficiente para simplificar gradualmente.
        # Ex: 30x=5 — MDC(5,30)=5 → dividir por 5 → 6x=1 → depois dividir por 6
        divisor_intermedio = safe_gcd(abs(const), abs(coef))

        if divisor_intermedio > 1 and divisor_intermedio != abs(coef):
            # Passo intermédio: simplificar pelo MDC primeiro
            coef_simplificado  = sp.Rational(coef,  divisor_intermedio)
            const_simplificado = sp.Rational(const, divisor_intermedio)
            left_intermedio  = coef_simplificado * x
            right_intermedio = const_simplificado

            steps.append(Step(
                before=f"{sp.latex(final_left)} = {sp.latex(final_right)}",
                after=f"{sp.latex(final_left)} = {sp.latex(final_right)}",
                explanation=f"Dividir ambos os lados por {sp.latex(divisor_intermedio)}"
            ))
            steps.append(Step(
                before=f"{sp.latex(final_left)} = {sp.latex(final_right)}",
                after=f"{sp.latex(left_intermedio)} = {sp.latex(right_intermedio)}",
            ))
            steps.append(Step(
                before=f"{sp.latex(left_intermedio)} = {sp.latex(right_intermedio)}",
                after=f"{sp.latex(left_intermedio)} = {sp.latex(right_intermedio)}",
                explanation=f"Dividir ambos os lados por {sp.latex(coef_simplificado)}"
            ))
            steps.append(Step(
                before=f"{sp.latex(left_intermedio)} = {sp.latex(right_intermedio)}",
                after=f"x = {sp.latex(solution)}",
            ))
        else:
            # Divisão directa sem passo intermédio
            steps.append(Step(
                before=f"{sp.latex(final_left)} = {sp.latex(final_right)}",
                after=f"{sp.latex(final_left)} = {sp.latex(final_right)}",
                explanation=f"Dividir ambos os lados por {sp.latex(coef)}"
            ))
            steps.append(Step(
                before=f"{sp.latex(final_left)} = {sp.latex(final_right)}",
                after=f"x = {sp.latex(solution)}",
            ))

        final_value = solution
        final_latex = sp.latex(final_value)

        _check_solution(final_value, final_latex, equation, steps)

    return steps



