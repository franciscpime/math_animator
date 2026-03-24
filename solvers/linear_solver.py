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

    def _sympy_stepwise(subst_str, sym_evaled, final_value):
        """
        Gera lista de tuplos (latex_str, explicacao) com cada passo aritmético.
        explicacao=None para transições, string para mensagens pedagógicas.
        """
        import re as _re

        def _fix_pm(s):
            s = _re.sub(r'-\s*-\s*', '+ ', s)
            s = _re.sub(r'\+\s*-\s*', '- ', s)
            return s

        def _add_ops(s):
            s = _re.sub(r'(\d)\s*(\\frac)', r'\1 + \2', s)
            s = _re.sub(r'(\})\s*(\\frac)', r'\1 + \2', s)
            return s

        result = [(subst_str, None)]

        if not (isinstance(final_value, sp.Rational) and final_value.q != 1):
            final_str = sp.latex(sp.simplify(sym_evaled))
            if final_str != subst_str:
                result.append((final_str, None))
            return result

        val_num = final_value.p
        val_den = final_value.q
        val_latex_s = sp.latex(final_value)
        val_esc = _re.escape(val_latex_s)
        val_num_str = '(' + str(val_num) + ')' if val_num < 0 else str(val_num)

        # Passo 1: cada "coef (val)" -> frac{coef*num}{den} -> frac{resultado}{den}
        pattern = r'(-?\s*\d+)\s*\(' + val_esc + r'\)'
        cur = subst_str
        while True:
            m = _re.search(pattern, cur)
            if not m: break
            coef = int(m.group(1).replace(' ', ''))
            produto = coef * val_num
            c_str = ('(' + str(coef) + ')') if coef < 0 else str(coef)
            frac_i = r'\frac{' + c_str + r' \cdot ' + val_num_str + r'}{' + str(val_den) + r'}'
            s1 = _fix_pm(_add_ops(cur[:m.start()] + frac_i + cur[m.end():]))
            if s1 != cur: result.append((s1, None)); cur = s1
            frac_f = sp.latex(sp.Rational(produto, val_den))
            s2 = _fix_pm(_add_ops(cur.replace(frac_i, frac_f, 1)))
            if s2 != cur: result.append((s2, None)); cur = s2

        final_val = sp.simplify(sym_evaled)
        final_str = sp.latex(final_val)

        # Encontrar inteiros soltos fora de chaves
        def find_isolated_ints(s):
            found = []; depth = 0; i = 0
            while i < len(s):
                c = s[i]
                if c == '{': depth += 1; i += 1; continue
                elif c == '}': depth -= 1; i += 1; continue
                if depth > 0: i += 1; continue
                if c.isdigit():
                    if i > 0 and (s[i-1].isdigit() or s[i-1] == '{'): i += 1; continue
                    j = i
                    while j < len(s) and s[j].isdigit(): j += 1
                    if j < len(s) and s[j] in '/}': i = j; continue
                    try: n = int(s[i:j])
                    except: i += 1; continue
                    if n != 0: found.append((i, j, n))
                    i = j
                else: i += 1
            return found

        # Passo 2: inteiros -> frac com denominador comum (FIX 4,5,6,7)
        if find_isolated_ints(cur):
            result.append((cur, f'Reduzir ao mesmo denominador ({val_den})'))
            for _ in range(30):
                iso = find_isolated_ints(cur)
                if not iso: break
                start, end, n = iso[0]
                frac = r'\frac{' + str(n * val_den) + r'}{' + str(val_den) + r'}'
                novo = _fix_pm(_add_ops(cur[:start] + frac + cur[end:]))
                if novo == cur: break
                result.append((novo, None)); cur = novo
                if cur == final_str: break

        # Passo 3: somar fracoes par a par (FIX 5,7)
        fp = r'(\\frac\{(-?\d+)\}\{(\d+)\})\s*([+-])\s*(\\frac\{(-?\d+)\}\{(\d+)\})'
        for _ in range(30):
            m = _re.search(fp, cur)
            if not m: break
            na, da = int(m.group(2)), int(m.group(3))
            op = m.group(4)
            nb, db = int(m.group(6)), int(m.group(7))
            if da != db: break
            soma_n = na + (nb if op == '+' else -nb)
            frac_soma = '0' if soma_n == 0 else sp.latex(sp.Rational(soma_n, da))
            novo = _fix_pm(_add_ops(cur[:m.start()] + frac_soma + cur[m.end():]))
            if novo == cur: break
            result.append((novo, None)); cur = novo
            if cur == final_str: break

        if cur != final_str:
            result.append((final_str, None))

        # Deduplicar mantendo mensagens
        deduped = [result[0]]
        for item in result[1:]:
            s, e = item
            ps = deduped[-1][0]
            if s != ps or e is not None:
                deduped.append(item)
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

        # Substituir x termo a termo para preservar Muls separados.
        # left_sym.xreplace() combinaria -30x + 40x em 10x antes de substituir.
        def _subst_terms(sym, val):
            """Substitui x em cada termo de um Add individualmente,
            reconstruindo sem avaliar para preservar cada Mul separado."""
            if isinstance(sym, sp.Add):
                new_terms = [t.xreplace({x: sp.UnevaluatedExpr(val)}) for t in sym.args]
                return sp.Add(*new_terms, evaluate=False)
            return sym.xreplace({x: sp.UnevaluatedExpr(val)})

        left_evaled  = _subst_terms(left_sym,  final_value)
        right_evaled = _subst_terms(right_sym, final_value)

          # Gerar passos — lista de (latex_str, explicacao_ou_None)
        left_tuples  = _sympy_stepwise(left_subst_str,  left_evaled,  final_value)
        right_tuples = _sympy_stepwise(right_subst_str, right_evaled, final_value)

        def _extract(tuples):
            sl, ex = [], {}
            for i, (s, e) in enumerate(tuples):
                sl.append(s)
                if e: ex[i] = e
            return sl, ex

        left_steps,  left_expls  = _extract(left_tuples)
        right_steps, right_expls = _extract(right_tuples)

        # Mostrar passos: primeiro lado esquerdo, depois lado direito
        cur_left  = left_subst_str
        cur_right = right_subst_str
        for i, l_after in enumerate(left_steps):
            expl = left_expls.get(i)
            before = f'{cur_left} = {cur_right}'
            if expl:
                steps.append(Step(before=before, after=before, explanation=expl))
            after = f'{l_after} = {cur_right}'
            if before != after:
                steps.append(Step(before=before, after=after))
            cur_left = l_after
        for i, r_after in enumerate(right_steps):
            expl = right_expls.get(i)
            before = f'{cur_left} = {cur_right}'
            if expl:
                steps.append(Step(before=before, after=before, explanation=expl))
            after = f'{cur_left} = {r_after}'
            if before != after:
                steps.append(Step(before=before, after=after))
            cur_right = r_after

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


