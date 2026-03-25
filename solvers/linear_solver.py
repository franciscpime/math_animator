import sympy as sp
from sympy import lcm
import re
from functools import reduce
from models.step import Step
from parser.equation_parser import parse_equation, normalize_expression, safe_sympify, fix_implicit_mul, detect_decimals, detect_raw_fractions
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


def _decimal_str(solution):
    """
    Converte solução racional para string decimal arredondada a 3 casas.
    Devolve None se for inteiro (não precisa de decimal).
    """
    if isinstance(solution, sp.Integer) or (isinstance(solution, sp.Rational) and solution.q == 1):
        return None
    val = float(solution)
    rounded = round(val, 3)
    if rounded == int(rounded):
        return str(int(rounded))
    s = f'{rounded:.3f}'.rstrip('0').rstrip('.')
    return s


def _fraction_simplification_steps(num_str, den_str):
    """
    Gera passos de simplificação de fração não reduzida.
    Ex: '4', '8' -> ['\frac{4}{8}', '\frac{1}{2}']
    """
    import math
    num, den = int(num_str), int(den_str)
    steps = [r'\frac{' + num_str + r'}{' + den_str + r'}']
    g = math.gcd(abs(num), den)
    if g > 1:
        steps.append(r'\frac{' + str(num // g) + r'}{' + str(den // g) + r'}')
    return steps


def _decimal_simplification_steps(decimal_str):
    """
    Gera passos de conversão de decimal para fração.
    Ex: '0.5' -> ['0.5', '\frac{5}{10}', '\frac{1}{2}']
         '0.9' -> ['0.9', '\frac{9}{10}']
    """
    import math
    s = decimal_str.replace(',', '.')
    if '.' not in s:
        return [s]
    is_neg = s.startswith('-')
    s_abs = s.lstrip('-')
    dec_part = s_abs.split('.')[1]
    n_dec = len(dec_part)
    den = 10 ** n_dec
    num = int(s_abs.replace('.', ''))
    if is_neg:
        num = -num
    frac_unreduced = r'\frac{' + str(num) + r'}{' + str(den) + r'}'
    steps = [decimal_str, frac_unreduced]
    g = math.gcd(abs(num), den)
    if g > 1:
        frac_reduced = r'\frac{' + str(num // g) + r'}{' + str(den // g) + r'}'
        steps.append(frac_reduced)
    return steps


# Função principal que resolve equações lineares passo a passo
def solve_linear(equation: str):

    left, right = parse_equation(equation)

    steps = []

    # Detectar e mostrar passos de simplificação de decimais e frações
    # antes de começar a resolver
    _raw_decimals = detect_decimals(equation)
    _raw_fractions = detect_raw_fractions(equation)

    # Converter a/b para \frac{a}{b} na equação de display desde o início
    import re as _re_disp
    def _eq_to_latex_display(eq):
        return _re_disp.sub(
            r'(?<!\\\\)(-?\d+)/(\d+)',
            lambda m: r'\frac{' + m.group(1) + r'}{' + m.group(2) + r'}',
            eq
        )

    equation_display = _eq_to_latex_display(equation)

    # Passos: decimal -> fração não reduzida -> fração reduzida
    current_eq_display = equation_display
    for dec_str, dec_val in _raw_decimals:
        dec_steps = _decimal_simplification_steps(dec_str)
        # dec_steps: ['0.5', '\frac{5}{10}', '\frac{1}{2}']
        # Substituir passo a passo: dec_steps[i-1] -> dec_steps[i]
        for i in range(1, len(dec_steps)):
            before_eq = current_eq_display
            src = dec_steps[i-1]  # o que está na string actual
            dst = dec_steps[i]    # o que vai ficar
            after_eq = before_eq.replace(src, dst, 1)
            if before_eq != after_eq:
                steps.append(Step(
                    before=before_eq,
                    after=after_eq,
                    explanation='Converter decimal para fração' if i == 1 else 'Simplificar fração'
                ))
                current_eq_display = after_eq

    # Passos: fração não reduzida -> fração reduzida
    for num_s, den_s, frac_val in _raw_fractions:
        frac_steps = _fraction_simplification_steps(num_s, den_s)
        frac_orig_str = num_s + '/' + den_s  # como aparece na equação escrita
        # Passo 1: mostrar a/b -> \frac{a}{b}
        if frac_orig_str in current_eq_display:
            eq_with_frac = current_eq_display.replace(frac_orig_str, frac_steps[0], 1)
            if eq_with_frac != current_eq_display:
                steps.append(Step(before=current_eq_display, after=eq_with_frac))
                current_eq_display = eq_with_frac
        # Passo 2: simplificar \frac{a}{b} -> \frac{a/g}{b/g} se possível
        if len(frac_steps) > 1:
            for i in range(1, len(frac_steps)):
                before_eq = current_eq_display
                after_eq  = before_eq.replace(frac_steps[i-1], frac_steps[i], 1)
                if before_eq != after_eq:
                    steps.append(Step(
                        before=before_eq,
                        after=after_eq,
                        explanation='Simplificar fração'
                    ))
                    current_eq_display = after_eq

    # A partir daqui usar a equação normalizada (com racionais)
    # O parse_equation já normalizou left e right
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
            before=equation_display,
            after=equation_display,
            explanation="Organizar termos"
        )
    )
    steps.append(
        Step(
            before=equation_display,
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
            # Valor inteiro — mostrar multiplicações e somas passo a passo
            cur = subst_str
            val_esc_int = _re.escape(sp.latex(final_value))
            pattern_int = r'(-?\s*\d+)\s*\(' + val_esc_int + r'\)'
            # Passo 1: resolver cada coef*(inteiro)
            while True:
                m = _re.search(pattern_int, cur)
                if not m: break
                coef = int(m.group(1).replace(' ', ''))
                produto = coef * int(final_value)
                # Se o sinal '-' estava na string e o resultado é positivo,
                # adicionar '+' para não ficar '10 20' sem operador
                prod_str = str(produto)
                before_m = cur[:m.start()]
                after_m  = cur[m.end():]
                # Adicionar operador explícito se necessário
                if before_m.rstrip() and before_m.rstrip()[-1].isdigit():
                    prod_str = ('+ ' if produto >= 0 else '- ') + str(abs(produto))
                s1 = _fix_pm(before_m + prod_str + after_m)
                if s1 == cur: break
                result.append((s1, None)); cur = s1
            # Passo 2: somar termos numéricos restantes
            final_str = sp.latex(sp.simplify(sym_evaled))
            # Tentar somar par a par usando Add simples
            sum_pattern = r'(-?\d+)\s*([+-])\s*(\d+)'
            for _ in range(10):
                m = _re.search(sum_pattern, cur)
                if not m: break
                a = int(m.group(1))
                op = m.group(2)
                b = int(m.group(3))
                soma = a + b if op == '+' else a - b
                novo = _fix_pm(cur[:m.start()] + str(soma) + cur[m.end():])
                if novo == cur: break
                result.append((novo, None)); cur = novo
                if cur == final_str: break
            if cur != final_str:
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
            # Fração não simplificada: mostrar produto/val_den antes de simplificar
            frac_unsimplified = r'\frac{' + str(produto) + r'}{' + str(val_den) + r'}'
            frac_simplified    = sp.latex(sp.Rational(produto, val_den))
            # Passo 2a: substituir frac_i por frac_unsimplified (resultado da mult)
            s2_raw = cur.replace(frac_i, frac_unsimplified, 1)
            s2 = _fix_pm(_add_ops(s2_raw))
            s2 = _re.sub(r'(\d)\s+(\d)', r'\1 + \2', s2)
            s2 = _fix_pm(s2)
            if s2 != cur: result.append((s2, None)); cur = s2
            # Passo 2b: simplificar a fração se o resultado for diferente
            if frac_simplified != frac_unsimplified:
                s3 = _fix_pm(_add_ops(cur.replace(frac_unsimplified, frac_simplified, 1)))
                s3 = _re.sub(r'(\d)\s+(\d)', r'\1 + \2', s3)
                s3 = _fix_pm(s3)
                if s3 != cur: result.append((s3, None)); cur = s3

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

        # Passo 2: inteiros -> frac com denominador comum
        # Calcular o MMC dos denominadores presentes na string actual
        has_frac = r'\frac' in cur
        if has_frac and find_isolated_ints(cur):
            # Extrair todos os denominadores presentes
            dens_presentes = [int(d) for d in _re.findall(r'\\frac\{[^}]+\}\{(\d+)\}', cur)]
            if dens_presentes:
                from math import lcm as _lcm
                from functools import reduce as _reduce
                den_comum = _reduce(_lcm, dens_presentes)
            else:
                den_comum = val_den
            result.append((cur, f'Reduzir ao mesmo denominador ({den_comum})'))
            for _ in range(30):
                iso = find_isolated_ints(cur)
                if not iso: break
                start, end, n = iso[0]
                frac = r'\frac{' + str(n * den_comum) + r'}{' + str(den_comum) + r'}'
                novo = _fix_pm(_add_ops(cur[:start] + frac + cur[end:]))
                if novo == cur: break
                result.append((novo, None)); cur = novo
                if cur == final_str: break

        # Passo 3: somar fracoes par a par
        # Se denominadores forem diferentes, converter primeiro para denominador comum
        fp = r'(\\frac\{(-?\d+)\}\{(\d+)\})\s*([+-])\s*(\\frac\{(-?\d+)\}\{(\d+)\})'
        from math import lcm as _lcm2
        for _ in range(30):
            m = _re.search(fp, cur)
            if not m: break
            na, da = int(m.group(2)), int(m.group(3))
            op = m.group(4)
            nb, db = int(m.group(6)), int(m.group(7))
            # Verificar se há '-' antes do primeiro rac (sinal externo)
            prefix = cur[:m.start()].rstrip()
            if prefix.endswith('-'):
                na = -abs(na)
                # Remover o '-' do prefix para não ficar duplicado
                cur = cur[:len(prefix)-1] + cur[len(prefix):]
                m = _re.search(fp, cur)  # re-search após remoção
                if not m: break
                na, da = int(m.group(2)), int(m.group(3))
                na = -abs(na)
                op = m.group(4)
                nb, db = int(m.group(6)), int(m.group(7))
            if da != db:
                # Denominadores diferentes: converter para denominador comum
                dc = _lcm2(da, db)
                na_new = na * (dc // da)
                nb_new = nb * (dc // db)
                frac_a_new = r'\frac{' + str(na_new) + r'}{' + str(dc) + r'}'
                frac_b_new = r'\frac{' + str(nb_new) + r'}{' + str(dc) + r'}'
                novo = _fix_pm(_add_ops(
                    cur[:m.start()] + frac_a_new + ' ' + op + ' ' + frac_b_new + cur[m.end():]
                ))
                if novo == cur: break
                result.append((novo, None)); cur = novo
                continue
            soma_n = na + (nb if op == '+' else -nb)
            # Passo intermédio: juntar numeradores antes de calcular
            # ex: rac{12}{3} - rac{23}{3} -> rac{12 - 23}{3} -> rac{-11}{3}
            nb_signed = nb if op == '+' else -nb
            if op == '+':
                num_expr_str = str(na) + ' + ' + str(nb)
            else:
                num_expr_str = str(na) + ' - ' + str(nb)
            frac_joined = r'\frac{' + num_expr_str + r'}{' + str(da) + r'}'
            novo_joined = _fix_pm(_add_ops((cur[:m.start()] + frac_joined + cur[m.end():]).strip()))
            if novo_joined != cur:
                result.append((novo_joined, None)); cur = novo_joined
            # Agora calcular o resultado
            frac_soma = '0' if soma_n == 0 else sp.latex(sp.Rational(soma_n, da))
            novo = _fix_pm(_add_ops((cur.replace(frac_joined, frac_soma, 1)).strip()))
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

        # Construir equation_latex a partir da equação original,
        # convertendo a/b para \frac{a}{b} mas preservando a ordem
        import re as _re2
        def _fracs_to_latex(s):
            # Substituir a/b por \frac{a}{b}, ignorando casos como x/y onde há letras
            return _re2.sub(r'(-?\d+)/(\d+)', lambda m: r'\frac{' + m.group(1) + r'}{' + m.group(2) + r'}', s)
        left_latex_for_display  = _fracs_to_latex(left_str.strip())
        right_latex_for_display = _fracs_to_latex(right_str.strip())
        # Normalizar multiplicação implícita para LaTeX (ex: 5/4x -> \frac{5}{4}x)
        left_latex_for_display  = _re2.sub(r'(\\frac\{\d+\}\{\d+\})(\s*)([a-zA-Z])', r'\1 \3', left_latex_for_display)
        right_latex_for_display = _re2.sub(r'(\\frac\{\d+\}\{\d+\})(\s*)([a-zA-Z])', r'\1 \3', right_latex_for_display)
        equation_latex = f"{left_latex_for_display} = {right_latex_for_display}"

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
        left_latex_orig  = left_latex_for_display
        right_latex_orig = right_latex_for_display
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
        # Mostrar aproximação decimal se a solução for fracção
        _dec = _decimal_str(final_value)
        if _dec:
            sol_latex = f'x = {final_latex}'
            steps.append(Step(
                before=sol_latex,
                after=sol_latex,
                explanation=f'x = {final_latex} \\approx {_dec}'
            ))
        _check_solution(final_value, final_latex, equation, steps)

    # CASO 2: precisa dividir ambos os lados pelo coeficiente de x
    else:
        solution = sp.Rational(const, coef)

        # Para coeficientes inteiros, verificar se há passo intermédio útil
        # (ex: 4x=-10 -> ÷2 -> 2x=-5 -> ÷2 -> x=-5/2)
        # Para coeficientes racionais (ex: 5/4), dividir directamente
        if isinstance(coef, sp.Integer) or (isinstance(coef, sp.Rational) and coef.q == 1):
            # Coeficiente inteiro — verificar passo intermédio
            coef_int = int(coef)
            const_rat = sp.Rational(const)
            divisor_intermedio = safe_gcd(abs(const_rat), abs(coef))
            coef_simpl  = coef / divisor_intermedio
            const_simpl = const_rat / divisor_intermedio
            mostrar_intermedio = (
                divisor_intermedio > 1
                and divisor_intermedio != abs(coef)
                and isinstance(coef_simpl, sp.Integer) or (isinstance(coef_simpl, sp.Rational) and coef_simpl.q == 1)
                and isinstance(const_simpl, sp.Integer) or (isinstance(const_simpl, sp.Rational) and const_simpl.q == 1)
                and coef_simpl != 1
            )
            if mostrar_intermedio:
                left_int  = coef_simpl * x
                right_int = const_simpl
                steps.append(Step(
                    before=f"{sp.latex(final_left)} = {sp.latex(final_right)}",
                    after=f"{sp.latex(final_left)} = {sp.latex(final_right)}",
                    explanation=f"Dividir ambos os lados por {sp.latex(divisor_intermedio)}"
                ))
                steps.append(Step(
                    before=f"{sp.latex(final_left)} = {sp.latex(final_right)}",
                    after=f"{sp.latex(left_int)} = {sp.latex(right_int)}",
                ))
                steps.append(Step(
                    before=f"{sp.latex(left_int)} = {sp.latex(right_int)}",
                    after=f"{sp.latex(left_int)} = {sp.latex(right_int)}",
                    explanation=f"Dividir ambos os lados por {sp.latex(coef_simpl)}"
                ))
                steps.append(Step(
                    before=f"{sp.latex(left_int)} = {sp.latex(right_int)}",
                    after=f"x = {sp.latex(solution)}",
                ))
            else:
                steps.append(Step(
                    before=f"{sp.latex(final_left)} = {sp.latex(final_right)}",
                    after=f"{sp.latex(final_left)} = {sp.latex(final_right)}",
                    explanation=f"Dividir ambos os lados por {sp.latex(coef)}"
                ))
                steps.append(Step(
                    before=f"{sp.latex(final_left)} = {sp.latex(final_right)}",
                    after=f"x = {sp.latex(solution)}",
                ))
        else:
            # Coeficiente racional (ex: 5/4 x = 55/6) — dividir directamente
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

        # Mostrar aproximação decimal se a solução for fracção
        _dec = _decimal_str(final_value)
        if _dec:
            sol_latex = f'x = {final_latex}'
            steps.append(Step(
                before=sol_latex,
                after=sol_latex,
                explanation=f'x = {final_latex} \\approx {_dec}'
            ))

        _check_solution(final_value, final_latex, equation, steps)

    return steps


