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


# ─────────────────────────────────────────────────────────────────────────────
# CORREÇÃO 1 — combine_terms_stepwise
#
# Antes: combinava directamente coef1·x + coef2·x sem mostrar o passo de
#        igualação de denominadores quando os coeficientes são frações.
#
# Agora: quando ambos os coeficientes são racionais com denominadores
#        diferentes, emite passos intermédios:
#          1) converter cada coef para o denominador comum → (a/d)x ± (b/d)x
#          2) juntar numeradores             → ((a±b)/d)x  [ainda não avaliado]
#          3) simplificar a fração resultado → (c/d)x
#        Cada passo gera uma entrada separada na lista devolvida por
#        combine_terms_stepwise, que é depois transformada em Steps pelo caller.
# ─────────────────────────────────────────────────────────────────────────────

def _coef_rational(term):
    """Devolve o coeficiente racional de um termo em x (pode ser Rational ou Integer)."""
    return sp.Rational(term.coeff(x))


def _render_x_term(coef):
    """Devolve a expressão SymPy coef*x para um coeficiente racional."""
    return coef * x


def combine_terms_stepwise(terms):
    """
    Combina termos semelhantes passo a passo, seguindo a lógica:

    Para termos com x (qualquer número):
      1. Converter TODOS para o denominador comum de uma vez
         → emite __latex__ com todos os termos convertidos
      2. Juntar todos os numeradores: (a + b + c)/d · x
         → emite __latex__ com a expressão não avaliada
      3. Emite lista SymPy com o resultado final

    Para constantes (qualquer número):
      1. Converter TODOS para denominador comum
      2. Juntar numeradores
      3. Resultado

    Devolve lista de entradas:
      - lista de termos SymPy  (estado actualizado)
      - ("__latex__", latex_str, sympy_terms_list)  (passo visual intermédio)
    """
    from math import lcm as _mlcm
    from functools import reduce as _freduce

    new_terms = terms.copy()
    steps = []

    def _frac_x_latex(num, den):
        """LaTeX de (num/den)·x forçando o denominador visível."""
        if den == 1:
            if num == 1:  return "x"
            if num == -1: return "- x"
            return f"{num} x"
        if num == 1:  return fr"\frac{{x}}{{{den}}}"
        if num == -1: return fr"- \frac{{x}}{{{den}}}"
        if num < 0:   return fr"- \frac{{{abs(num)} x}}{{{den}}}"
        return fr"\frac{{{num} x}}{{{den}}}"

    def _frac_latex(num, den):
        """LaTeX de num/den forçando o denominador visível."""
        if den == 1: return str(num)
        if num < 0:  return fr"- \frac{{{abs(num)}}}{{{den}}}"
        return fr"\frac{{{num}}}{{{den}}}"

    def _join_latex(parts):
        """Junta strings LaTeX com +/- explícitos e espaços correctos."""
        result = parts[0]
        for p in parts[1:]:
            if p.startswith("- "):
                result += " " + p
            else:
                result += " + " + p
        return result

    # ── Separar x-terms e const-terms ────────────────────────────────────────
    x_terms    = [t for t in new_terms if     t.has(x)]
    const_terms = [t for t in new_terms if not t.has(x)]

    # ── Processar variáveis ───────────────────────────────────────────────────
    if len(x_terms) > 1:
        coefs = [sp.Rational(t.coeff(x)) for t in x_terms]
        dens  = [c.q for c in coefs]
        den_comum = _freduce(_mlcm, dens)

        # Verificar se já têm todos o mesmo denominador
        all_same = all(d == den_comum for d in dens)

        if not all_same:
            # Passo A: todos os termos convertidos para o denominador comum
            ns = [c.p * (den_comum // c.q) for c in coefs]
            parts_a = [_frac_x_latex(n, den_comum) for n in ns]
            latex_a = _join_latex(parts_a)
            steps.append(("__latex__", latex_a, new_terms.copy()))
        else:
            ns = [c.p for c in coefs]

        # Passo B: (n1 + n2 + ...)/den · x  — numeradores não avaliados
        sum_num = sum(ns)
        num_expr = " + ".join(str(n) if n >= 0 else str(n) for n in ns)
        # Simplificar sinal: "1 + -6" → "1 - 6"
        num_expr = re.sub(r'\+\s*-', '- ', num_expr)
        if den_comum > 1:
            latex_b = fr"\frac{{({num_expr}) x}}{{{den_comum}}}"
        else:
            latex_b = f"({num_expr}) x"
        if not all_same:
            steps.append(("__latex__", latex_b, new_terms.copy()))

        # Passo C: resultado final
        combined = sp.Rational(sum_num, den_comum) * x
        new_terms = const_terms + [combined] if const_terms else [combined]
        # devolver só o x-term actualizado + consts originais na mesma ordem
        new_terms = [combined] + const_terms
        steps.append(new_terms.copy())

    # ── Processar constantes ──────────────────────────────────────────────────
    elif len(const_terms) > 1:
        coefs_c = [sp.Rational(t) for t in const_terms]
        dens_c  = [c.q for c in coefs_c]
        den_comum_c = _freduce(_mlcm, dens_c)
        all_same_c = all(d == den_comum_c for d in dens_c)

        if not all_same_c:
            ns_c = [c.p * (den_comum_c // c.q) for c in coefs_c]
            parts_a = [_frac_latex(n, den_comum_c) for n in ns_c]
            latex_a = _join_latex(parts_a)
            steps.append(("__latex__", latex_a, new_terms.copy()))
        else:
            ns_c = [c.p for c in coefs_c]

        sum_num_c = sum(ns_c)
        num_expr_c = " + ".join(str(n) for n in ns_c)
        num_expr_c = re.sub(r'\+\s*-', '- ', num_expr_c)
        latex_b_c = fr"\frac{{{num_expr_c}}}{{{den_comum_c}}}" if den_comum_c > 1 else f"({num_expr_c})"
        if not all_same_c:
            steps.append(("__latex__", latex_b_c, new_terms.copy()))

        combined_c = sp.Rational(sum_num_c, den_comum_c)
        new_terms_c = x_terms + [combined_c] if x_terms else [combined_c]
        steps.append(new_terms_c.copy())

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
    """Return the largest rational divisor that divides all constant terms."""
    if not terms:
        return sp.Integer(1)

    rats = [sp.Rational(t) for t in terms]
    denoms = [r.q for r in rats]
    mmc = lcm(*denoms) if len(denoms) > 0 else 1
    nums = [int((r * mmc)) for r in rats]
    gcd_nums = abs(reduce(sp.gcd, [sp.Integer(n) for n in nums])) if nums else 1

    return sp.Rational(gcd_nums, mmc)


def _unwrap(expr):
    if isinstance(expr, sp.UnevaluatedExpr):
        return _unwrap(expr.args[0])
    if expr.args:
        new_args = [_unwrap(a) for a in expr.args]
        if any(na is not a for na, a in zip(new_args, expr.args)):
            return expr.func(*new_args, evaluate=False)
    return expr


def _collect_mul_add_steps(sym):
    pairs = []
    sym = _unwrap(sym)

    def _reduce_muls(expr):
        if isinstance(expr, sp.Mul) and all(a.is_number for a in expr.args):
            result = sp.Mul(*expr.args)
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
            result = sp.Add(*expr.args)
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
    expr_str = expr_str.strip()
    cleaned = fix_implicit_mul(expr_str)
    sym = safe_sympify(cleaned)

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

    final_latex = sp.latex(sp.simplify(final))
    if steps_latex[-1] != final_latex:
        steps_latex.append(final_latex)

    deduped = [steps_latex[0]]
    for s in steps_latex[1:]:
        if s != deduped[-1]:
            deduped.append(s)

    return deduped


def merge_side_steps(left_steps, right_steps, initial_left, initial_right, steps):
    cur_left  = initial_left
    cur_right = initial_right

    for l_after in left_steps:
        before = f"{cur_left} = {cur_right}"
        after  = f"{l_after} = {cur_right}"
        if before != after:
            steps.append(Step(before=before, after=after))
        cur_left = l_after

    for r_after in right_steps:
        before = f"{cur_left} = {cur_right}"
        after  = f"{cur_left} = {r_after}"
        if before != after:
            steps.append(Step(before=before, after=after))
        cur_right = r_after

    return cur_left, cur_right


def _decimal_str(solution):
    if isinstance(solution, sp.Integer) or (isinstance(solution, sp.Rational) and solution.q == 1):
        return None
    val = float(solution)
    rounded = round(val, 3)
    if rounded == int(rounded):
        return str(int(rounded))
    s = f'{rounded:.3f}'.rstrip('0').rstrip('.')
    return s


def _fraction_simplification_steps(num_str, den_str):
    import math
    num, den = int(num_str), int(den_str)
    steps = [r'\frac{' + num_str + r'}{' + den_str + r'}']
    g = math.gcd(abs(num), den)
    if g > 1:
        steps.append(r'\frac{' + str(num // g) + r'}{' + str(den // g) + r'}')
    return steps


def _decimal_simplification_steps(decimal_str):
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


# ─────────────────────────────────────────────────────────────────────────────
# CORREÇÃO 2 — _rational_coef_solve_steps
#
# Quando o coeficiente de x é racional (ex: -5/2), em vez de saltar
# directamente para x = solução, mostra os seguintes passos intermédios:
#
#   Eq inicial :  (-5/2)x = -12
#   Passo 1    :  -5x = (-12)·2          (multiplicar ambos os lados por den)
#   Passo 2    :  x = ((-12)·2) / (-5)   (dividir ambos os lados por num)
#   Passo 3    :  x = -24 / -5           (avaliar a multiplicação)
#   Passo 4    :  x = 24/5               (simplificar o sinal)
#
# A função devolve uma lista de Step's prontos a concatenar ao array steps
# principal de solve_linear.
# ─────────────────────────────────────────────────────────────────────────────

def _rational_coef_solve_steps(coef_rational, const_rational, final_left_latex, final_right_latex):
    """
    Gera passos para resolver (p/q)·x = c quando q > 1.

    Sequência:
      -x/5 = 25/2
      → [expl] Multiplicar ambos os lados por 5
      → -x = 25·5/2          (se c é fracção, manter numerador/denominador)
      → -x = 125/2           (avaliar produto)
      → [expl] Dividir ambos os lados por -1
      → x = -(125/2)
      → x = -125/2           (simplificar sinal)
    """
    result_steps = []

    p = coef_rational.p   # numerador do coeficiente  (ex: -1  ou  -5)
    q = coef_rational.q   # denominador               (ex:  5  ou   2)
    c = const_rational    # lado direito como Rational (ex: Rational(25,2))

    eq_0 = f"{final_left_latex} = {final_right_latex}"

    # ── Passo 0: explicação — multiplicar por q ───────────────────────────────
    result_steps.append(Step(
        before=eq_0, after=eq_0,
        explanation=f"Multiplicar ambos os lados por {q}"
    ))

    # ── Passo 1: p·x = c·q ───────────────────────────────────────────────────
    # lado esquerdo: p·x  (inteiro, denominador eliminado)
    left_1_latex = sp.latex(sp.Integer(p) * x)

    # lado direito: c·q  — se c é fracção mostramos (cn/cd)·q = cn·q/cd
    cn, cd = c.p, c.q
    # Parênteses se cn negativo
    cn_str = f"({cn})" if cn < 0 else str(cn)
    if cd == 1:
        # c é inteiro
        right_1_latex = f"{cn_str} \\cdot {q}"
    else:
        # c é fracção: mostrar cn·q / cd
        right_1_latex = f"\\frac{{{cn_str} \\cdot {q}}}{{{cd}}}"

    eq_1 = f"{left_1_latex} = {right_1_latex}"
    result_steps.append(Step(before=eq_0, after=eq_1))

    # Avaliar o produto  cn·q
    prod_num = cn * q      # ex: 25 * 5 = 125
    if cd == 1:
        right_eval_latex = f"{prod_num}"
    else:
        right_eval_latex = sp.latex(sp.Rational(prod_num, cd))

    eq_1b = f"{left_1_latex} = {right_eval_latex}"
    if eq_1b != eq_1:
        result_steps.append(Step(before=eq_1, after=eq_1b))
    else:
        eq_1b = eq_1

    # ── Passo 2: dividir por p → x = resultado/p ─────────────────────────────
    result_steps.append(Step(
        before=eq_1b, after=eq_1b,
        explanation=f"Dividir ambos os lados por {p}"
    ))

    # Calcular solução: (c·q) / p = cn·q / (cd·p)
    solution = sp.Rational(prod_num, cd * p)   # SymPy simplifica automaticamente

    # Mostrar passo intermédio: x = right_eval / p  (não avaliado)
    if cd == 1:
        right_2_latex = f"\\frac{{{prod_num}}}{{{p}}}"
    else:
        # right_eval já é frac{prod_num}{cd}, dividir por p → frac{prod_num}{cd·p}
        right_2_latex = f"\\frac{{{prod_num}}}{{{cd * p}}}"

    eq_2 = f"x = {right_2_latex}"
    result_steps.append(Step(before=eq_1b, after=eq_2))

    # ── Passo 3: simplificar → x = solução ───────────────────────────────────
    eq_3 = f"x = {sp.latex(solution)}"
    if eq_3 != eq_2:
        result_steps.append(Step(before=eq_2, after=eq_3))

    return result_steps, solution


# Função principal que resolve equações lineares passo a passo
def solve_linear(equation: str):

    left, right = parse_equation(equation)

    steps = []

    _raw_decimals   = detect_decimals(equation)
    _raw_fractions  = detect_raw_fractions(equation)

    import re as _re_disp
    def _eq_to_latex_display(eq):
        return _re_disp.sub(
            r'(?<!\\\\)(-?\d+)/(\d+)',
            lambda m: r'\frac{' + m.group(1) + r'}{' + m.group(2) + r'}',
            eq
        )

    equation_display = _eq_to_latex_display(equation)

    current_eq_display = equation_display
    for dec_str, dec_val in _raw_decimals:
        dec_steps = _decimal_simplification_steps(dec_str)
        for i in range(1, len(dec_steps)):
            before_eq = current_eq_display
            src = dec_steps[i-1]
            dst = dec_steps[i]
            after_eq = before_eq.replace(src, dst, 1)
            if before_eq != after_eq:
                steps.append(Step(
                    before=before_eq,
                    after=after_eq,
                    explanation='Converter decimal para fração' if i == 1 else 'Simplificar fração'
                ))
                current_eq_display = after_eq

    for num_s, den_s, frac_val in _raw_fractions:
        frac_steps   = _fraction_simplification_steps(num_s, den_s)
        frac_orig_str = num_s + '/' + den_s
        if frac_orig_str in current_eq_display:
            eq_with_frac = current_eq_display.replace(frac_orig_str, frac_steps[0], 1)
            if eq_with_frac != current_eq_display:
                steps.append(Step(before=current_eq_display, after=eq_with_frac))
                current_eq_display = eq_with_frac
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

    left_terms  = extract_terms(left)
    right_terms = extract_terms(right)

    left_x, left_const   = [], []
    right_x, right_const = [], []

    for t in left_terms:
        if t.has(x): left_x.append(t)
        else:        left_const.append(t)

    for t in right_terms:
        if t.has(x): right_x.append(t)
        else:        right_const.append(t)

    variable_terms  = left_x + [-t for t in right_x]
    constant_terms  = right_const + [-t for t in left_const]

    new_eq = build_equation(variable_terms, constant_terms)

    steps.append(Step(
        before=equation_display,
        after=equation_display,
        explanation="Organizar termos"
    ))
    steps.append(Step(
        before=equation_display,
        after=new_eq,
    ))

    if len(variable_terms) > 1:
        steps.append(Step(
            before=new_eq,
            after=new_eq,
            explanation="Vamos resolver o lado das variáveis"
        ))

    # --- Simplify variables --- (CORREÇÃO 1: lidar com marcadores __latex__)
    current_vars = variable_terms
    var_steps    = combine_terms_stepwise(variable_terms)

    for entry in var_steps:
        if isinstance(entry, tuple) and entry[0] == "__latex__":
            # entry = ("__latex__", var_latex_str, current_sympy_terms)
            # Construir before a partir do estado SymPy actual (current_vars)
            # e after com o LaTeX manual fornecido + lado direito do before
            _, var_latex, state_terms = entry
            before_eq  = build_equation(current_vars, constant_terms)
            const_side = before_eq.split("=")[1].strip()
            after_eq   = f"{var_latex} = {const_side}"
            if before_eq != after_eq:
                steps.append(Step(before=before_eq, after=after_eq))
            # NÃO actualizar current_vars aqui — o step seguinte (lista SymPy)
            # fá-lo com o estado final correcto
        else:
            new_vars = entry
            steps.append(Step(
                before=build_equation(current_vars, constant_terms),
                after=build_equation(new_vars, constant_terms)
            ))
            current_vars = new_vars

    # --- Simplify constants ---
    current_consts = constant_terms

    if len(constant_terms) > 1:
        steps.append(Step(
            before=build_equation(current_vars, current_consts),
            after=build_equation(current_vars, current_consts),
            explanation="Agora vamos resolver o lado das constantes"
        ))

    const_steps = combine_terms_stepwise(constant_terms)

    for entry in const_steps:
        if isinstance(entry, tuple) and entry[0] == "__latex__":
            _, const_latex, _ = entry
            before_eq = build_equation(current_vars, current_consts)
            var_side  = before_eq.split("=")[0].strip()
            after_eq  = f"{var_side} = {const_latex}"
            if before_eq != after_eq:
                steps.append(Step(before=before_eq, after=after_eq))
        else:
            new_consts = entry
            steps.append(Step(
                before=build_equation(current_vars, current_consts),
                after=build_equation(current_vars, new_consts),
            ))
            current_consts = new_consts

    # Final solve
    final_left  = current_vars[0]
    final_right = current_consts[0]

    coef  = final_left.coeff(x)
    const = final_right

    def safe_gcd(a, b):
        return sp.gcd(sp.Rational(a), sp.Rational(b))

    def _sympy_stepwise(subst_str, sym_evaled, final_value):
        """
        Gera lista de tuplos (latex_str, explicacao_ou_None) mostrando cada
        passo aritmético após substituição de x por final_value.

        Sequência para final_value = p/q (fração):
          1. Expandir cada coef*(p/q)  →  (coef*p)/(q)   [produto no numerador]
          2. Simplificar cada fração   →  forma reduzida
          3. Converter inteiros para denominador comum
          4. Somar/subtrair frações par a par: mostrar (a op b)/d antes do resultado

        Sequência para final_value inteiro:
          1. Calcular cada produto coef*val
          2. Somar termos numéricos
        """
        import re as _re
        from math import lcm as _mlcm
        from functools import reduce as _freduce

        def _fix_pm(s):
            s = _re.sub(r'-\s*-\s*', '+ ', s)
            s = _re.sub(r'\+\s*-\s*', '- ', s)
            return s.strip()

        result = [(subst_str, None)]
        cur = subst_str

        # ── Caso inteiro ──────────────────────────────────────────────────────
        if not (isinstance(final_value, sp.Rational) and final_value.q != 1):
            val_int = int(final_value)
            val_latex = sp.latex(final_value)
            val_esc = _re.escape(val_latex)
            # Passo 1: resolver cada coef*(val)
            pattern = r'(-?\s*\d+)\s*\(\s*' + val_esc + r'\s*\)'
            for _ in range(20):
                m = _re.search(pattern, cur)
                if not m: break
                coef_v = int(m.group(1).replace(' ', ''))
                prod = coef_v * val_int
                before_m = cur[:m.start()]
                after_m  = cur[m.end():]
                prod_str = str(prod)
                if before_m.rstrip() and before_m.rstrip()[-1] not in '+-=(,':
                    prod_str = ('+ ' if prod >= 0 else '- ') + str(abs(prod))
                s1 = _fix_pm(before_m + prod_str + after_m)
                if s1 == cur: break
                result.append((s1, None)); cur = s1
            # Passo 2: somar pares numéricos
            final_str = sp.latex(sp.simplify(sym_evaled))
            for _ in range(20):
                m = _re.search(r'(-?\d+)\s*([+-])\s*(\d+)', cur)
                if not m: break
                a, op, b = int(m.group(1)), m.group(2), int(m.group(3))
                soma = a + b if op == '+' else a - b
                novo = _fix_pm(cur[:m.start()] + str(soma) + cur[m.end():])
                if novo == cur: break
                result.append((novo, None)); cur = novo
                if cur == final_str: break
            if cur != final_str:
                result.append((final_str, None))
            return result

        # ── Caso fração p/q ───────────────────────────────────────────────────
        val_num = final_value.p   # ex: -125
        val_den = final_value.q   # ex:   2
        val_latex = sp.latex(final_value)
        val_esc   = _re.escape(val_latex)

        # Passo 1a: coef_inteiro * (\frac{p}{q})
        # Sequência: \frac{coef \cdot (val_num)}{val_den}
        #          → \frac{prod_num}{val_den}
        #          → simplificado (inteiro ou fração reduzida)
        pat_coef_frac = r'(-?\s*\d+)\s*\(\s*' + val_esc + r'\s*\)'
        for _ in range(20):
            m = _re.search(pat_coef_frac, cur)
            if not m: break
            coef_v   = int(m.group(1).replace(' ', ''))
            prod_num = coef_v * val_num
            before_m = cur[:m.start()]
            after_m  = cur[m.end():]

            # Sempre mostrar \cdot, com parênteses se val_num < 0
            value_num_str = f"({val_num})" if val_num < 0 else str(val_num)
            frac_prod = r'\frac{' + str(coef_v) + r' \cdot ' + value_num_str + r'}{' + str(val_den) + r'}'
            s1 = _fix_pm(before_m + frac_prod + after_m)
            if s1 != cur: result.append((s1, None)); cur = s1

            # Avaliar numerador: \frac{prod_num}{val_den}
            # Se cancela para inteiro, saltar directamente (sem mostrar a fração intermédia)
            frac_eval = r'\frac{' + str(prod_num) + r'}{' + str(val_den) + r'}'
            frac_simp = sp.latex(sp.Rational(prod_num, val_den))
            if frac_simp == frac_eval:
                # Não simplifica — mostrar frac_eval e ficar
                s2 = _fix_pm(cur.replace(frac_prod, frac_eval, 1))
                if s2 != cur: result.append((s2, None)); cur = s2
            else:
                # Simplifica (inteiro ou fração reduzida) — saltar frac_eval se for inteiro
                simp_is_int = not ('\\frac' in frac_simp)
                if simp_is_int:
                    # Vai directamente de frac_prod para o inteiro
                    s3 = _fix_pm(cur.replace(frac_prod, frac_simp, 1))
                    if s3 != cur: result.append((s3, None)); cur = s3
                else:
                    # Fração reduzida: mostrar frac_eval primeiro, depois simplificar
                    s2 = _fix_pm(cur.replace(frac_prod, frac_eval, 1))
                    if s2 != cur: result.append((s2, None)); cur = s2
                    s3 = _fix_pm(cur.replace(frac_eval, frac_simp, 1))
                    if s3 != cur: result.append((s3, None)); cur = s3

        # Passo 1b: \frac{a}{b} * (\frac{p}{q})
        # Sequência: \frac{a \cdot (val_num)}{b \cdot val_den}
        #          → \frac{prod_n}{prod_d}
        #          → simplificado
        pat_frac_frac = r'\\frac\{(\d+)\}\{(\d+)\}\s*\(\s*' + val_esc + r'\s*\)'
        for _ in range(20):
            m = _re.search(pat_frac_frac, cur)
            if not m: break
            fa, fb = int(m.group(1)), int(m.group(2))
            prod_n = fa * val_num
            prod_d = fb * val_den
            before_m = cur[:m.start()]
            after_m  = cur[m.end():]

            value_num_str = f"({val_num})" if val_num < 0 else str(val_num)
            frac_prod2 = r'\frac{' + str(fa) + r' \cdot ' + value_num_str + r'}{' + str(fb) + r' \cdot ' + str(val_den) + r'}'
            s1 = _fix_pm(before_m + frac_prod2 + after_m)
            if s1 != cur: result.append((s1, None)); cur = s1

            frac_eval2 = r'\frac{' + str(prod_n) + r'}{' + str(prod_d) + r'}'
            frac_simp2 = sp.latex(sp.Rational(prod_n, prod_d))
            if frac_simp2 == frac_eval2:
                s2 = _fix_pm(cur.replace(frac_prod2, frac_eval2, 1))
                if s2 != cur: result.append((s2, None)); cur = s2
            else:
                simp_is_int2 = '\\frac' not in frac_simp2
                if simp_is_int2:
                    s3 = _fix_pm(cur.replace(frac_prod2, frac_simp2, 1))
                    if s3 != cur: result.append((s3, None)); cur = s3
                else:
                    s2 = _fix_pm(cur.replace(frac_prod2, frac_eval2, 1))
                    if s2 != cur: result.append((s2, None)); cur = s2
                    s3 = _fix_pm(cur.replace(frac_eval2, frac_simp2, 1))
                    if s3 != cur: result.append((s3, None)); cur = s3

        # Passo 2: converter inteiros isolados para o denominador comum das frações presentes
        def _find_isolated_ints(s):
            """Encontra inteiros fora de {}, não seguidos de /."""
            found = []; depth = 0; i = 0
            while i < len(s):
                c = s[i]
                if c == '{': depth += 1; i += 1; continue
                if c == '}': depth -= 1; i += 1; continue
                if depth > 0: i += 1; continue
                # Sinal opcional
                sign = 1
                start = i
                if c == '-' and i + 1 < len(s) and s[i+1].isdigit():
                    # verificar se é sinal de menos (precedido por espaço/operador)
                    if i == 0 or s[i-1] in ' +=({':
                        sign = -1; i += 1; c = s[i]
                    else:
                        i += 1; continue
                if c.isdigit():
                    j = i
                    while j < len(s) and s[j].isdigit(): j += 1
                    # ignorar se dentro de {} ou seguido de /
                    if j < len(s) and s[j] in '/{': i = j; continue
                    try: n = sign * int(s[i:j])
                    except: i += 1; continue
                    if n != 0: found.append((start, j, n))
                    i = j
                else: i += 1
            return found

        def _dens_in(s):
            return [int(d) for d in _re.findall(r'\\frac\{[^}]+\}\{(\d+)\}', s)]

        dens = _dens_in(cur)
        if dens and _find_isolated_ints(cur):
            den_comum = _freduce(_mlcm, dens)
            result.append((cur, f'Reduzir ao mesmo denominador ({den_comum})'))
            for _ in range(30):
                iso = _find_isolated_ints(cur)
                if not iso: break
                start, end, n = iso[0]
                new_num = n * den_comum
                # Manter sinal: se n < 0, mostrar -\frac{abs*den}{den}
                frac = r'\frac{' + str(new_num) + r'}{' + str(den_comum) + r'}'
                novo = _fix_pm(cur[:start] + frac + cur[end:])
                if novo == cur: break
                result.append((novo, None)); cur = novo

        # Passo 3: somar/subtrair frações par a par
        # Padrão: \frac{a}{d} op \frac{b}{d}  (mesmo denominador)
        fp = r'(\\frac\{(-?\d+)\}\{(\d+)\})\s*([+-])\s*(\\frac\{(-?\d+)\}\{(\d+)\})'
        for _ in range(30):
            m = _re.search(fp, cur)
            if not m: break
            na, da = int(m.group(2)), int(m.group(3))
            op   = m.group(4)
            nb, db = int(m.group(6)), int(m.group(7))
            # Verificar sinal externo (- antes do primeiro \frac)
            prefix = cur[:m.start()].rstrip()
            if prefix.endswith('-'):
                na = -abs(na)
                cur = cur[:len(prefix)-1].rstrip() + ' ' + cur[len(prefix):]
                m = _re.search(fp, cur)
                if not m: break
                na, da = int(m.group(2)), int(m.group(3))
                na = -abs(na)
                op = m.group(4)
                nb, db = int(m.group(6)), int(m.group(7))
            # Se denominadores diferentes, converter para denominador comum
            if da != db:
                from math import lcm as _lcm2
                dc = _lcm2(da, db)
                na_new = na * (dc // da)
                nb_new = nb * (dc // db)
                f_a = r'\frac{' + str(na_new) + r'}{' + str(dc) + r'}'
                f_b = r'\frac{' + str(nb_new) + r'}{' + str(dc) + r'}'
                novo = _fix_pm(cur[:m.start()] + f_a + ' ' + op + ' ' + f_b + cur[m.end():])
                if novo == cur: break
                result.append((novo, None)); cur = novo
                continue
            # Mesmo denominador: mostrar (a op b)/d, depois avaliar numerador, depois simplificar
            nb_s = nb if op == '+' else -nb
            num_expr = f"{na} + {nb}" if op == '+' else f"{na} - {nb}"
            frac_joined = r'\frac{' + num_expr + r'}{' + str(da) + r'}'
            novo_j = _fix_pm(cur[:m.start()] + frac_joined + cur[m.end():])
            if novo_j != cur:
                result.append((novo_j, None)); cur = novo_j

            soma_n = na + nb_s
            # FIX 3: mostrar \frac{soma_n}{da} antes de simplificar
            frac_evaluated = r'\frac{' + str(soma_n) + r'}{' + str(da) + r'}'
            frac_result    = sp.latex(sp.Rational(soma_n, da))
            # Passo intermédio: numerador avaliado (apenas se diferente do joined e do final)
            if frac_evaluated != frac_joined:
                novo_eval = _fix_pm(cur.replace(frac_joined, frac_evaluated, 1))
                if novo_eval != cur:
                    result.append((novo_eval, None)); cur = novo_eval
            # Passo final: simplificar (apenas se diferente do avaliado)
            if frac_result != frac_evaluated:
                novo = _fix_pm(cur.replace(frac_evaluated, frac_result, 1))
                if novo == cur: break
                result.append((novo, None)); cur = novo
            elif frac_result == frac_evaluated:
                # Já está no formato final — substituir o joined pelo resultado
                novo = _fix_pm(cur.replace(frac_joined, frac_result, 1) if frac_joined in cur else cur)
                if novo != cur:
                    result.append((novo, None)); cur = novo

        final_str = sp.latex(sp.simplify(sym_evaled))
        if cur != final_str:
            result.append((final_str, None))

        # Deduplicar mantendo mensagens
        deduped = [result[0]]
        for item in result[1:]:
            s, e = item
            if s != deduped[-1][0] or e is not None:
                deduped.append(item)
        return deduped


    def _check_solution(final_value, final_latex, equation, steps):
        left_str, right_str = equation.split("=")
        left_sym  = sp.sympify(normalize_expression(left_str.strip()), evaluate=False)
        right_sym = sp.sympify(normalize_expression(right_str.strip()), evaluate=False)

        import re as _re2
        def _fracs_to_latex(s):
            return _re2.sub(r'(-?\d+)/(\d+)', lambda m: r'\frac{' + m.group(1) + r'}{' + m.group(2) + r'}', s)
        left_latex_for_display  = _fracs_to_latex(left_str.strip())
        right_latex_for_display = _fracs_to_latex(right_str.strip())
        left_latex_for_display  = _re2.sub(r'(\\frac\{\d+\}\{\d+\})(\s*)([a-zA-Z])', r'\1 \3', left_latex_for_display)
        right_latex_for_display = _re2.sub(r'(\\frac\{\d+\}\{\d+\})(\s*)([a-zA-Z])', r'\1 \3', right_latex_for_display)
        equation_latex = f"{left_latex_for_display} = {right_latex_for_display}"


        steps.append(Step(
            before=equation_latex,
            after="",
            explanation="Vamos verificar!"
        ))

        steps.append(Step(
            before="",
            after=equation_latex,
            explanation=f"Agora vamos substituir x por {final_latex}"
        ))

        left_latex_orig  = left_latex_for_display
        right_latex_orig = right_latex_for_display
        val_latex = sp.latex(final_value)

        def _sub_x_in_latex(latex_str, val):
            import re as _re
            return _re.sub(r'(?<![a-zA-Z])x(?![a-zA-Z])', lambda m: '(' + val + ')', latex_str)

        left_subst_str  = _sub_x_in_latex(left_latex_orig, val_latex)
        right_subst_str = _sub_x_in_latex(right_latex_orig, val_latex)

        substituted_display = f"{left_subst_str} = {right_subst_str}"
        steps.append(Step(
            before=equation_latex,
            after=substituted_display,
        ))

        def _subst_terms(sym, val):
            if isinstance(sym, sp.Add):
                new_terms = [t.xreplace({x: sp.UnevaluatedExpr(val)}) for t in sym.args]
                return sp.Add(*new_terms, evaluate=False)
            return sym.xreplace({x: sp.UnevaluatedExpr(val)})

        left_evaled  = _subst_terms(left_sym,  final_value)
        right_evaled = _subst_terms(right_sym, final_value)

        left_tuples  = _sympy_stepwise(left_subst_str,  left_evaled,  final_value)
        right_tuples = _sympy_stepwise(right_subst_str, right_evaled, final_value)

        def _extract(tuples):
            sl, ex = [], {}
            for i, (s, e) in enumerate(tuples):
                sl.append(s)
                if e: ex[i] = e
            return sl, ex

        left_steps_v,  left_expls  = _extract(left_tuples)
        right_steps_v, right_expls = _extract(right_tuples)

        cur_left  = left_subst_str
        cur_right = right_subst_str
        for i, l_after in enumerate(left_steps_v):
            expl = left_expls.get(i)
            before = f'{cur_left} = {cur_right}'
            if expl:
                steps.append(Step(before=before, after=before, explanation=expl))
            after = f'{l_after} = {cur_right}'
            if before != after:
                steps.append(Step(before=before, after=after))
            cur_left = l_after
        for i, r_after in enumerate(right_steps_v):
            expl = right_expls.get(i)
            before = f'{cur_left} = {cur_right}'
            if expl:
                steps.append(Step(before=before, after=before, explanation=expl))
            after = f'{cur_left} = {r_after}'
            if before != after:
                steps.append(Step(before=before, after=after))
            cur_right = r_after

        left_orig  = sp.sympify(normalize_expression(left_str.strip()), evaluate=False)
        right_orig = sp.sympify(normalize_expression(right_str.strip()), evaluate=False)
        left_final  = left_orig.subs(x, final_value)
        right_final = right_orig.subs(x, final_value)

        is_true = sp.simplify(left_final - right_final) == 0
        explanation = "The solution is correct!" if is_true else "The solution does not satisfy the equation."

        steps.append(Step(
            before=f"{cur_left} = {cur_right}",
            after=f"{cur_left} = {cur_right}",
            explanation=explanation
        ))

    # ── CASO 1: coeficiente já é 1 ────────────────────────────────────────────
    if coef == 1:
        steps.append(Step(
            before=f"{sp.latex(final_left)} = {sp.latex(final_right)}",
            after=f"x = {sp.latex(const)}"
        ))
        final_value = const
        final_latex = sp.latex(final_value)
        _dec = _decimal_str(final_value)
        if _dec:
            sol_latex = f'x = {final_latex}'
            steps.append(Step(
                before=sol_latex,
                after=sol_latex,
                explanation=f'x = {final_latex} \\approx {_dec}'
            ))
        _check_solution(final_value, final_latex, equation, steps)

    # ── CASO 2: precisa dividir pelo coeficiente ──────────────────────────────
    else:
        coef_rat  = sp.Rational(coef)
        const_rat = sp.Rational(const)

        final_left_latex  = sp.latex(final_left)
        final_right_latex = sp.latex(final_right)

        # ── CORREÇÃO 2: coeficiente racional com denominador > 1 ─────────────
        if coef_rat.q > 1:
            extra_steps, solution = _rational_coef_solve_steps(
                coef_rat, const_rat,
                final_left_latex, final_right_latex
            )
            steps.extend(extra_steps)

        # ── Coeficiente inteiro (comportamento original) ──────────────────────
        else:
            coef_int = int(coef_rat)
            divisor_intermedio = safe_gcd(abs(const_rat), abs(coef_rat))
            coef_simpl  = coef_rat  / divisor_intermedio
            const_simpl = const_rat / divisor_intermedio
            mostrar_intermedio = (
                divisor_intermedio > 1
                and divisor_intermedio != abs(coef_rat)
                and isinstance(coef_simpl, sp.Integer) or (isinstance(coef_simpl, sp.Rational) and coef_simpl.q == 1)
                and isinstance(const_simpl, sp.Integer) or (isinstance(const_simpl, sp.Rational) and const_simpl.q == 1)
                and coef_simpl != 1
            )
            if mostrar_intermedio:
                left_int  = coef_simpl * x
                right_int = const_simpl
                steps.append(Step(
                    before=f"{final_left_latex} = {final_right_latex}",
                    after=f"{final_left_latex} = {final_right_latex}",
                    explanation=f"Dividir ambos os lados por {sp.latex(divisor_intermedio)}"
                ))
                steps.append(Step(
                    before=f"{final_left_latex} = {final_right_latex}",
                    after=f"{sp.latex(left_int)} = {sp.latex(right_int)}",
                ))
                steps.append(Step(
                    before=f"{sp.latex(left_int)} = {sp.latex(right_int)}",
                    after=f"{sp.latex(left_int)} = {sp.latex(right_int)}",
                    explanation=f"Dividir ambos os lados por {sp.latex(coef_simpl)}"
                ))
                steps.append(Step(
                    before=f"{sp.latex(left_int)} = {sp.latex(right_int)}",
                    after=f"x = {sp.latex(sp.Rational(const_rat, coef_rat))}",
                ))
            else:
                steps.append(Step(
                    before=f"{final_left_latex} = {final_right_latex}",
                    after=f"{final_left_latex} = {final_right_latex}",
                    explanation=f"Dividir ambos os lados por {sp.latex(coef_rat)}"
                ))
                steps.append(Step(
                    before=f"{final_left_latex} = {final_right_latex}",
                    after=f"x = {sp.latex(sp.Rational(const_rat, coef_rat))}",
                ))
            solution = sp.Rational(const_rat, coef_rat)

        final_value = solution
        final_latex = sp.latex(final_value)

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


