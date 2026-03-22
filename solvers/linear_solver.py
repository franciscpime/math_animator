import sympy as sp
from sympy import lcm
import re
from functools import reduce
from models.step import Step
from parser.equation_parser import parse_equation, normalize_expression, safe_sympify, fix_implicit_mul
from math_utils.mmc import compute_mmc, apply_mmc
from utils.term_extractor import extract_terms, detailed_multiplication
from utils.equation_builder import build_equation, render_terms

# Define o símbolo 'x' que representa a incógnita em todas as equações.
# O SymPy precisa de saber que 'x' é uma variável simbólica e não um número.
x = sp.symbols("x")


def combine_terms_stepwise(terms: list[sp.Basic]) -> list[list[sp.Basic]]:
    """
    Combina termos semelhantes numa lista passo a passo, garantindo que
    só são combinados termos do mesmo tipo entre si:
      - termos com x só são somados com outros termos com x
      - constantes só são somadas com outras constantes
    Isto evita combinações incorrectas como somar '2x' com '3'.

    Devolve uma lista de snapshots — cada elemento é o estado da lista
    de termos depois de uma combinação.
    """
    new_terms = terms.copy()
    steps = []

    i = 0
    while i < len(new_terms) - 1:
        current_term = new_terms[i]
        next_term = new_terms[i + 1]

        # Verificar se ambos os termos contêm x
        both_have_x = current_term.has(x) and next_term.has(x)

        # Verificar se ambos os termos são constantes (sem x)
        both_are_const = not current_term.has(x) and not next_term.has(x)

        if both_have_x:
            # Somar os coeficientes de x e criar o novo termo combinado
            combined_coefficient = current_term.coeff(x) + next_term.coeff(x)
            new_term = combined_coefficient * x

            # Substituir os dois termos pelo novo termo combinado na lista
            new_terms = new_terms[:i] + [new_term] + new_terms[i + 2:]

            # Guardar uma cópia do estado actual como passo intermédio
            steps.append(new_terms.copy())

            # Não avançar i >> pode haver mais termos com x para combinar
        elif both_are_const:
            # Somar as duas constantes directamente
            new_term = current_term + next_term
            new_terms = new_terms[:i] + [new_term] + new_terms[i + 2:]
            steps.append(new_terms.copy())
        else:
            # Os dois termos são de tipos diferentes (um com x, outro constante)
            # não combinar, avançar para o par seguinte
            i += 1

    return steps


def substitution_steps(expression: sp.Basic, value: int) -> list[sp.Basic]:
    """
    Gera uma sequência de expressões que mostram a substituição de x
    pelo valor dado, seguida de simplificação gradual.

    Passos produzidos:
      1. Expressão com x substituído mas sem avaliar
      2. Expressão expandida (se diferente do passo anterior)
      3. Expressão completamente simplificada (se diferente)
    """
    steps = []

    # Substituir x pelo valor sem avaliar imediatamente
    replaced_expression = expression.subs(x, sp.Integer(value), evaluate=False)
    steps.append(replaced_expression)

    # Expandir a expressão (distribui multiplicações, etc.)
    expanded = sp.expand(replaced_expression)
    if expanded != replaced_expression:
        steps.append(expanded)

    # Simplificar completamente
    final = sp.simplify(expanded)
    if final != expanded:
        steps.append(final)

    return steps


def common_divisor_of_constants(terms: list[sp.Basic]) -> sp.Rational:
    """
    Calcula o maior divisor racional comum a todos os termos constantes da lista.

    Para constantes racionais, devolve mdc(numeradores) / mmc(denominadores)
    como um sp.Rational — funciona tanto para inteiros como para frações.

    Exemplo: para [6, 4] devolve 2; para [1/2, 1/3] devolve 1/6.
    """
    if not terms:
        return sp.Integer(1)

    # Converter todos os termos para Rational para tratar inteiros e frações uniformemente
    rats = [sp.Rational(t) for t in terms]

    # Calcular o mínimo múltiplo comum dos denominadores
    denoms = [r.q for r in rats]
    mmc = lcm(*denoms) if len(denoms) > 0 else 1

    # Escalar cada racional para o denominador comum e calcular o mdc dos numeradores
    nums = [int((r * mmc)) for r in rats]
    gcd_nums = abs(reduce(sp.gcd, [sp.Integer(n) for n in nums])) if nums else 1

    return sp.Rational(gcd_nums, mmc)


def _unwrap(expr: sp.Basic) -> sp.Basic:
    """
    Remove recursivamente qualquer UnevaluatedExpr que envolva um valor.

    O SymPy usa UnevaluatedExpr para impedir a avaliação automática de
    sub-expressões. Contudo, quando percorremos o AST para reduzir
    multiplicações e somas numéricas, o UnevaluatedExpr bloqueia a
    detecção correcta de 'is_number' nos nós pai.

    Exemplo: UnevaluatedExpr(3) tem is_number=True mas o Mul(2, UnevaluatedExpr(3))
    não é reconhecido como "todos os args são números" sem este unwrap.
    """
    if isinstance(expr, sp.UnevaluatedExpr):
        # Desembrulhar e continuar recursivamente
        return _unwrap(expr.args[0])
    if expr.args:
        # Aplicar recursivamente a todos os filhos do nó
        new_args = [_unwrap(a) for a in expr.args]
        # Se algum filho mudou, reconstruir o nó sem avaliar
        if any(na is not a for na, a in zip(new_args, expr.args)):
            return expr.func(*new_args, evaluate=False)
    return expr


def _collect_mul_add_steps(sym: sp.Basic) -> tuple[list[tuple[sp.Basic, sp.Basic]], sp.Basic]:
    """
    Percorre o AST SymPy de 'sym' (mantido sem avaliar) e devolve:
      - uma lista de pares (before, after): cada par representa uma redução
        aritmética elementar que foi feita (ex: Mul(2,3) >> 6)
      - a expressão final depois de todas as reduções

    Ordem de processamento:
      1.ª passagem: encontrar e reduzir todos os Mul cujos args são números
      2.ª passagem: encontrar e reduzir todos os Add cujos args são números
    O processamento é post-order: os nós mais internos são reduzidos primeiro.

    Porquê não usar evaluate=False directamente?
    Porque o SymPy avalia expressões puramente numéricas imediatamente,
    mesmo com evaluate=False — esse flag só suprime simplificação algébrica,
    não constant folding de inteiros/racionais.
    """
    pairs = []

    # Remover UnevaluatedExpr antes de processar para que is_number funcione
    sym = _unwrap(sym)

    def _reduce_muls(expr: sp.Basic) -> sp.Basic:
        """
        Encontra o primeiro Mul cujos args são todos números, avalia-o,
        regista o par (antes, depois) e devolve a expressão modificada.
        Recursivo em post-order para tratar sub-expressões internas primeiro.
        """
        if isinstance(expr, sp.Mul) and all(a.is_number for a in expr.args):
            # Avaliar intencionalmente: 2*3 >> 6
            result = sp.Mul(*expr.args)
            pairs.append((expr, result))
            return result
        # Percorrer os filhos do nó
        new_args = []
        changed = False
        for a in expr.args:
            na = _reduce_muls(a)
            if na is not a:
                changed = True
                new_args.append(na)
            else:
                new_args.append(a)
        # Reconstruir o nó sem avaliar se algum filho mudou
        if changed:
            return expr.func(*new_args, evaluate=False)
        return expr

    def _reduce_adds(expr: sp.Basic) -> sp.Basic:
        """
        Encontra o primeiro Add cujos args são todos números, avalia-o,
        regista o par (antes, depois) e devolve a expressão modificada.
        Recursivo em post-order — idêntico a _reduce_muls mas para somas.
        """
        if isinstance(expr, sp.Add) and all(a.is_number for a in expr.args):
            # Avaliar intencionalmente: 1+6 >> 7
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

    # Reduzir todos os Mul até não haver mais nenhum para reduzir
    current = sym
    while True:
        before = current
        current = _reduce_muls(current)
        if current == before:
            break  # nenhum Mul foi reduzido — terminar

    # Reduzir todos os Add até não haver mais nenhum para reduzir
    while True:
        before = current
        current = _reduce_adds(current)
        if current == before:
            break  # nenhum Add foi reduzido — terminar

    return pairs, current


def stepwise_string_eval(expr_str: str) -> list[str]:
    """
    Recebe uma string de expressão numérica (depois da substituição de x)
    e devolve uma lista ordenada de strings LaTeX, uma por cada redução
    aritmética elementar.

    O primeiro elemento é a expressão tal como está.
    O último elemento é o valor completamente simplificado.
    Entradas adjacentes diferem por exactamente uma operação.

    Nota: esta função recebe strings LaTeX e usa safe_sympify para as
    converter. Para expressões que já estão em objectos SymPy, usar
    _sympy_stepwise (definida dentro de solve_linear) é mais seguro.
    """
    expr_str = expr_str.strip()

    # Tratar multiplicação implícita como "2(3)" >> "2*(3)"
    cleaned = fix_implicit_mul(expr_str)
    sym = safe_sympify(cleaned)

    # Se já é um número simples (inteiro, racional), não há passos a mostrar
    if sym.is_number and not isinstance(sym, (sp.Add, sp.Mul)):
        return [sp.latex(sym)]

    steps_latex = [sp.latex(sym)]

    # Recolher todos os pares de redução elementar
    pairs, final = _collect_mul_add_steps(sym)

    seen = set()
    for before, after in pairs:
        key = (str(before), str(after))
        # Ignorar pares duplicados ou sem mudança
        if key in seen or before == after:
            continue
        seen.add(key)
        steps_latex.append(sp.latex(after))

    # Garantir que o valor final simplificado é sempre o último elemento
    final_latex = sp.latex(sp.simplify(final))
    if steps_latex[-1] != final_latex:
        steps_latex.append(final_latex)

    # Remover entradas consecutivas idênticas (podem surgir de simplificações)
    deduped = [steps_latex[0]]
    for s in steps_latex[1:]:
        if s != deduped[-1]:
            deduped.append(s)

    return deduped


def merge_side_steps(
        left_steps: list[str], 
        right_steps: list[str], 
        initial_left: str,
        initial_right: str, 
        steps: list[Step]
    ) -> tuple[str, str]:
    """
    Intercala os passos do lado esquerdo e direito de uma equação em
    objectos Step, adicionando-os à lista 'steps'.

    Cada Step representa uma transição visível na animação.
    Só são emitidos Steps em que before != after — transições nulas
    (onde nada muda visualmente) são ignoradas.

    Se os dois lados tiverem números diferentes de passos, o lado mais
    curto é preenchido repetindo o seu último valor.

    Devolve (cur_left, cur_right): as strings LaTeX finais de cada lado.
    """
    # Alinhar os dois lados ao mesmo comprimento
    max_len = max(len(left_steps), len(right_steps))
    left_padded  = left_steps  + [left_steps[-1]]  * (max_len - len(left_steps))
    right_padded = right_steps + [right_steps[-1]] * (max_len - len(right_steps))

    cur_left  = initial_left
    cur_right = initial_right

    for l_after, r_after in zip(left_padded, right_padded):
        before = f"{cur_left} = {cur_right}"
        after  = f"{l_after} = {r_after}"

        # Só adicionar o Step se houver uma mudança visível
        if before != after:
            steps.append(Step(before=before, after=after))

        # Actualizar o estado actual de cada lado
        cur_left  = l_after
        cur_right = r_after

    return cur_left, cur_right


def solve_linear(equation: str) -> list[Step]:
    """
    Resolve uma equação linear de primeiro grau passo a passo.

    Recebe a equação como string (ex: "2x + 1 = 7") e devolve uma lista
    de objectos Step, cada um com:
      - before: a expressão antes da transformação (em LaTeX ou string)
      - after:  a expressão depois da transformação
      - explanation: texto opcional a mostrar na animação

    Fluxo geral:
      1. Organizar termos (variáveis à esquerda, constantes à direita)
      2. Simplificar variáveis (combinar termos com x)
      3. Simplificar constantes (combinar termos numéricos)
      4. Isolar x (dividir pelo coeficiente se necessário)
      5. Verificar a solução substituindo x na equação original
    """

    # Separar a equação em lado esquerdo e lado direito, já normalizados
    left, right = parse_equation(equation)

    steps = []

    # Extrair todos os termos de cada lado como objectos SymPy
    left_terms  = extract_terms(left)
    right_terms = extract_terms(right)

    # Separar os termos com x das constantes em cada lado
    left_x, left_const   = [], []
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

    # Reagrupar: todos os termos com x à esquerda, constantes à direita.
    # Os termos que mudam de lado trocam de sinal (ex: +1 passa para -1).
    variable_terms = left_x + [-t for t in right_x]
    constant_terms = right_const + [-t for t in left_const]

    # Construir a representação da equação reorganizada
    new_eq = build_equation(variable_terms, constant_terms)

    # Mostrar a transição da equação original para a equação organizada
    steps.append(
        Step(
            before=equation,
            after=new_eq,
            explanation="Organizar termos"
        )
    )

    # Se houver mais do que um termo com x, anunciar que vamos simplificá-los
    if len(variable_terms) > 1:
        steps.append(
            Step(
                before=new_eq,
                after=new_eq,
                explanation="Vamos resolver o lado das variáveis"
            )
        )

    # Simplificar os termos com x
    current_vars = variable_terms
    var_steps = combine_terms_stepwise(variable_terms)

    for new_vars in var_steps:
        # Mostrar cada combinação de termos com x como um Step
        steps.append(
            Step(
                before=build_equation(current_vars, constant_terms),
                after=build_equation(new_vars, constant_terms)
            )
        )
        current_vars = new_vars

    # Simplificar as constantes
    current_consts = constant_terms

    # Se houver mais do que uma constante, anunciar que vamos simplificá-las
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
        # Calcular o divisor comum das constantes actuais (para mostrar explicação relevante)
        divisor = common_divisor_of_constants(current_consts)

        # Mostrar a explicação da divisão ANTES de a fazer, num Step separado,
        # para que o utilizador veja o motivo antes de ver o resultado
        if divisor != 1:
            steps.append(
                Step(
                    before=build_equation(current_vars, current_consts),
                    after=build_equation(current_vars, current_consts),
                    explanation=f"Dividir ambos os lados por {sp.latex(divisor)}"
                )
            )

        # Step com a transição real (sem explicação — já foi mostrada acima)
        steps.append(
            Step(
                before=build_equation(current_vars, current_consts),
                after=build_equation(current_vars, new_consts),
            )
        )
        current_consts = new_consts

    # Extrair o único termo de cada lado depois de toda a simplificação
    final_left  = current_vars[0]
    final_right = current_consts[0]

    # Coeficiente de x e valor constante do lado direito
    coef  = final_left.coeff(x)
    const = final_right

    def safe_gcd(a: sp.Basic, b: sp.Basic) -> sp.Rational:
        """
        Calcula o mdc de dois valores usando sp.Rational para ambos,
        evitando erros de truncagem que ocorreriam com int().
        Necessário para suportar coeficientes fraccionários.
        """
        return sp.gcd(sp.Rational(a), sp.Rational(b))

    def _sympy_stepwise(sym: sp.Basic) -> list[str]:
        """
        Recebe um objecto SymPy (unevaluated, após substituição de x)
        e devolve uma lista de strings LaTeX, uma por cada redução
        aritmética elementar.

        Ao contrário de stepwise_string_eval, trabalha directamente com
        o AST SymPy — nunca converte para string intermédia — evitando
        o problema do safe_sympify não conseguir parsear LaTeX com \\cdot.

        A cada passo, reduz UM sub-nó (Mul ou Add) e reconstrói a
        expressão completa para que o LaTeX mostre sempre o contexto
        total e não apenas o sub-nó isolado.
        """
        # Remover UnevaluatedExpr que possa bloquear a detecção de números
        sym = _unwrap(sym)

        # Se já é um número simples, não há passos a mostrar
        if sym.is_number and not isinstance(sym, (sp.Add, sp.Mul)):
            return [sp.latex(sym)]

        # Primeiro elemento: a expressão tal como está (ordem preservada)
        result = [sp.latex(sym, order='none')]
        current = sym

        def _step_one_mul(expr: sp.Basic) -> tuple[sp.Basic, sp.Basic, sp.Basic] | None:
            """
            Encontra o primeiro Mul numérico na árvore (post-order),
            avalia-o e devolve (expressão_completa_nova, before, after).
            Devolve None se não encontrou nenhum Mul para reduzir.
            """
            if isinstance(expr, sp.Mul) and all(a.is_number for a in expr.args):
                # Este nó é um Mul totalmente numérico — avaliar
                return sp.Mul(*expr.args), expr, sp.Mul(*expr.args)
            # Procurar recursivamente nos filhos
            for i, a in enumerate(expr.args):
                result_inner = _step_one_mul(a)
                if result_inner is not None:
                    reduced, b, af = result_inner
                    # Substituir apenas o filho reduzido, manter o resto
                    new_args = list(expr.args)
                    new_args[i] = reduced
                    # Reconstruir o nó pai sem avaliar
                    return expr.func(*new_args, evaluate=False), b, af
            return None

        def _step_one_add(expr: sp.Basic) -> tuple[sp.Basic, sp.Basic, sp.Basic] | None:
            """
            Encontra o primeiro Add numérico na árvore (post-order),
            avalia-o e devolve (expressão_completa_nova, before, after).
            Devolve None se não encontrou nenhum Add para reduzir.
            """
            if isinstance(expr, sp.Add) and all(a.is_number for a in expr.args):
                # Este nó é um Add totalmente numérico — avaliar
                return sp.Add(*expr.args), expr, sp.Add(*expr.args)
            for i, a in enumerate(expr.args):
                result_inner = _step_one_add(a)
                if result_inner is not None:
                    reduced, b, af = result_inner
                    new_args = list(expr.args)
                    new_args[i] = reduced
                    return expr.func(*new_args, evaluate=False), b, af
            return None

        # Reduzir Muls um a um — cada iteração reduz exactamente um Mul
        # e adiciona a expressão completa resultante à lista de passos
        while True:
            r = _step_one_mul(current)
            if r is None:
                break  # não há mais Muls numéricos para reduzir
            current, _, _ = r
            latex_step = sp.latex(current, order='none')
            if latex_step != result[-1]:
                result.append(latex_step)

        # Reduzir Adds um a um — mesmo processo mas para somas
        while True:
            r = _step_one_add(current)
            if r is None:
                break  # não há mais Adds numéricos para reduzir
            current, _, _ = r
            latex_step = sp.latex(current, order='none')
            if latex_step != result[-1]:
                result.append(latex_step)

        # Garantir que o valor final completamente simplificado é o último passo
        final_latex = sp.latex(sp.simplify(current), order='none')
        if result[-1] != final_latex:
            result.append(final_latex)

        # Remover entradas consecutivas idênticas
        deduped = [result[0]]
        for s in result[1:]:
            if s != deduped[-1]:
                deduped.append(s)
        return deduped

    def _check_solution(
            final_value: sp.Basic, 
            final_latex: str, 
            equation: str, 
            steps: list[Step]
    ) -> None:
        """
        Verifica a solução substituindo o valor encontrado na equação original
        e mostrando cada passo aritmético da simplificação.

        Passos produzidos:
          1. Equação original com explicação "Vamos verificar!"
          2. Equação original com explicação de qual valor vai ser substituído
          3. Equação com x substituído pelo valor (ex: 2(3)+1=7)
          4. Passos aritméticos intercalados dos dois lados (ex: 6+1=7, 7=7)
          5. Mensagem final confirmando se a solução é correcta
        """

        # Converter os dois lados da equação original para objectos SymPy
        left_str, right_str = equation.split("=")
        left_sym  = sp.sympify(normalize_expression(left_str.strip()), evaluate=False)
        right_sym = sp.sympify(normalize_expression(right_str.strip()), evaluate=False)

        # Representação LaTeX da equação original, com ordem dos termos preservada
        equation_latex = f"{sp.latex(left_sym, order='none')} = {sp.latex(right_sym, order='none')}"

        # Mostrar a equação original antes de qualquer operação, com explicação
        steps.append(
            Step(
                before=equation_latex,
                after=equation_latex,
                explanation="Vamos verificar!"
            )
        )

        # Anunciar qual o valor que vai ser substituído
        steps.append(
            Step(
                before=equation_latex,
                after=equation_latex,
                explanation=f"Agora vamos substituir x por {final_latex}"
            )
        )

        # Construir a string LaTeX da substituição directamente sobre a string
        # da equação original — desta forma a ordem dos termos é preservada.
        # Não usamos xreplace aqui porque o SymPy reordena os args do Add
        # quando substitui um símbolo por um número (ex: 2x+1 >> 1+2·3).
        left_latex_orig  = sp.latex(left_sym, order='none')
        right_latex_orig = sp.latex(right_sym, order='none')
        val_latex = sp.latex(final_value)

        def _sub_x_in_latex(latex_str: str, val: str) -> str:
            """
            Substitui 'x' por '(val)' na string LaTeX, usando regex para
            garantir que só substitui 'x' isolado e não parte de outro
            identificador (ex: não substitui o 'x' em 'exp').
            """
            return re.sub(r'(?<![a-zA-Z])x(?![a-zA-Z])', '(' + val + ')', latex_str)

        # Strings LaTeX com x substituído, na ordem visual original
        left_subst_str  = _sub_x_in_latex(left_latex_orig, val_latex)
        right_subst_str = _sub_x_in_latex(right_latex_orig, val_latex)

        # Mostrar a transição da equação original para a expressão substituída
        substituted_display = f"{left_subst_str} = {right_subst_str}"
        steps.append(
            Step(
                before=equation_latex,
                after=substituted_display,
            )
        )

        # Para os passos aritméticos (ex: 2·3 >> 6 >> 7) usamos o AST SymPy,
        # que garante que as reduções são feitas uma a uma correctamente.
        # xreplace com UnevaluatedExpr impede a avaliação imediata.
        left_evaled  = left_sym.xreplace({x: sp.UnevaluatedExpr(final_value)})
        right_evaled = right_sym.xreplace({x: sp.UnevaluatedExpr(final_value)})

        left_steps  = _sympy_stepwise(left_evaled)
        right_steps = _sympy_stepwise(right_evaled)

        # Substituir o primeiro elemento de cada lado pela string com a
        # ordem visual original — o _sympy_stepwise pode reordenar os termos
        if left_steps:
            left_steps[0] = left_subst_str
        if right_steps:
            right_steps[0] = right_subst_str

        # Intercalar os passos dos dois lados em Steps animáveis
        cur_left, cur_right = merge_side_steps(
            left_steps,
            right_steps,
            initial_left=left_subst_str,
            initial_right=right_subst_str,
            steps=steps,
        )

        # Verificar a igualdade final — substituir \cdot por * para o
        # sp.sympify conseguir parsear expressões LaTeX com produto explícito
        left_final  = sp.sympify(cur_left.replace(r'\cdot', '*'), evaluate=True)
        right_final = sp.sympify(cur_right.replace(r'\cdot', '*'), evaluate=True)

        is_true = sp.simplify(left_final - right_final) == 0

        explanation = "The solution is correct!" if is_true else "The solution does not satisfy the equation."

        # Mostrar a confirmação final (equação não muda, só aparece a mensagem)
        steps.append(
            Step(
                before=f"{cur_left} = {cur_right}",
                after=f"{cur_left} = {cur_right}",
                explanation=explanation
            )
        )

    # CASO 1: o coeficiente de x já é 1 (ex: x = 3)
    # Não é preciso dividir — x já está isolado
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

    # CASO 2: o coeficiente de x é diferente de 1 (ex: 2x = 6)
    # É necessário dividir ambos os lados pelo coeficiente
    else:
        # Calcular a solução como racional para preservar frações exactas
        solution = sp.Rational(const, coef)

        # Mostrar a explicação da divisão ANTES de a efectuar,
        # para que o utilizador veja o motivo antes de ver o resultado
        steps.append(
            Step(
                before=f"{sp.latex(final_left)} = {sp.latex(final_right)}",
                after=f"{sp.latex(final_left)} = {sp.latex(final_right)}",
                explanation=f"Dividir ambos os lados por {sp.latex(coef)}"
            )
        )
        # Mostrar a transição real: 2x=6 >> x=3
        steps.append(
            Step(
                before=f"{sp.latex(final_left)} = {sp.latex(final_right)}",
                after=f"x = {sp.latex(solution)}",
            )
        )

        final_value = solution
        final_latex = sp.latex(final_value)

        # Verificar a solução encontrada
        _check_solution(final_value, final_latex, equation, steps)

    return steps


