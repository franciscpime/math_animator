import sympy as sp  
from sympy import lcm
import re  
from models.step import Step  
from parser.equation_parser import parse_equation  

# Define a variável simbólica x que será usada nas equações
x = sp.symbols("x")  

# Função que recebe uma expressão (string) e devolve os termos separados
def extract_terms(expr: str):  
    # Remove todos os espaços para facilitar o processamento
    expr = expr.replace(" ", "")  
    # Divide a expressão em termos (ex: "2x-3+x" >> ["2x", "-3", "+x"])
    terms = re.findall(r'[+-]?[^+-]+', expr)  
    
    # Lista onde vamos guardar os termos convertidos para objetos SymPy
    sympy_terms = []
    
    # Percorre cada termo encontrado
    for term in terms:  
        if term in ["+", "-"]:                  # Ignora símbolos soltos (casos raros de parsing)
            continue                            # Passa ao próximo termo

        sympy_terms.append(sp.sympify(term))    # Converte o termo para SymPy (ex: "2x" >> 2*x)

    return sympy_terms                          # Devolve a lista de termos como objetos SymPy


# Função que transforma uma lista de termos numa string LaTeX
def render_terms(terms):  
    
    # Converte cada termo para LaTeX (ex: 2*x >> "2x")
    parts = [sp.latex(t) for t in terms]  
    
    # Junta os termos com " + " (ex: "2x + -3")
    expr = " + ".join(parts)  

    # Corrige sinais (ex: "2x + -3" >> "2x - 3")
    expr = expr.replace("+ -", "- ")  

    return expr                                 # Devolve a expressão em formato LaTeX


# Constrói uma equação completa a partir de duas listas de termos
def build_equation(left_terms, right_terms):  

    left = render_terms(left_terms)         # Converte os termos do lado esquerdo para LaTeX
    right = render_terms(right_terms)       # Converte os termos do lado direito para LaTeX

    return f"{left} = {right}"              # Junta os dois lados com "="


# Função que combina termos passo a passo (sem simplificar tudo de uma vez)
def combine_terms_stepwise(terms):
    
    # Cria uma cópia da lista para não alterar a original
    new_terms = terms.copy()  
    steps = []                  # Lista onde vamos guardar cada passo intermédio
    
    # Continua enquanto houver mais do que um termo
    while len(new_terms) > 1:  

        t1 = new_terms[-2]          # Penúltimo termo da lista
        t2 = new_terms[-1]          # Último termo da lista
        
        # Se o termo tiver a variável x (termo algébrico)
        if t1.has(x):  
            coef = t1.coeff(x) + t2.coeff(x)    # Soma os coeficientes de x (ex: 2x + -3x >> -1x)
            new_term = coef * x                 # Cria o novo termo combinado
        else:
            new_term = t1 + t2                  # Soma diretamente constantes (ex: 4 + -7 >> -3)

        new_terms.pop()                         # Remove o último termo usado
        new_terms.pop()                         # Remove o penúltimo termo usado
        new_terms.append(new_term)              # Adiciona o novo termo combinado

        steps.append(new_terms.copy())          # Guarda o estado atual da lista como um passo

    return steps                                # Devolve todos os passos de simplificação


def substitution_steps(expr, value):
    steps = []

    # PASSO 1: substituição sem avaliar tudo
    expr_sub = expr.subs(x, sp.Integer(value), evaluate=False)
    steps.append(expr_sub)

    # PASSO 2: expandir multiplicações (ex: 10*(5) → 50)
    expanded = sp.expand(expr_sub)
    if expanded != expr_sub:
        steps.append(expanded)

    # PASSO 3: simplificar tudo (ex: 50 + 20 → 70)
    final = sp.simplify(expanded)
    if final != expanded:
        steps.append(final)

    return steps


def detailed_multiplication(expr):
    steps = []

    if isinstance(expr, sp.Mul):
        args = expr.args

        # Ex: 12 * (-1/12)
        nums = []
        dens = []

        for a in args:
            if isinstance(a, sp.Rational):
                nums.append(a.p)
                dens.append(a.q)
            else:
                nums.append(a)
                dens.append(1)

        # PASSO 1: forma explícita (-1*12/12)
        num_expr = sp.Mul(*nums)
        den_expr = sp.Mul(*dens)

        step1 = sp.Mul(num_expr, sp.Pow(den_expr, -1))
        steps.append(step1)

        # PASSO 2: numerador resolvido (-12/12)
        step2 = sp.together(step1)
        steps.append(step2)

        # PASSO 3: simplificado (-1)
        step3 = sp.simplify(step2)
        steps.append(step3)

    return steps


# Função principal que resolve equações lineares passo a passo
def solve_linear(equation: str):  
    
    # Separa a equação em lado esquerdo e direito
    left, right = parse_equation(equation)  

    steps = []                                  # Lista que vai guardar todos os passos da resolução

    # Inicio
    steps.append(
        Step(
            before = equation,  
            after = equation,  
            explanation = "Organise terms"  
        )
    )

    # Extract terms
    left_terms = extract_terms(left)        # Extrai termos do lado esquerdo
    right_terms = extract_terms(right)      # Extrai termos do lado direito

    left_x, left_const = [], []             # Listas para variáveis e constantes do lado esquerdo
    right_x, right_const = [], []           # Listas para variáveis e constantes do lado direito

    for t in left_terms:                    # Percorre termos do lado esquerdo
        if t.has(x):                        # Se tiver x
            left_x.append(t)                # Vai para lista de variáveis
        else:
            left_const.append(t)            # Vai para lista de constantes

    for t in right_terms:                   # Percorre termos do lado direito
        if t.has(x):                        # Se tiver x
            right_x.append(t)               # Vai para lista de variáveis
        else:
            right_const.append(t)           # Vai para lista de constantes

    # Move termos com x para a esquerda (mudando o sinal)
    variable_terms = left_x + [-t for t in right_x]       

    # Move constantes para a direita (mudando o sinal)
    constant_terms = right_const + [-t for t in left_const]

    # Cria a equação após reorganização
    # Cria a equação após reorganização
    new_eq = build_equation(variable_terms, constant_terms)  

    # Mensagem inicial (só lado esquerdo)
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

    # Mensagem lado direito
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
    final_left = current_vars[0]        # Termo final com x (ex: -10x)
    final_right = current_consts[0]     # Valor final constante (ex: 4)

    coef = final_left.coeff(x)
    const = final_right

    # CASO 1: já está resolvido (tipo x = 3)
    if coef == 1:
        steps.append(
            Step(
                before=f"{sp.latex(final_left)} = {sp.latex(final_right)}",
                after=f"x = {sp.latex(const)}"
            )
        )

    # CASO 2: precisa dividir (tipo 12x = -1)
    else:
        num_raw = const
        den_raw = coef

        sign = "-" if (num_raw * den_raw) < 0 else ""

        numerator = abs(int(num_raw))
        denominator = abs(int(den_raw))

        divisor = sp.gcd(numerator, denominator)

        unsimplified_latex = f"{sign}\\frac{{{numerator}}}{{{denominator}}}"

        raw_fraction = sp.Rational(const, coef)

        # PASSO 1: fração simplificada mas ainda como fração (5/1)
        reduced_num = raw_fraction.p
        reduced_den = raw_fraction.q

        fraction_latex = f"\\frac{{{reduced_num}}}{{{reduced_den}}}"

        # PASSO 2: resultado final
        final_value = reduced_num if reduced_den == 1 else raw_fraction
        final_latex = sp.latex(final_value)

        # PASSO divisão
        steps.append(
            Step(
                before=f"{sp.latex(final_left)} = {sp.latex(final_right)}",
                after=f"x = {unsimplified_latex}",
                explanation=f"Let's divide by {coef}"
            )
        )

        # PASSO simplificação (se necessário)
        if fraction_latex != unsimplified_latex:
            steps.append(
                Step(
                    before=f"x = {unsimplified_latex}",
                    after=f"x = {fraction_latex}"
                )
            )

        if reduced_den == 1:
            steps.append(
                Step(
                    before=f"x = {fraction_latex}",
                    after=f"x = {final_latex}"
                )
            )
            
        # --- CHECK SOLUTION ---

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

        substituted = equation.replace("x", f"({final_value})")

        steps.append(
            Step(
                before=equation,
                after=substituted
            )
        )

        # -------- STRING-BASED STEP SOLVER --------

        def stepwise_string_eval(expr):
            steps_local = []

            current = expr

            # PASSO 1: multiplicações tipo 10(5)
            mult_pattern = r'(\d+)\((\-?\d+)\)'

            while re.search(mult_pattern, current):
                new = re.sub(
                    mult_pattern,
                    lambda m: str(int(m.group(1)) * int(m.group(2))),
                    current
                )
                steps_local.append(new)
                current = new

            # PASSO 2: soma final
            try:
                result = str(eval(current))
                if result != current:
                    steps_local.append(result)
            except:
                pass

            return steps_local

        # aplicar aos dois lados
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

        # -------- VERIFICAÇÃO FINAL COM SYMPY --------

        left_final = sp.sympify(current_left)
        right_final = sp.sympify(current_right)

        is_true = sp.simplify(left_final - right_final) == 0

        explanation = "A solução está correta" if is_true else "A solução não verifica a equação"

        steps.append(
            Step(
                before=f"{current_left} = {current_right}",
                after=f"{current_left} = {current_right}",
                explanation=explanation
            )
        )

    
    return steps            # Devolve todos os passos para animação

