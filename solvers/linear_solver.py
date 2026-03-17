import sympy as sp  
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
    new_eq = build_equation(variable_terms, constant_terms)  

    steps.append(
        Step(
            before = equation,  
            after = new_eq  
        )
    )

    # Simplify variables
    current_vars = variable_terms                       # Estado atual dos termos com x
    var_steps = combine_terms_stepwise(variable_terms)  # Passos de simplificação
    
    # Percorre cada passo
    for new_vars in var_steps:  
        before = build_equation(current_vars, constant_terms)  # Antes da simplificação
        after = build_equation(new_vars, constant_terms)        # Depois da simplificação

        steps.append(
            Step(
                before=before,  
                after=after  
            )
        )
        
        # Atualiza estado
        current_vars = new_vars  

    # Simplify constants
    current_consts = constant_terms                         # Estado atual das constantes
    const_steps = combine_terms_stepwise(constant_terms)    # Passos de simplificação

    # Percorre cada passo
    for new_consts in const_steps:      
        before = build_equation(current_vars, current_consts)  
        after = build_equation(current_vars, new_consts)  

        steps.append(
            Step(
                before=before,  
                after=after  
            )
        )
        
        # Atualiza estado
        current_consts = new_consts  

    # Final solve
    final_left = current_vars[0]        # Termo final com x (ex: -10x)
    final_right = current_consts[0]     # Valor final constante (ex: 4)

    coef = final_left.coeff(x)          # Extrai coeficiente de x (ex: -10)
    const = final_right                 # Valor constante

    # PASSO 1: fração não simplificada
    unsimplified = sp.Rational(const, coef)  # Cria a fração (ex: 4/(-10))

    steps.append(
        Step(
            before=f"{sp.latex(final_left)} = {sp.latex(final_right)}",     # Equação antes da divisão
            after=f"x = \\frac{{{sp.latex(const)}}}{{{sp.latex(coef)}}}"    # Resultado como fração
        )
    )

    # PASSO 2: simplificar (se necessário)
    simplified = sp.simplify(unsimplified)  # Simplifica a fração
    
    # Só adiciona passo se houver simplificação
    if simplified != unsimplified:  
        steps.append(
            Step(
                before=f"x = \\frac{{{sp.latex(const)}}}{{{sp.latex(coef)}}}",  # Fração original
                after=f"x = {sp.latex(simplified)}"                             # Fração simplificada
            )
        )

    return steps                   # Devolve todos os passos para animação


