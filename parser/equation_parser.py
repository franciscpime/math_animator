import sympy as sp
import re


def normalize_expression(expr):
    # Normalizar vírgula decimal para ponto (ex: 0,9 -> 0.9)
    expr = re.sub(r'(\d),(\d)', r'\1.\2', expr)
    expr = expr.replace("^", "**")
    expr = re.sub(r'(\d)([a-zA-Z])', r'\1*\2', expr)
    expr = re.sub(r'([a-zA-Z])\(', r'\1*(', expr)
    expr = re.sub(r'\)\(', ')*(', expr)
    return expr


def decimals_to_rationals(expr_str):
    """
    Converte decimais numa string de expressão para racionais exactos.
    Ex: '0.9*x + 7' -> 'Rational(9,10)*x + 7'
    Usado internamente para que o SymPy trabalhe com racionais.
    """
    def replace_decimal(m):
        s = m.group(0)
        dec_part = s.split('.')[1]
        n_dec = len(dec_part)
        den = 10 ** n_dec
        num = int(s.replace('.', '').lstrip('0') or '0')
        if s.startswith('-'):
            num = -int(s.replace('.', '').replace('-', '').lstrip('0') or '0')
        return f'Rational({num},{den})'
    return re.sub(r'-?\d+\.\d+', replace_decimal, expr_str)


def parse_equation(equation: str):
    left, right = equation.split("=")
    left = left.strip()
    right = right.strip()
    left = normalize_expression(left)
    right = normalize_expression(right)
    return left, right


def latex_to_sympy(expr: str) -> str:
    """Converte notação LaTeX básica para sintaxe que o sympify consegue parsear."""
    expr = re.sub(r'\\frac\{([^}]+)\}\{([^}]+)\}', r'(\1)/(\2)', expr)
    expr = expr.replace("\\", "")
    return expr.strip()


def fix_implicit_mul(expr: str):
    # 10(3/4) -> 10*(3/4)
    return re.sub(r'(\d)\(', r'\1*(', expr)


def safe_sympify(expr: str) -> sp.Basic:
    from sympy.parsing.latex import parse_latex as _parse_latex
    expr = expr.strip()
    try:
        parsed = _parse_latex(expr)
        return sp.sympify(parsed, evaluate=False)
    except Exception:
        pass
    try:
        cleaned = latex_to_sympy(expr)
        cleaned = fix_implicit_mul(cleaned)
        return sp.sympify(cleaned, evaluate=False)
    except Exception:
        return sp.nan


def detect_raw_fractions(expr_str):
    """
    Detecta frações escritas como a/b na expressão original (antes de normalizar).
    Devolve lista de (numerador_str, denominador_str, valor_Rational).
    Ex: '1/2 + 3x' -> [('1', '2', Rational(1,2))]
    """
    fracs = []
    for m in re.finditer(r'(-?\d+)/(\d+)', expr_str):
        num = int(m.group(1))
        den = int(m.group(2))
        if den != 0:
            fracs.append((m.group(1), m.group(2), sp.Rational(num, den)))
    return fracs


def detect_decimals(expr_str):
    """
    Detecta decimais escritos como a.b ou a,b na expressão original.
    Devolve lista de (decimal_str_original, valor_Rational).
    """
    s = re.sub(r'(\d),(\d)', r'\1.\2', expr_str)
    decimals = []
    for m in re.finditer(r'-?\d+\.\d+', s):
        d_str = m.group(0)
        dec_part = d_str.lstrip('-').split('.')[1]
        n_dec = len(dec_part)
        den = 10 ** n_dec
        num_str = d_str.replace('.', '').replace('-', '')
        num = int(num_str)
        if d_str.startswith('-'):
            num = -num
        decimals.append((d_str, sp.Rational(num, den)))
    return decimals

