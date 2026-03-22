import sympy as sp
import re

def normalize_expression(expr):
    expr = expr.replace("^", "**")

    expr = re.sub(r'(\d)([a-zA-Z])', r'\1*\2', expr)
    expr = re.sub(r'([a-zA-Z])\(', r'\1*(', expr)
    expr = re.sub(r'\)([a-zA-Z0-9])', r')*\1', expr)
    expr = re.sub(r'\)\(', ')*(', expr)

    return expr


def expand_multiplications(expr: str):
    matches = re.findall(r'(\d+)\((\-?\d+)\)', expr)

    if not matches:
        return None

    new_expr = expr

    for a, b in matches:
        result = int(a) * int(b)
        new_expr = new_expr.replace(f"{a}({b})", str(result))

    return new_expr


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
    # 10(3/4) → 10*(3/4)
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
        return sp.sympify(cleaned, evaluate=False)  # 🔥 AQUI ESTÁ O FIX
    except Exception:
        return sp.nan