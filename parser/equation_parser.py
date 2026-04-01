import sympy as sp
import re


def normalize_expression(expr):
    # Replace commas with dots: 0,9 >> 0.9
    expr = re.sub(r'(\d),(\d)', r'\1.\2', expr)

    # Replace "^" with "**" 
    expr = expr.replace("^", "**")

    # Insert a "*" between a number and a letter: 3x >> 3*x
    expr = re.sub(r'(\d)([a-zA-Z])', r'\1*\2', expr)

    # Insert a "*" between a letter and an opening parenthesis: x( >> x*(
    expr = re.sub(r'([a-zA-Z])\(', r'\1*(', expr)

    # Insert a "*" between two adjacent parentheses: ()( >> ()*(
    expr = re.sub(r'\)\(', ')*(', expr)

    return expr


# def decimals_to_rationals(expr_str):
#     """
#     Convert every decimal number in an expression string to an exact SymPy
#     Rational so that SymPy never introduces floating-point rounding errors.
#     Ex: '0.9*x + 7' -> 'Rational(9,10)*x + 7'
#     """

#     def replace_decimal(m):
#         s = m.group(0)

#         # The number of decimal places determines the denominator power of ten.
#         dec_part = s.split('.')[1]
#         n_dec = len(dec_part)
#         den = 10 ** n_dec

#         # Remove the decimal point to get the raw integer numerator.
#         num = int(s.replace('.', '').lstrip('0') or '0')

#         # Restore the negative sign if the original number was negative.
#         if s.startswith('-'):
#             num = -int(s.replace('.', '').replace('-', '').lstrip('0') or '0')

#         return f'Rational({num},{den})'

#     # Apply the replacement to every decimal found in the string.
#     return re.sub(r'-?\d+\.\d+', replace_decimal, expr_str)


def parse_equation(equation: str):
    # Split on "=" to get the left and right sides as separate strings.
    left, right = equation.split("=")
    left = left.strip()
    right = right.strip()

    # Normalize both sides so SymPy can parse them correctly.
    left = normalize_expression(left)
    right = normalize_expression(right)

    return left, right


# def latex_to_sympy(expr: str) -> str:
#     """
#     Convert basic LaTeX notation to a syntax that SymPy's sympify can parse.
#     Only handles the subset of LaTeX that appears in this project
#     (mainly \\frac{a}{b} and backslash commands).
#     """
#     # Turn \frac{a}{b} into (a)/(b) which sympify understands.
#     expr = re.sub(r'\\frac\{([^}]+)\}\{([^}]+)\}', r'(\1)/(\2)', expr)

#     # Strip any remaining backslash commands that sympify cannot handle.
#     expr = expr.replace("\\", "")

#     return expr.strip()

######################## no being used
def fix_implicit_mul(expr: str):
    # Insert "*" between a digit and an opening parenthesis so
    # "10(3/4)" is treated as "10*(3/4)" by SymPy.
    return re.sub(r'(\d)\(', r'\1*(', expr)


def safe_sympify(expr: str) -> sp.Basic:
    """
    Try to parse a LaTeX or plain-math string into a SymPy expression.
    First attempts SymPy's LaTeX parser; if that fails, falls back to
    converting the expression manually and calling sympify directly.
    Returns sp.nan if both attempts fail so the caller can handle the error.
    """
    from sympy.parsing.latex import parse_latex as _parse_latex

    expr = expr.strip()

    # Attempt 1: use SymPy's built-in LaTeX parser.
    try:
        parsed = _parse_latex(expr)
        return sp.sympify(parsed, evaluate=False)
    except Exception:
        pass

    # Attempt 2: manually convert LaTeX to sympify-compatible syntax.
    try:
        cleaned = latex_to_sympy(expr)
        cleaned = fix_implicit_mul(cleaned)
        return sp.sympify(cleaned, evaluate=False)
    except Exception:
        return sp.nan


def detect_raw_fractions(expr_str):
    """
    Find every fraction written as a/b in the original expression string
    (before any normalization) and return a list of tuples:
      (numerator_str, denominator_str, Rational value)
    Ex: '1/2 + 3x' -> [('1', '2', Rational(1, 2))]
    This is used by solve_linear to know which fractions to display and
    potentially simplify before solving.
    """
    fracs = []

    for m in re.finditer(r'(-?\d+)/(\d+)', expr_str):
        num = int(m.group(1))
        den = int(m.group(2))

        # Skip division by zero just in case.
        if den != 0:
            fracs.append((m.group(1), m.group(2), sp.Rational(num, den)))

    return fracs


def detect_decimals(expr_str):
    """
    Find every decimal number written as a.b or a,b in the original expression
    string and return a list of tuples:
      (original_decimal_str, Rational value)
    Ex: '0.5x + 1' -> [('0.5', Rational(1, 2))]
    This is used by solve_linear to convert decimals to fractions before solving,
    so the animation can show the conversion step by step.
    """
    # Normalise commas to dots first so both "0,5" and "0.5" are detected.
    s = re.sub(r'(\d),(\d)', r'\1.\2', expr_str)
    decimals = []

    for m in re.finditer(r'-?\d+\.\d+', s):
        d_str = m.group(0)

        # The number of decimal places determines the denominator.
        dec_part = d_str.lstrip('-').split('.')[1]
        n_dec = len(dec_part)
        den = 10 ** n_dec

        # Remove the decimal point and sign to get the raw integer numerator.
        num_str = d_str.replace('.', '').replace('-', '')
        num = int(num_str)

        # Restore the negative sign if the original decimal was negative.
        if d_str.startswith('-'):
            num = -num

        decimals.append((d_str, sp.Rational(num, den)))

    return decimals

