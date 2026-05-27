import sympy as sp
import re

'''
This function converts mathematical expressions into a normalized form
that can be more easily parsed by SymPy.
'''
def normalize_expression(expr):
    # Replace commas with dots: 0,9 >> 0.9
    expr = re.sub(r'(\d),(\d)', r'\1.\2', expr)

    expr = re.sub(r'-?\d+\.\d+', lambda m: str(sp.Rational(m.group(0))), expr)

    # Replace "^" with "**"
    expr = expr.replace("^", "**")

    # Insert a "*" between a number and a letter: 3x >> 3*x
    expr = re.sub(r'(\d)([a-zA-Z])', r'\1*\2', expr)

    # Insert a "*" between a letter and an opening parenthesis: x( >> x*(
    expr = re.sub(r'([a-zA-Z])\(', r'\1*(', expr)

    # Insert a "*" between two adjacent parentheses: ()( >> ()*(
    expr = re.sub(r'\)\(', ')*(', expr)

    return expr


'''
This function parses a mathematical equation and returns 
the left and right sides as SymPy expressions.

Example: 2*x + 3 = 4*x - 5
Left side: 2*x + 3
Right side: 4*x - 5
'''
def parse_equation(equation: str):

    left, right = equation.split("=")
    left = left.strip()
    right = right.strip()

    # Normalize both sides so SymPy can parse them correctly.
    left = normalize_expression(left)
    right = normalize_expression(right)

    return left, right


'''
This function fixes implicit multiplication in a mathematical expression.

Example: 10(3/4) becomes 10*(3/4)
'''
def fix_implicit_mul(expr: str):

    # Insert "*" between a digit and an opening parenthesis so
    return re.sub(r'(\d)\(', r'\1*(', expr)


'''
This function safely converts a string expression into a SymPy expression.

First - it tries to parse the expression using SymPy's built-in LaTeX parser.
Second - if that fails, it attempts to fix implicit multiplication and call sympify directly.
Finally - if both attempts fail, it returns sp.nan, which means 
the expression could not be parsed.
'''
def safe_sympify(expression: str):

    from sympy.parsing.latex import parse_latex as _parse_latex

    expression = expression.strip()

    # Try parsing with LaTeX
    try:
        parsed = _parse_latex(expression)
        return sp.sympify(parsed, evaluate=False)
    except Exception:
        pass

    # Try fixing implicit multiplication and call sympify directly.
    try:
        cleaned = fix_implicit_mul(expression)
        return sp.sympify(cleaned, evaluate=False)
    except Exception:
        return sp.nan


'''
This function detects raw fractions in a given expression string.

Example: '1/2 + 3x' -> [('1', '2', Rational(1, 2))]
'''
def detect_raw_fractions(expr_str):
    # Dictionary to hold detected fractions
    fracs = []

    # Find all matches of the fraction pattern
    for match in re.finditer(r'(-?\d+)/(\d+)', expr_str):
        numerator = int(match.group(1))
        denominator = int(match.group(2))

        # Skip zero denominators
        if denominator != 0:
            fracs.append(
                (
                    match.group(1),
                    match.group(2),
                    sp.Rational(numerator, denominator)
                )
            )

    return fracs


'''
This function detects decimal numbers in a given expression string.

Example: '0.5 + 1.2x' -> [('0.5', Rational(1, 2)), ('1.2', Rational(6, 5))]
'''
def detect_decimals(expr_str):
    # Dictionary to hold detected decimals
    decimals = []

    # Normalize the expression string
    normalized_expression = re.sub(r'(\d),(\d)', r'\1.\2', expr_str)
    normalized_expression = normalized_expression.replace(" ", "")

    # Find all matches of the decimal pattern
    for match in re.finditer(r'-?\d+\.\d+', normalized_expression):
        decimal_raw_str = match.group(0)

        # Get the decimal part and calculate the denominator
        decimal_part = decimal_raw_str.lstrip('-').split('.')[1]
        number_of_decimal_places = len(decimal_part)
        denominator = 10 ** number_of_decimal_places

        # Create the raw numerator string
        # Example: '-0.5' -> '05'
        raw_numerator_str = decimal_raw_str.replace('.', '').replace('-', '')
        num = int(raw_numerator_str)

        # Normalize the numerator
        if decimal_raw_str.startswith('-'):
            num = -num

        decimals.append(
            (
                decimal_raw_str, 
                sp.Rational(num, denominator)
            )
        )

    return decimals

"""
Detect if the equation contains a division by zero.
Returns (numerator_str, sign) if found, None otherwise.

Example: '12x + 4/0 = 3'     >>  ('4', '+')
            '12x - 4/0 = 3'  >>  ('4', '-')
            '4/0 = 3'        >>  ('4', '+')
            '12x + 3 = 5'    >>  None
"""
def detect_division_by_zero(equation: str):
    
    match = re.search(r'([+-])?\s*(\d+)\s*/\s*0', equation)
    
    if match:
        sign = match.group(1) if match.group(1) else '+'
        numerator = match.group(2)

        return numerator, sign
    
    return None

