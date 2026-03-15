import re

def analyze_equation(equation: str):
    """
    Normalize the equation and return left and right sides.

    Responsibilities:
    - remove spaces
    - convert implicit multiplication (10x -> 10*x)
    - split equation into left and right expressions
    """

    equation = equation.replace(" ", "")

    # Convert implicit multiplication: 10x -> 10*x
    equation = re.sub(r'(\d+/\d+)(x)', r'\1*x', equation)
    equation = re.sub(r'(\d)(x)', r'\1*x', equation)
    # NEW: (x-3)(x+3) -> (x-3)*(x+3)
    equation = re.sub(r'\)\(', r')*(', equation)

    if "=" not in equation:
        raise ValueError("Invalid equation: '=' not found.")

    left, right = equation.split("=")

    return left, right
