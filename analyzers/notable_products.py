from sympy import Wild

a = Wild("a")
b = Wild("b")

PATTERNS = {
    "square_sum": (a + b) ** 2,
    "square_diff": (a - b) ** 2,
    "sum_times_diff": (a + b) * (a - b),
    "sum_times_sum": (a + b) * (a + b),
}

def detect_notable(expr):
    """
    Detecta casos notáveis numa expressão.
    Retorna (tipo, match_dict)
    """

    for name, pattern in PATTERNS.items():

        match = expr.match(pattern)

        if match:
            return name, match

    return None, None