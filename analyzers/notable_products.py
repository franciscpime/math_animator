from sympy import expand, expand_mul
from sympy.core.mul import Mul
from sympy.core.add import Add


def notable_products(expr, steps, Step):
    """
    Detects and handles notable products.
    If none match, checks for a general binomial product.
    """

    result = square_of_sum(expr, steps, Step)
    if result is not None:
        return result

    result = square_of_difference(expr, steps, Step)
    if result is not None:
        return result

    result = difference_of_squares(expr, steps, Step)
    if result is not None:
        return result

    result = binomial_times_binomial(expr, steps, Step)
    if result is not None:
        return result

    return None


# -----------------------------
# (a+b)^2
# -----------------------------

def square_of_sum(expr, steps, Step):

    if not expr.is_Pow:
        return None

    base, power = expr.args

    if power != 2:
        return None

    if not isinstance(base, Add) or len(base.args) != 2:
        return None

    a, b = base.args

    steps.append(Step(expr="(a+b)^2", explanation="Square of a sum"))
    steps.append(Step(expr="a^2 + 2ab + b^2", explanation="Apply formula"))

    expanded = expand(expr)

    steps.append(Step(expr=str(expanded), explanation="Substitute values"))

    return expanded


# -----------------------------
# (a-b)^2
# -----------------------------

def square_of_difference(expr, steps, Step):

    if not expr.is_Pow:
        return None

    base, power = expr.args

    if power != 2:
        return None

    if not isinstance(base, Add) or len(base.args) != 2:
        return None

    a, b = base.args

    steps.append(Step(expr="(a-b)^2", explanation="Square of a difference"))
    steps.append(Step(expr="a^2 - 2ab + b^2", explanation="Apply formula"))

    expanded = expand(expr)

    steps.append(Step(expr=str(expanded), explanation="Substitute values"))

    return expanded


# -----------------------------
# (a-b)(a+b)
# -----------------------------

def difference_of_squares(expr, steps, Step):

    if not isinstance(expr, Mul) or len(expr.args) != 2:
        return None

    left, right = expr.args

    if not isinstance(left, Add) or not isinstance(right, Add):
        return None

    if len(left.args) != 2 or len(right.args) != 2:
        return None

    steps.append(Step(expr="(a-b)(a+b)", explanation="Difference of squares"))
    steps.append(Step(expr="a^2 - b^2", explanation="Apply formula"))

    expanded = expand(expr)

    steps.append(Step(expr=str(expanded), explanation="Substitute values"))

    return expanded


# -----------------------------
# (a+b)(c+d)  ← NOVO CASO
# -----------------------------

def binomial_times_binomial(expr, steps, Step):

    if not isinstance(expr, Mul) or len(expr.args) != 2:
        return None

    left, right = expr.args

    if not isinstance(left, Add) or not isinstance(right, Add):
        return None

    if len(left.args) != 2 or len(right.args) != 2:
        return None

    m, n = left.args
    o, p = right.args

    # modelo geral
    steps.append(
        Step(
            expr="(m+n)(o+p)",
            explanation="Product of two binomials"
        )
    )

    # distributiva
    steps.append(
        Step(
            expr="(m)(o)+(m)(p)+(n)(o)+(n)(p)",
            explanation="Apply distributive property"
        )
    )

    # substituição
    substituted = f"({m})({o}) + ({m})({p}) + ({n})({o}) + ({n})({p})"

    steps.append(
        Step(
            expr=substituted,
            explanation="Substitute actual values"
        )
    )

    expanded = expand(expr)

    steps.append(
        Step(
            expr=str(expanded),
            explanation="Combine like terms"
        )
    )

    return expanded