from sympy import expand
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
    expanded = expand(expr)

    steps.append(Step(
        before=str(expr),
        after="(a+b)^2",
        explanation="Square of a sum"
    ))
    steps.append(Step(
        before="(a+b)^2",
        after="a^2 + 2ab + b^2",
        explanation="Apply formula"
    ))
    steps.append(Step(
        before="a^2 + 2ab + b^2",
        after=str(expanded),
        explanation="Substitute values"
    ))

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
    expanded = expand(expr)

    steps.append(Step(
        before=str(expr),
        after="(a-b)^2",
        explanation="Square of a difference"
    ))
    steps.append(Step(
        before="(a-b)^2",
        after="a^2 - 2ab + b^2",
        explanation="Apply formula"
    ))
    steps.append(Step(
        before="a^2 - 2ab + b^2",
        after=str(expanded),
        explanation="Substitute values"
    ))

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

    expanded = expand(expr)

    steps.append(Step(
        before=str(expr),
        after="(a-b)(a+b)",
        explanation="Difference of squares"
    ))
    steps.append(Step(
        before="(a-b)(a+b)",
        after="a^2 - b^2",
        explanation="Apply formula"
    ))
    steps.append(Step(
        before="a^2 - b^2",
        after=str(expanded),
        explanation="Substitute values"
    ))

    return expanded


# -----------------------------
# (a+b)(c+d)
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

    substituted = f"({m})({o}) + ({m})({p}) + ({n})({o}) + ({n})({p})"
    expanded = expand(expr)

    steps.append(Step(
        before=str(expr),
        after="(m+n)(o+p)",
        explanation="Product of two binomials"
    ))
    steps.append(Step(
        before="(m+n)(o+p)",
        after="(m)(o)+(m)(p)+(n)(o)+(n)(p)",
        explanation="Apply distributive property"
    ))
    steps.append(Step(
        before="(m)(o)+(m)(p)+(n)(o)+(n)(p)",
        after=substituted,
        explanation="Substitute actual values"
    ))
    steps.append(Step(
        before=substituted,
        after=str(expanded),
        explanation="Combine like terms"
    ))

    return expanded

