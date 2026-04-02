import sympy as sp
from models.step import Step


def rational_coef_solve_steps(coef_rational, const_rational, final_left_latex, final_right_latex):
    """
    Generate the animation steps for solving (p/q)*x = c when q > 1.

    The strategy is to first clear the denominator by multiplying both sides
    by q, then isolate x by dividing by p.

    Example for -x/5 = 25/2:
      [explanation]  Multiply both sides by 5
      >> -x = 25 * 5 / 2
      >> -x = 125/2
      [explanation]  Divide both sides by -1
      >> x = 125 / -2
      >> x = -125/2

    Parameters
    ----------
    coef_rational     : sp.Rational  -- coefficient of x, e.g. Rational(-1, 5)
    const_rational    : sp.Rational  -- right-hand side,   e.g. Rational(25, 2)
    final_left_latex  : str          -- LaTeX of the left side before solving
    final_right_latex : str          -- LaTeX of the right side before solving

    Returns
    -------
    (list[Step], sp.Rational)  -- animation steps and the numeric solution
    """
    x = sp.symbols("x")
    result_steps = []

    coef_numerator = coef_rational.p
    coef_denominator = coef_rational.q
    right_side_constant = const_rational

    eq_before_multiply = f"{final_left_latex} = {final_right_latex}"

    result_steps.append(
        Step(
            before=eq_before_multiply,
            after=eq_before_multiply,
            explanation=f"Multiply both sides by {coef_denominator}",
        )
    )

    # Step 1: multiply both sides by q to clear the fraction
    left_without_den = sp.latex(sp.Integer(coef_numerator) * x)

    costante_numerator = right_side_constant.p
    costante_denominator = right_side_constant.q

    if costante_numerator < 0:
        cn_str = f"({costante_numerator})"
    else:
        cn_str = str(costante_numerator)

    if costante_denominator == 1:
        right_side_product_unevaluated = f"{cn_str} \\cdot {coef_denominator}"
    else:
        right_side_product_unevaluated = f"\\frac{{{cn_str} \\cdot {coef_denominator}}}{{{costante_denominator}}}"

    eq_after_multiply_unevaluated = f"{left_without_den} = {right_side_product_unevaluated}"

    result_steps.append(Step(before=eq_before_multiply, after=eq_after_multiply_unevaluated))

    numerator_product = costante_numerator * coef_denominator

    if costante_denominator == 1:
        right_side_product_evaluated = str(numerator_product)
    else:
        right_side_product_evaluated = sp.latex(sp.Rational(numerator_product, costante_denominator))

    eq_after_multiply_evaluated = f"{left_without_den} = {right_side_product_evaluated}"

    if eq_after_multiply_evaluated != eq_after_multiply_unevaluated:
        result_steps.append(Step(before=eq_after_multiply_unevaluated, after=eq_after_multiply_evaluated))
    else:
        eq_after_multiply_evaluated = eq_after_multiply_unevaluated

    result_steps.append(
        Step(
            before=eq_after_multiply_evaluated,
            after=eq_after_multiply_evaluated,
            explanation=f"Divide both sides by {coef_numerator}",
        )
    )

    # Step 2: divide by the numerator p to isolate x
    solution = sp.Rational(numerator_product, costante_denominator * coef_numerator)

    if costante_denominator == 1:
        right_side_divided = f"\\frac{{{numerator_product}}}{{{coef_numerator}}}"
    else:
        right_side_divided = f"\\frac{{{numerator_product}}}{{{costante_denominator * coef_numerator}}}"

    eq_after_divide_unevaluated = f"x = {right_side_divided}"

    result_steps.append(Step(before=eq_after_multiply_evaluated, after=eq_after_divide_unevaluated))

    eq_solution_simplified = f"x = {sp.latex(solution)}"

    if eq_solution_simplified != eq_after_divide_unevaluated:
        result_steps.append(Step(before=eq_after_divide_unevaluated, after=eq_solution_simplified))

    return result_steps, solution

