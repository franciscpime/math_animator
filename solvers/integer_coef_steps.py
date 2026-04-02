import sympy as sp

from models.step import Step
from math_utils.gcd_utils import safe_gcd

x = sp.symbols("x")


def integer_coef_solve_steps(coefficient_rational, constant_rational, final_left_latex, final_right_latex):
    """
    Generate the animation steps for solving n*x = c when n is a non-zero
    integer (i.e. the coefficient of x has no fractional part).

    Two sub-cases are handled:

    Sub-case A -- two-stage division (intermediate simplification possible):
      When the GCD of |c| and |n| is greater than 1 but does not equal |n|,
      and both simplified values are still whole numbers, we first divide by
      the GCD to reduce both sides, then divide by the remaining coefficient.

      Example: 12x = 8
        Divide both sides by 4  >>  3x = 2
        Divide both sides by 3  >>  x = 2/3

    Sub-case B -- direct division:
      In all other cases we divide directly by n, showing the unreduced
      fraction first and then the SymPy-simplified result.

      Example: 15x = 3
        Divide both sides by 15  >>  x = 3/15  >>  x = 1/5

    Parameters
    ----------
    coefficient_rational : sp.Rational -- coefficient of x (integer, q == 1)
    constant_rational    : sp.Rational -- right-hand side constant
    final_left_latex     : str         -- LaTeX of the left side before solving
    final_right_latex    : str         -- LaTeX of the right side before solving

    Returns
    -------
    (list[Step], sp.Rational) -- animation steps and the exact solution
    """
    result_steps = []

    common_divisor   = safe_gcd(abs(constant_rational),    abs(coefficient_rational))
    simplified_coef  = coefficient_rational  / common_divisor
    simplified_const = constant_rational / common_divisor

    # Determine whether a two-stage division is worth showing.
    coef_is_whole  = isinstance(simplified_coef,  sp.Integer) or simplified_coef.q  == 1
    const_is_whole = isinstance(simplified_const, sp.Integer) or simplified_const.q == 1

    has_intermediate_step = (
        common_divisor > 1
        and common_divisor != abs(coefficient_rational)
        and coef_is_whole
        and const_is_whole
        and simplified_coef != 1
    )

    if has_intermediate_step:
        # Sub-case A: two-stage division.
        left_after_division  = simplified_coef * x
        right_after_division = simplified_const

        # Step 1: announce the first division (by the common divisor).
        result_steps.append(
            Step(
                before=f"{final_left_latex} = {final_right_latex}",
                after=f"{final_left_latex} = {final_right_latex}",
                explanation=f"Divide both sides by {sp.latex(common_divisor)}",
            )
        )

        # Step 2: show the equation after dividing by the common divisor.
        result_steps.append(
            Step(
                before=f"{final_left_latex} = {final_right_latex}",
                after=f"{sp.latex(left_after_division)} = {sp.latex(right_after_division)}",
            )
        )

        # Step 3: announce the second division (by the remaining coefficient).
        result_steps.append(
            Step(
                before=f"{sp.latex(left_after_division)} = {sp.latex(right_after_division)}",
                after=f"{sp.latex(left_after_division)} = {sp.latex(right_after_division)}",
                explanation=f"Divide both sides by {sp.latex(simplified_coef)}",
            )
        )

        # Step 4: show the final isolated x.
        solution = sp.Rational(constant_rational, coefficient_rational)

        result_steps.append(
            Step(
                before=f"{sp.latex(left_after_division)} = {sp.latex(right_after_division)}",
                after=f"x = {sp.latex(solution)}",
            )
        )

    else:
        # Sub-case B: direct division.
        solution = sp.Rational(constant_rational, coefficient_rational)

        unreduced_fraction = (
            r"\frac{"
            + str(int(constant_rational))
            + r"}{"
            + str(int(coefficient_rational))
            + r"}"
        )

        reduced_fraction = sp.latex(solution)

        # Step 1: announce the division.
        result_steps.append(
            Step(
                before=f"{final_left_latex} = {final_right_latex}",
                after=f"{final_left_latex} = {final_right_latex}",
                explanation=f"Divide both sides by {sp.latex(coefficient_rational)}",
            )
        )

        # Step 2: show x = unreduced fraction.
        result_steps.append(
            Step(
                before=f"{final_left_latex} = {final_right_latex}",
                after=f"x = {unreduced_fraction}",
            )
        )

        # Step 3: simplify the fraction if possible.
        if reduced_fraction != unreduced_fraction:
            result_steps.append(
                Step(
                    before=f"x = {unreduced_fraction}",
                    after=f"x = {reduced_fraction}",
                )
            )

    return result_steps, solution

