import sympy as sp

from models.step import Step
from parser.equation_parser import (
    parse_equation,
    detect_decimals,
    detect_raw_fractions,
)
from utils.latex_helpers import equation_to_latex_display, decimal_str
from utils.simplification_steps import fraction_simplification_steps, decimal_simplification_steps
from utils.term_rearranger import rearrange_terms
from utils.term_combiner import combine_terms_stepwise
from utils.solution_checker import check_solution
from utils.equation_builder import build_equation
from solvers.rational_coef_steps import rational_coef_solve_steps
from solvers.integer_coef_steps import integer_coef_solve_steps

x = sp.symbols("x")


def solve_linear(equation: str):
    """
    Solve a linear equation in one variable (x) and return the full list of
    animation Steps, each representing a single visible change in the equation.

    The function handles:
      - Decimal coefficients  (converted to fractions first)
      - Unreduced fractions   (simplified before solving)
      - Multiple variable and constant terms on both sides
      - Rational (fractional) coefficients of x
      - Verification of the solution by back-substitution
    """
    left, right = parse_equation(equation)
    steps = []

    raw_decimals  = detect_decimals(equation)
    raw_fractions = detect_raw_fractions(equation)

    equation_display   = equation_to_latex_display(equation)
    current_eq_display = equation_display

    # ------------------------------------------------------------------
    # Pre-solve step 1: convert decimal coefficients to fractions.
    # Each decimal is shown as an unreduced fraction first, then reduced.
    # ------------------------------------------------------------------
    for dec_str, _dec_val in raw_decimals:
        dec_steps = decimal_simplification_steps(dec_str)

        for i in range(1, len(dec_steps)):
            before_eq = current_eq_display
            after_eq  = before_eq.replace(dec_steps[i - 1], dec_steps[i], 1)

            if before_eq != after_eq:
                if i == 1:
                    current_explanation = "Convert decimal to fraction"
                else:
                    current_explanation = "Simplify fraction"

                steps.append(
                    Step(
                        before=before_eq,
                        after=after_eq,
                        explanation=current_explanation,
                    )
                )

                current_eq_display = after_eq

    # ------------------------------------------------------------------
    # Pre-solve step 2: simplify unreduced fractions (e.g. 6/4 >> 3/2).
    # ------------------------------------------------------------------
    for num_s, den_s, _frac_val in raw_fractions:
        frac_steps    = fraction_simplification_steps(num_s, den_s)
        frac_orig_str = num_s + "/" + den_s

        # First show the a/b text as \frac{a}{b} before reducing.
        if frac_orig_str in current_eq_display:
            eq_with_frac = current_eq_display.replace(frac_orig_str, frac_steps[0], 1)

            if eq_with_frac != current_eq_display:
                steps.append(
                    Step(
                        before=current_eq_display,
                        after=eq_with_frac,
                    )
                )

                current_eq_display = eq_with_frac

        # Then reduce the fraction if possible.
        for i in range(1, len(frac_steps)):
            before_eq = current_eq_display
            after_eq  = before_eq.replace(frac_steps[i - 1], frac_steps[i], 1)

            if before_eq != after_eq:
                steps.append(
                    Step(
                        before=before_eq,
                        after=after_eq,
                    )
                )

                current_eq_display = after_eq

    # ------------------------------------------------------------------
    # Rearrange: move all x-terms to the left, constants to the right.
    # ------------------------------------------------------------------
    variable_terms, constant_terms = rearrange_terms(left, right)

    new_eq = build_equation(variable_terms, constant_terms)

    steps.append(
        Step(
            before=equation_display,
            after=equation_display,
            explanation="Rearrange terms",
        )
    )

    steps.append(
        Step(
            before=equation_display,
            after=new_eq,
        )
    )

    # ------------------------------------------------------------------
    # Simplify variable terms step by step.
    # ------------------------------------------------------------------
    if len(variable_terms) > 1:
        steps.append(
            Step(
                before=new_eq,
                after=new_eq,
                explanation="Simplify the variable side",
            )
        )

    current_vars = variable_terms

    for entry in combine_terms_stepwise(variable_terms):
        if isinstance(entry, tuple) and entry[0] == "__latex__":
            _, var_latex, _state = entry
            before_eq  = build_equation(current_vars, constant_terms)
            const_side = before_eq.split("=")[1].strip()
            after_eq   = f"{var_latex} = {const_side}"

            if before_eq != after_eq:
                steps.append(
                    Step(
                        before=before_eq,
                        after=after_eq,
                    )
                )

        else:
            new_vars = entry

            steps.append(
                Step(
                    before=build_equation(current_vars, constant_terms),
                    after=build_equation(new_vars, constant_terms),
                )
            )

            current_vars = new_vars

    # ------------------------------------------------------------------
    # Simplify constant terms step by step.
    # ------------------------------------------------------------------
    current_consts = constant_terms

    if len(constant_terms) > 1:
        steps.append(
            Step(
                before=build_equation(current_vars, current_consts),
                after=build_equation(current_vars, current_consts),
                explanation="Simplify the constant side",
            )
        )

    for entry in combine_terms_stepwise(constant_terms):
        if isinstance(entry, tuple) and entry[0] == "__latex__":
            _, const_latex, _state = entry
            before_eq = build_equation(current_vars, current_consts)
            var_side  = before_eq.split("=")[0].strip()
            after_eq  = f"{var_side} = {const_latex}"

            if before_eq != after_eq:
                steps.append(
                    Step(
                        before=before_eq,
                        after=after_eq,
                    )
                )

        else:
            new_consts = entry

            steps.append(
                Step(
                    before=build_equation(current_vars, current_consts),
                    after=build_equation(current_vars, new_consts),
                )
            )

            current_consts = new_consts

    # ------------------------------------------------------------------
    # Isolate x.
    # ------------------------------------------------------------------
    final_left  = current_vars[0]
    final_right = current_consts[0]
    coef        = final_left.coeff(x)
    const       = final_right

    # Case 1: coefficient is already 1 -- no division needed.
    if coef == 1:
        steps.append(
            Step(
                before=f"{sp.latex(final_left)} = {sp.latex(final_right)}",
                after=f"x = {sp.latex(const)}",
            )
        )

        final_value           = const
        final_latex           = sp.latex(final_value)
        decimal_approximation = decimal_str(final_value)

        if decimal_approximation:
            steps.append(
                Step(
                    before=f"x = {final_latex}",
                    after=f"x = {final_latex} \\approx {decimal_approximation}",
                )
            )

        check_solution(final_value, final_latex, equation, steps)

    # Case 2: coefficient requires a division step.
    else:
        coefficient_rational = sp.Rational(coef)
        constant_rational    = sp.Rational(const)
        final_left_latex     = sp.latex(final_left)
        final_right_latex    = sp.latex(final_right)

        # Sub-case 2a: fractional coefficient (e.g. -x/5 = 3).
        # Multiply both sides by the denominator first, then divide by the numerator.
        if coefficient_rational.q > 1:
            extra_steps, solution = rational_coef_solve_steps(
                coefficient_rational,
                constant_rational,
                final_left_latex,
                final_right_latex,
            )

            steps.extend(extra_steps)

        # Sub-case 2b: integer coefficient (e.g. 15x = 3).
        else:
            extra_steps, solution = integer_coef_solve_steps(
                coefficient_rational,
                constant_rational,
                final_left_latex,
                final_right_latex,
            )

            steps.extend(extra_steps)

        final_value           = solution
        final_latex           = sp.latex(final_value)
        decimal_approximation = decimal_str(final_value)

        if decimal_approximation:
            steps.append(
                Step(
                    before=f"x = {final_latex}",
                    after=f"x = {final_latex} \\approx {decimal_approximation}",
                )
            )

        check_solution(final_value, final_latex, equation, steps)

    return steps

