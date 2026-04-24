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

'''
This function solves a linear equation in one variable (x) and returns the full list of
animation Steps, each representing a single visible change in the equation.

The function handles:
    - Decimal coefficients  (converted to fractions first)
    - Unreduced fractions   (simplified before solving)
    - Multiple variable and constant terms on both sides
    - Rational (fractional) coefficients of x
    - Verification of the solution by back-substitution
'''
def solve_linear(equation: str):
    
    # Parse the equation string into normalized left and right side strings
    # Example: '2x + 4 = 10'  >>  left = '2*x+4', right = '10'
    left, right = parse_equation(equation)

    # Will hold all the animation steps to return to the caller
    steps = []

    # Detect any decimal numbers in the original equation string
    # Example: '0.5x + 1 = 3'  >>  raw_decimals = [('0.5', Rational(1, 2))]
    raw_decimals = detect_decimals(equation)

    # Detect any unreduced fractions in the original equation string
    # Example: '6/4 x + 1 = 3'  >>  raw_fractions = [('6', '4', Rational(3, 2))]
    raw_fractions = detect_raw_fractions(equation)

    # Convert the equation string to a LaTeX display format for the animation
    # Example: '1/2 x + 3 = 7/4'  >>  '\frac{1}{2} x + 3 = \frac{7}{4}'
    equation_display = equation_to_latex_display(equation)

    # Keep a running copy of the display string as pre-solve steps modify it
    current_eq_display = equation_display

    # ------------------------------------------------------------------
    # Pre-solve step 1: convert decimal coefficients to fractions.
    # Each decimal is shown as an unreduced fraction first, then reduced.
    # Example: '0.5x + 1 = 3'  >>  '\frac{5}{10}x + 1 = 3'  >>  '\frac{1}{2}x + 1 = 3'
    # ------------------------------------------------------------------

    # Process each detected decimal one at a time
    for dec_str, _dec_val in raw_decimals:

        # Get the list of LaTeX strings walking through the conversion
        # Example: '0.5'  >>  ['0.5', '\frac{5}{10}', '\frac{1}{2}']
        dec_steps = decimal_simplification_steps(dec_str)

        # Walk through each transition in the conversion sequence
        for i in range(1, len(dec_steps)):

            # The equation before this transition
            before_eq = current_eq_display

            # Replace only the first occurrence so we handle one decimal at a time
            after_eq = before_eq.replace(dec_steps[i - 1], dec_steps[i], 1)

            # Only emit a step if something actually changed
            if before_eq != after_eq:

                steps.append(
                    Step(
                        before = before_eq,
                        after = after_eq
                    )
                )

                # Update the running display string for the next iteration
                current_eq_display = after_eq

    # ------------------------------------------------------------------
    # Pre-solve step 2: simplify unreduced fractions (e.g. 6/4 >> 3/2).
    # ------------------------------------------------------------------

    # Process each detected fraction one at a time
    for num_s, den_s, _frac_val in raw_fractions:

        # Get the list of LaTeX strings walking through the simplification
        # Example: ('6', '4')  >>  ['\frac{6}{4}', '\frac{3}{2}']
        frac_steps = fraction_simplification_steps(num_s, den_s)

        # Build the original plain-text fraction string to look for in the display
        # Example: num_s='6', den_s='4'  >>  frac_orig_str = '6/4'
        frac_orig_str = num_s + "/" + den_s

        # First show the plain a/b text as \frac{a}{b} before any reduction
        if frac_orig_str in current_eq_display:

            # Replace the plain fraction with the LaTeX version
            # Example: '6/4 x + 1 = 3'  >>  '\frac{6}{4} x + 1 = 3'
            eq_with_frac = current_eq_display.replace(frac_orig_str, frac_steps[0], 1)

            # Only emit a step if something actually changed
            if eq_with_frac != current_eq_display:
                steps.append(
                    Step(
                        before = current_eq_display,
                        after = eq_with_frac,
                    )
                )

                # Update the running display string
                current_eq_display = eq_with_frac

        # Then walk through any further reduction steps
        # Example: '\frac{6}{4}'  >>  '\frac{3}{2}'
        for i in range(1, len(frac_steps)):

            # The equation before this transition
            before_eq = current_eq_display

            # Replace only the first occurrence of the previous form with the next
            after_eq = before_eq.replace(frac_steps[i - 1], frac_steps[i], 1)

            # Only emit a step if something actually changed
            if before_eq != after_eq:
                steps.append(
                    Step(
                        before = before_eq,
                        after = after_eq,
                    )
                )

                # Update the running display string
                current_eq_display = after_eq

    # ------------------------------------------------------------------
    # Rearrange: move all x-terms to the left, constants to the right.
    # Example: '2x + 4 = 10'  >>  variable_terms=[2x], constant_terms=[10, -4]
    # ------------------------------------------------------------------

    # Delegate the term separation and sign-flipping to the dedicated utility
    variable_terms, constant_terms, already_organised = rearrange_terms(left, right)

    # Build the LaTeX equation string showing all terms in their new positions
    # Example: variable_terms=[2x], constant_terms=[6]  >>  '2 x = 6'
    new_eq = build_equation(variable_terms, constant_terms)

    if not already_organised:
        # Announce the rearrangement before showing the result
        steps.append(
            Step(
                before = current_eq_display,
                after = current_eq_display,
                explanation = "Rearrange terms",
            )
        )

        # Show the equation after rearranging all terms
        steps.append(
            Step(
                before = current_eq_display,
                after = new_eq,
            )
        )

    # ------------------------------------------------------------------
    # Simplify variable terms step by step.
    # Example: 2x + (4/5)x - 3x  >>  -x/5
    # ------------------------------------------------------------------

    # Only announce the simplification if there is more than one variable term to combine
    if len(variable_terms) > 1:
        steps.append(
            Step(
                before = new_eq,
                after = new_eq,
                explanation = "Simplify the variable side",
            )
        )

    # Keep a running list of the current variable terms as they get combined
    current_vars = variable_terms

    # Walk through every intermediate state emitted by the stepwise combiner
    for entry in combine_terms_stepwise(variable_terms):

        # Display-only intermediate step -- the LaTeX is provided directly
        # by the combiner and the variable terms have NOT been updated yet
        if isinstance(entry, tuple) and entry[0] == "__latex__":
            _, var_latex, _state = entry

            # Build the full equation using the current (not yet updated) variable terms
            before_eq = build_equation(current_vars, constant_terms)

            # Extract the constant side as-is and pair it with the new variable LaTeX
            const_side = before_eq.split("=")[1].strip()
            after_eq   = f"{var_latex} = {const_side}"

            # Only emit a step if something actually changed
            if before_eq != after_eq:
                steps.append(
                    Step(
                        before = before_eq,
                        after = after_eq,
                    )
                )

        # Plain list entry -- the variable terms have been fully updated
        else:
            new_vars = entry

            # Show the transition from the old variable terms to the new combined term
            steps.append(
                Step(
                    before = build_equation(current_vars, constant_terms),
                    after = build_equation(new_vars, constant_terms)
                )
            )

            # Update current_vars to the newly combined state
            current_vars = new_vars

    # ------------------------------------------------------------------
    # Simplify constant terms step by step.
    # Example: 10 - 4  >>  6
    # ------------------------------------------------------------------

    # Keep a running list of the current constant terms as they get combined
    current_consts = constant_terms

    # Only announce the simplification if there is more than one constant term to combine
    if len(constant_terms) > 1:
        steps.append(
            Step(
                before = build_equation(current_vars, current_consts),
                after = build_equation(current_vars, current_consts),
                explanation = "Simplify the constant side",
            )
        )

    # Walk through every intermediate state emitted by the stepwise combiner
    for entry in combine_terms_stepwise(constant_terms):

        # Display-only intermediate step -- the LaTeX is provided directly
        # by the combiner and the constant terms have NOT been updated yet
        if isinstance(entry, tuple) and entry[0] == "__latex__":
            _, const_latex, _state = entry

            # Build the full equation using the current (not yet updated) constant terms
            before_eq = build_equation(current_vars, current_consts)

            # Extract the variable side as-is and pair it with the new constant LaTeX
            var_side = before_eq.split("=")[0].strip()
            after_eq = f"{var_side} = {const_latex}"

            # Only emit a step if something actually changed
            if before_eq != after_eq:
                steps.append(
                    Step(
                        before = before_eq,
                        after = after_eq,
                    )
                )

        # Plain list entry -- the constant terms have been fully updated
        else:
            new_consts = entry

            # Show the transition from the old constant terms to the new combined term
            steps.append(
                Step(
                    before = build_equation(current_vars, current_consts),
                    after = build_equation(current_vars, new_consts),
                )
            )

            # Update current_consts to the newly combined state
            current_consts = new_consts

    # ------------------------------------------------------------------
    # Isolate x.
    # ------------------------------------------------------------------

    # Extract the single remaining variable term and constant term
    # Example: current_vars=[6x], current_consts=[6]  >>  final_left=6x, final_right=6
    final_left  = current_vars[0]
    final_right = current_consts[0]

    # Extract the coefficient of x from the left side
    # Example: 6x  >>  coef = 6
    coef = final_left.coeff(x)

    # The right side constant is the value x will equal after division
    # Example: final_right = 6  >>  const = 6
    const = final_right

    # ------------------------------------------------------------------
    # Case 1: coefficient is already 1 -- no division needed.
    # Example: x = 6  >>  solution is immediately x = 6
    # ------------------------------------------------------------------
    if coef == 1:

        # Show the direct step from '1*x = const' to 'x = const'
        steps.append(
            Step(
                before = f"{sp.latex(final_left)} = {sp.latex(final_right)}",
                after = f"x = {sp.latex(const)}",
            )
        )

        # The solution is the constant itself
        final_value = const

        # Get the LaTeX string of the solution for display
        # Example: sp.Integer(6)  >>  final_latex = '6'
        final_latex = sp.latex(final_value)

        # Check whether a decimal approximation is needed
        # Example: Rational(1, 3)  >>  decimal_approximation = '0.333'
        #          Integer(6)      >>  decimal_approximation = None
        decimal_approximation = decimal_str(final_value)

        # Show the decimal approximation step only when the solution is not a whole number
        if decimal_approximation:
            steps.append(
                Step(
                    before = f"x = {final_latex}",
                    after = f"x = {final_latex} \\approx {decimal_approximation}",
                )
            )

        # Verify the solution by substituting back into the original equation
        check_solution(final_value, final_latex, equation, steps)

    # ------------------------------------------------------------------
    # Case 2: coefficient requires a division step.
    # Example: 6x = 6  >>  divide both sides by 6  >>  x = 1
    # ------------------------------------------------------------------
    else:

        # Convert the coefficient and constant to SymPy Rationals for safe arithmetic
        # Example: coef=6, const=6  >>  coefficient_rational=Rational(6,1), constant_rational=Rational(6,1)
        coefficient_rational = sp.Rational(coef)
        constant_rational    = sp.Rational(const)

        # Get the LaTeX strings of the left and right sides before solving
        # Example: 6x  >>  final_left_latex = '6 x'
        final_left_latex  = sp.latex(final_left)
        final_right_latex = sp.latex(final_right)

        # Sub-case 2a: fractional coefficient (e.g. -x/5 = 3).
        # Multiply both sides by the denominator first, then divide by the numerator.
        if coefficient_rational.q > 1:
            extra_steps, solution = rational_coef_solve_steps(
                coefficient_rational,
                constant_rational,
                final_left_latex,
                final_right_latex,
            )

            # Add all the steps generated by the rational coefficient solver
            steps.extend(extra_steps)

        # Sub-case 2b: integer coefficient (e.g. 15x = 3).
        else:
            extra_steps, solution = integer_coef_solve_steps(
                coefficient_rational,
                constant_rational,
                final_left_latex,
                final_right_latex,
            )

            # Add all the steps generated by the integer coefficient solver
            steps.extend(extra_steps)

        # The solution returned by whichever solver ran above
        final_value = solution

        # Get the LaTeX string of the solution for display
        # Example: Rational(1, 5)  >>  final_latex = '\frac{1}{5}'
        final_latex = sp.latex(final_value)

        # Check whether a decimal approximation is needed
        # Example: Rational(1, 5)  >>  decimal_approximation = '0.2'
        #          Integer(1)      >>  decimal_approximation = None
        decimal_approximation = decimal_str(final_value)

        # Show the decimal approximation step only when the solution is not a whole number
        if decimal_approximation:
            steps.append(
                Step(
                    before = f"x = {final_latex}",
                    after = f"x = {final_latex} \\approx {decimal_approximation}",
                )
            )

        # Verify the solution by substituting back into the original equation
        check_solution(final_value, final_latex, equation, steps)

    # Return the complete list of animation steps to the caller
    return steps

