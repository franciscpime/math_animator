import sympy as sp
from models.step import Step


'''
This function generates the steps to solve a linear equation with a rational coefficient.

The strategy is to first clear the denominator by multiplying both sides
by q, then isolate x by dividing by p.

Example: -x/5 = 25/2:
      Multiply both sides by 5 (q)
        >>> -x = 25 * 5 / 2
        >>> -x = 125/2
      Divide both sides by -1 (p)
        >>> x = 125 / -2
        >>> x = -125/2

Parameters
----------
coef_rational       -- coefficient of x, e.g. Rational(-1, 5)
const_rational      -- right-hand side,   e.g. Rational(25, 2)
final_left_latex    -- LaTeX of the left side before solving
final_right_latex   -- LaTeX of the right side before solving
'''
def rational_coef_solve_steps(
        coef_rational: sp.Rational,
        const_rational: sp.Rational,
        final_left_latex: str,
        final_right_latex: str
):

    x = sp.symbols("x")

    # Will hold all the animation steps 
    result_steps = []

    # Extract the numerator (p) of the coefficient 
    # Example: coef_rational = -1/5  >>  coef_numerator = -1
    coef_numerator = coef_rational.p

    # Extract the denominator (q) of the coefficient
    # Example: coef_rational = -1/5  >>  coef_denominator = 5
    coef_denominator = coef_rational.q

    # Store the right-hand side constant for easier reference throughout
    # Example: const_rational = 25/2
    right_side_constant = const_rational

    # Build the equation string as it looks before any operation is applied
    # Example: '-\frac{x}{5} = \frac{25}{2}'
    eq_before_multiply = f"{final_left_latex} = {final_right_latex}"

    # Announce the upcoming multiplication before showing the result
    result_steps.append(
        Step(
            before=eq_before_multiply,
            after=eq_before_multiply,
            explanation=f"Multiply both sides by {coef_denominator}",
        )
    )

    # ---------------------------------------------------------------
    # Step 1: multiply both sides by q to clear the fraction
    # Example: (-1/5)x = 25/2  >>  multiply by 5  >>  -x = 25*5/2
    # ---------------------------------------------------------------

    # Build the left side after multiplying by q 
    # Example: coef_numerator = -1, x >> '-x'
    left_without_denominator = sp.latex(sp.Integer(coef_numerator) * x)

    # Extract numerator and denominator of the right-hand side constant
    # Example: const_rational = 25/2  >>  costante_numerator = 25, costante_denominator = 2
    costante_numerator = right_side_constant.p
    costante_denominator = right_side_constant.q

    # Wrap a negative right-hand numerator in parentheses for visual clarity
    # Example: costante_numerator = -25  >>  cn_str = '(-25)'
    #          costante_numerator = 25   >>  cn_str = '25'
    if costante_numerator < 0:
        costante_numerator_str = f"({costante_numerator})"
    else:
        costante_numerator_str = str(costante_numerator)

    # Build the right side showing the multiplication before evaluating it
    # Example: costante_denominator = 1  >> '25 \cdot 5'
    #          costante_denominator = 2  >> '\frac{25 \cdot 5}{2}'
    if costante_denominator == 1:
        right_side_product_unevaluated = f"{costante_numerator_str} \\cdot {coef_denominator}"
    else:
        right_side_product_unevaluated = f"\\frac{{{costante_numerator_str} \\cdot {coef_denominator}}}{{{costante_denominator}}}"

    # Build the full equation string showing the unevaluated multiplication
    # Example: '-x = \frac{25 \cdot 5}{2}'
    eq_after_multiply_unevaluated = f"{coef_denominator} \\cdot {final_left_latex} = {right_side_product_unevaluated}"

    # Emit the step showing the unevaluated multiplication on the right side
    result_steps.append(
        Step(
            before=eq_before_multiply,
            after=eq_after_multiply_unevaluated
        )
    )

    # Evaluate the product costante_numerator * coef_denominator
    # Example: 25 * 5  >>  numerator_product = 125
    numerator_product = costante_numerator * coef_denominator

    # Build the right side with the product now evaluated
    # Example: costante_denominator = 1  >> '125'
    #          costante_denominator = 2  >> '\frac{125}{2}'
    if costante_denominator == 1:
        right_side_product_evaluated = str(numerator_product)
    else:
        right_side_product_evaluated = sp.latex(sp.Rational(numerator_product, costante_denominator))

    # Build the full equation string with the evaluated product
    # Example: '-x = \frac{125}{2}'
    eq_after_multiply_evaluated = f"{left_without_denominator} = {right_side_product_evaluated}"

    # Only emit this step if something actually changed after evaluation
    # Example: '\frac{25 \cdot 5}{2}'  !=  '\frac{125}{2}'  >>  emit the step
    if eq_after_multiply_evaluated != eq_after_multiply_unevaluated:
        result_steps.append(
            Step(
                before=eq_after_multiply_unevaluated, 
                after=eq_after_multiply_evaluated
            )
        )
    else:
        # Nothing changed -- keep the unevaluated form as the current state
        eq_after_multiply_evaluated = eq_after_multiply_unevaluated



    # ---------------------------------------------------------------
    # Step 2: divide both sides by p to isolate x
    # Example: -x = 125/2  >>  divide by -1  >>  x = 125/-2
    # ---------------------------------------------------------------

    # Example: Rational(125, 2 * -1)  >>  Rational(-125, 2)  >>  -125/2
    solution = sp.Rational(numerator_product, costante_denominator * coef_numerator)

    # Build the right side showing the division as an explicit fraction before simplifying
    # Example: costante_denominator = 1  >> '\frac{125}{-1}'
    #          costante_denominator = 2  >> '\frac{125}{2 * -1}' >> '\frac{125}{-2}'
    if costante_denominator == 1:
        right_side_divided = f"{numerator_product}"
    else:
        right_side_divided = f"\\frac{{{numerator_product}}}{{{costante_denominator * coef_numerator}}}"

    # Build the full equation string showing the unevaluated division
    # Example: 'x = \frac{125}{-2}'
    eq_after_divide_unevaluated = f"x = {right_side_divided}"


    # Example: Rational(-125, 2)  >>  '-\frac{125}{2}'
    eq_solution_simplified = f"x = {sp.latex(solution)}"

    if coef_numerator == 1:
        # Emit the step showing x equal to the unevaluated fraction
        result_steps.append(
            Step(
                before=eq_after_multiply_evaluated,
                after=eq_after_divide_unevaluated
            )
        )

        result_steps.append(
            Step(
                before=eq_after_divide_unevaluated,
                after=eq_solution_simplified
            )
        )

    else:
        # Announce the upcoming division before showing the result
        result_steps.append(
            Step(
                before=eq_after_multiply_evaluated,
                after=eq_after_multiply_evaluated,
                explanation=f"Divide both sides by {coef_numerator}",
            )
        )

        # Emit the step showing x equal to the unevaluated fraction
        result_steps.append(
            Step(
                before=eq_after_multiply_evaluated,
                after=eq_after_divide_unevaluated
            )
        )

        result_steps.append(
            Step(
                before=eq_after_divide_unevaluated,
                after=eq_solution_simplified
            )
        )

    # Only emit the simplification step if the fraction actually changed
    # Example: 'x = \frac{125}{-2}' != 'x = -\frac{125}{2}' >> emit the step
    if eq_solution_simplified != eq_after_divide_unevaluated:
        result_steps.append(
            Step(
                before=eq_after_divide_unevaluated,
                after=eq_solution_simplified
            )
        )

    # print(f"numerator product: {numerator_product}")
    

    return result_steps, solution
