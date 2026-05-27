import sympy as sp
from models.step import Step
import re

"""
Generate the animation steps for a division by zero case.

Example: numerator = '4'
    Step 1: show the division by zero
            >> '\frac{4}{0}'
    Step 2: show the limits
            >> '\frac{4}{0^+} = +\infty \quad \lor \quad \frac{4}{0^-} = -\infty'
    Step 3: show the impossible solution message
            >> explanation = 'Impossible solution'
"""
def impossible_solution_steps(numerator_latex: str, sign: str, equation_display: str, equation_raw: str) -> list[Step]:

    div_by_zero = r"\frac{" + numerator_latex + r"}{0}"

    if sign == '+':
        infinity = r"\infty"     
    else:
        infinity = r"-\infty"

    limits = (
        r"\frac{" + numerator_latex + r"}{0^+} = +\infty"
        r" \quad \lor \quad "
        r"\frac{" + numerator_latex + r"}{0^-} = -\infty"
    )

    # Replace the division by zero in the display equation with the infinity symbol
    div_by_zero_pattern = sign + numerator_latex + "/0"
    equation_with_infinity = equation_display.replace(
        r"\frac{" + numerator_latex + r"}{0}",
        infinity
    )

    steps = []

    steps.append(
        Step(
            before = equation_display, 
            after = equation_display
        )
    )

    steps.append(
        Step(
            before = div_by_zero, 
            after = limits
        )
    )

    steps.append(
        Step(
            before = limits, 
            after = equation_display
        )
    )

    steps.append(
        Step(
            before = equation_display, 
            after = equation_with_infinity
        )
    )

    steps.append(
        Step(
            before = 
            equation_with_infinity, 
            after = equation_with_infinity, explanation="Impossible solution"
        )
    )
    
    return steps


