import sympy as sp

from models.step import EquationStep

from solver.utils.term_extractor import extract_terms
from solver.utils.term_combiner import combine_terms_stepwise
from solver.utils.equation_builder import build_equation

x = sp.symbols("x")


def solve_linear(left, right, equation):

    steps = []

    # first: display the original equation
    steps.append(
        EquationStep(
            before=equation,
            after=equation
        )
    )

    # Extract individual terms
    left_terms = extract_terms(left)
    right_terms = extract_terms(right)

    left_x = []
    left_const = []

    right_x = []
    right_const = []

    # Separate variables and constants on the left
    for term in left_terms:
        if term.has(x):
            left_x.append(term)
        else:
            left_const.append(term)

    # Separate variables and constants on the right
    for term in right_terms:
        if term.has(x):
            right_x.append(term)
        else:
            right_const.append(term)

    # Move variable terms to the left side
    variable_terms = []

    for term in left_x:
        variable_terms.append(term)

    for term in right_x:
        variable_terms.append(-term)

    # Move constant terms to the right side
    constant_terms = []

    for term in right_const:
        constant_terms.append(term)

    for term in left_const:
        constant_terms.append(-term)

    # First transformation step
    step1 = build_equation(variable_terms, constant_terms)

    steps.append(
        EquationStep(
            before=equation,
            after=step1
        )
    )

    # Simplify variable terms step-by-step
    variable_steps = combine_terms_stepwise(variable_terms)

    current_variables = variable_terms

    for new_variables in variable_steps:

        before = build_equation(current_variables, constant_terms)
        after = build_equation(new_variables, constant_terms)

        steps.append(
            EquationStep(
                before=before,
                after=after
            )
        )

        current_variables = new_variables

    # Simplify constant terms step-by-step
    constant_steps = combine_terms_stepwise(constant_terms)

    current_consts = constant_terms

    for new_consts in constant_steps:

        before = build_equation(current_variables, current_consts)
        after = build_equation(current_variables, new_consts)

        steps.append(
            EquationStep(
                before=before,
                after=after
            )
        )

        current_consts = new_consts

    # Final linear equation
    final_left = current_variables[0]
    final_right = current_consts[0]

    coef = final_left.coeff(x)

    steps.append(
        EquationStep(
            before=f"{sp.latex(final_left)} = {sp.latex(final_right)}",
            after=f"x = \\frac{{{sp.latex(final_right)}}}{{{sp.latex(coef)}}}"
        )
    )

    solution = sp.simplify(final_right / coef)

    steps.append(
        EquationStep(
            before=f"x = \\frac{{{sp.latex(final_right)}}}{{{sp.latex(coef)}}}",
            after=f"x = {sp.latex(solution)}"
        )
    )

    return steps