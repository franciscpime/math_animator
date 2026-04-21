import re
import sympy as sp
from math import lcm as _mlcm
from functools import reduce as _freduce
from utils.latex_helpers import frac_x_latex, frac_latex, join_latex

x = sp.symbols("x")


"""
    Combines like terms step by step, emitting intermediate visual states so
    the animation can show every arithmetic operation explicitly.

    Given terms like [2x, (4/5)x, -3x, 7, 1/2]:
      - Variable terms  >>  [2x, (4/5)x, -3x]
      - Constant terms  >>  [7, 1/2]

    For variable terms the sequence is:
      Step A -- convert every coefficient to the common denominator:
                2x + (4/5)x - 3x  >>  10x/5 + 4x/5 - 15x/5
      Step B -- group all numerators under the common denominator (unevaluated):
                >>  (10 + 4 - 15)x / 5
      Step C -- emit the simplified SymPy result:
                >>  -x/5

    For constant terms the same three-step pattern applies:
      Step A -- 7 + 1/2  >>  14/2 + 1/2
      Step B -- >>  (14 + 1) / 2
      Step C -- >>  15/2

    Returns a list of entries, each being either:
      - a plain Python list of SymPy terms     (final simplified state)
      - a tuple ("__latex__", latex_str, sympy_terms_list)
                                               (display-only intermediate step)
    """
def combine_terms_stepwise(terms):

    # Start with a copy of the input so we never mutate the original list
    new_terms = terms.copy()

    # Will hold all the intermediate visual steps to return to the caller
    steps = []

    # Separate the incoming terms into those that contain x and pure numbers
    # Example: [2x, (4/5)x, -3x, 7, 1/2]  
    #           >>>  x_terms=[2x, 4/5x, -3x]
    #           >>>  const_terms=[7, 1/2]
    x_terms = []
    const_terms = []

    for term in new_terms:
        if term.has(x):
            x_terms.append(term)
        else:
            const_terms.append(term)

    # --------------------------
    # Process variable terms
    # --------------------------

    # Only combine if there is more than one x-term to work with
    # Example: [2x, (4/5)x, -3x]  >>  needs combining;  
    #          [2x]  >>  nothing to do
    if len(x_terms) > 1:

        # Extract the rational coefficient from each x-term
        # Example: [2x, (4/5)x, -3x]  >>  coefs = [2, 4/5, -3]
        coefs = []

        for term in x_terms:
            coefficient = sp.Rational(term.coeff(x))
            coefs.append(coefficient)

        # Collect the denominator of every coefficient
        # Example: [2, 4/5, -3]  >>  denominators = [1, 5, 1]
        denominators = []

        for coefficient in coefs:
            denominators.append(coefficient.q)

        # Find the least common denominator across all coefficients
        # Example: lcm(1, 5, 1)  >>  common_denominator = 5
        common_denominator = _freduce(_mlcm, denominators)

        # Check whether all terms already share the same denominator
        # Example: [1, 5, 1]  >>  not all the same  >>  all_same = False
        all_same = True

        for denominator in denominators:
            if denominator != common_denominator:
                all_same = False
                break

        if not all_same:

            # Scale every numerator up to the common denominator
            # Example: 2   with den=1  >>  2   * (5//1) = 10
            #          4/5 with den=5  >>  4   * (5//5) = 4
            #          -3  with den=1  >>  -3  * (5//1) = -15
            adjusted_numerators = []

            for coefficient in coefs:
                adjusted_numerator = coefficient.p * (common_denominator // coefficient.q)
                adjusted_numerators.append(adjusted_numerator)

            # Build the LaTeX string for each adjusted term
            # Example: [10, 4, -15] with den=5  >>  ['\frac{10 x}{5}', '\frac{4 x}{5}', '- \frac{15 x}{5}']
            parts_a = []

            for adjusted_numerator in adjusted_numerators:
                parts_a.append(frac_x_latex(adjusted_numerator, common_denominator))

            # Emit Step A: display-only step showing all terms over the common denominator
            # Example: '\frac{10 x}{5} + \frac{4 x}{5} - \frac{15 x}{5}'
            steps.append(("__latex__", join_latex(parts_a), new_terms.copy()))

        else:

            # All denominators are already the same -- just collect the numerators 
            # Example: [x/5, 4x/5]  >>  adjusted_numerators = [1, 4]
            adjusted_numerators = []

            for coefficient in coefs:
                adjusted_numerators.append(coefficient.p)

        # Add all scaled numerators together to get the combined numerator
        # Example: 10 + 4 + (-15)  >>  numerator_sum = -1
        numerator_sum = sum(adjusted_numerators)

        # Convert each numerator to a string to build the display expression
        # Example: [10, 4, -15]  >>  ['10', '4', '-15']
        numerator_str_list = []

        for adjusted_numerator in adjusted_numerators:
            numerator_str_list.append(str(adjusted_numerator))

        # Join numerators with " + " then clean up any "+ -" into "- "
        # Example: '10 + 4 + -15'  >>  '10 + 4 - 15'
        numerators_joined = " + ".join(numerator_str_list)
        numerators_joined = re.sub(r'\+\s*-', '- ', numerators_joined)

        # Wrap in a \frac if the common denominator is greater than 1
        # Example: common_denominator=5  >>  '\frac{(10 + 4 - 15) x}{5}'
        if common_denominator > 1:
            grouped_numerator = fr"\frac{{({numerators_joined}) x}}{{{common_denominator}}}"
        else:
            # Denominator is 1 -- no fraction needed, just show the sum of numerators
            # Example: '(10 + 4 - 15) x'
            grouped_numerator = f"({numerators_joined}) x"

        # Only emit Step B when denominators were actually different (Step A ran)
        if not all_same:
            steps.append(("__latex__", grouped_numerator, new_terms.copy()))

            running = adjusted_numerators[0]

            for i in adjusted_numerators[1:]:
                running = running + i
                

        # Step C: let SymPy evaluate and simplify the final combined x-term
        # Example: Rational(-1, 5) * x  >>  -x/5
        combined = sp.Rational(numerator_sum, common_denominator) * x

        # Rebuild the term list: combined x-term first, then any constants
        # Example: [-x/5, 7, 1/2]
        new_terms = [combined] + const_terms

        # Emit Step C: the final simplified state
        steps.append(new_terms.copy())

    # ------------------------------------------------------------------
    # Process constant terms
    # ------------------------------------------------------------------

    # Only combine if there is more than one constant term to work with
    # Example: [7, 1/2]  >>  needs combining;  [7]  >>  nothing to do
    elif len(const_terms) > 1:

        # Convert every constant term to a SymPy Rational for safe arithmetic
        # Example: [7, 1/2]  >>  constant_coefs = [Rational(7,1), Rational(1,2)]
        constant_coefs = []

        for term in const_terms:
            constant_coefs.append(sp.Rational(term))

        # Collect the denominator of every constant coefficient
        # Example: [Rational(7,1), Rational(1,2)]  >>  constant_denominators = [1, 2]
        constant_denominators = []

        for constant_coef in constant_coefs:
            constant_denominators.append(constant_coef.q)

        # Find the least common denominator across all constants
        # Example: lcm(1, 2)  >>  common_denominator_const = 2
        common_denominator_const = _freduce(_mlcm, constant_denominators)

        # Check whether all constants already share the same denominator
        # Example: [1, 2]  >>  not all the same  >>  all_same_denominator = False
        all_same_denominator = True

        for current_denominator in constant_denominators:
            if current_denominator != common_denominator_const:
                all_same_denominator = False
                break

        if not all_same_denominator:

            # Scale each constant numerator up to the common denominator
            # Example: 7   with den=1  >>  7  * (2//1) = 14
            #          1/2 with den=2  >>  1  * (2//2) = 1
            adjusted_const_numerators = []

            for constant_coef in constant_coefs:
                adjusted_numerator = constant_coef.p * (common_denominator_const // constant_coef.q)
                adjusted_const_numerators.append(adjusted_numerator)

            # Build the LaTeX string for each scaled constant term
            # Example: [14, 1] with den=2  >>  ['\frac{14}{2}', '\frac{1}{2}']
            parts_a = []

            for adjusted_numerator in adjusted_const_numerators:
                parts_a.append(frac_latex(adjusted_numerator, common_denominator_const))

            # Emit Step A: display-only step showing all constants over the common denominator
            # Example: '\frac{14}{2} + \frac{1}{2}'
            steps.append(("__latex__", join_latex(parts_a), new_terms.copy()))

        else:

            # All denominators are the same -- collect the numerators as-is
            # Example: [1/2, 3/2]  >>  adjusted_const_numerators = [1, 3]
            adjusted_const_numerators = []

            for constant_coef in constant_coefs:
                adjusted_const_numerators.append(constant_coef.p)

        # Sum all the scaled numerators to get the combined numerator
        # Example: 14 + 1  >>  sum_of_const_numerators = 15
        sum_of_const_numerators = 0

        for adjusted_numerator in adjusted_const_numerators:
            sum_of_const_numerators += adjusted_numerator

        # Convert each numerator to a string for building the display expression
        # Example: [14, 1]  >>  ['14', '1']
        const_numerator_strings = []

        for adjusted_numerator in adjusted_const_numerators:
            const_numerator_strings.append(str(adjusted_numerator))

        # Join numerators with " + " then clean up any "+ -" into "- "
        # Example: '14 + 1'  >>  '14 + 1'  (no cleanup needed here)
        const_numerator_expression = " + ".join(const_numerator_strings)
        const_numerator_expression = re.sub(r'\+\s*-', '- ', const_numerator_expression)

        # Wrap in a \frac if the common denominator is greater than 1
        # Example: common_denominator_const=2  >>  '\frac{14 + 1}{2}'
        if common_denominator_const > 1:
            grouped_const_numerator = fr"\frac{{{const_numerator_expression}}}{{{common_denominator_const}}}"
        else:
            # Denominator is 1 -- no fraction needed, just show the sum in parentheses
            # Example: '(14 + 1)'
            grouped_const_numerator = f"({const_numerator_expression})"

        # Only emit Step B when denominators were actually different (Step A ran)
        if not all_same_denominator:
            steps.append(("__latex__", grouped_const_numerator, new_terms.copy()))

        # Step C: let SymPy evaluate and simplify the final combined constant
        # Example: Rational(15, 2)  >>  15/2
        combined_constant = sp.Rational(sum_of_const_numerators, common_denominator_const)

        # Rebuild the term list: any x-terms first, then the combined constant
        # Example: [2x, 15/2]
        new_term_list = list(x_terms) + [combined_constant]

        # Emit Step C: the final simplified state
        steps.append(new_term_list.copy())

    return steps

