import re
import sympy as sp
from math import lcm as _mlcm
from functools import reduce as _freduce

from utils.latex_helpers import frac_x_latex, frac_latex, join_latex

x = sp.symbols("x")


def combine_terms_stepwise(terms):
    """
    Combine like terms step by step, emitting intermediate visual states so
    the animation can show every arithmetic operation explicitly.

    For variable terms (those containing x):
      Step A >> convert every coefficient to the common denominator
      Step B >> group all numerators under the common denominator (unevaluated)
      Step C >> emit the simplified SymPy result

    For constant terms (no x):
      Same three-step pattern applied to the numeric values.

    Return value is a list of entries, each being either:
      - a plain Python list of SymPy terms  (updated state, ready to render)
      - a tuple ("__latex__", latex_str, sympy_terms_list)
        which signals a visual-only intermediate step.
    """
    new_terms = terms.copy()
    steps = []

    x_terms = []
    const_terms = []

    for term in new_terms:
        if term.has(x):
            x_terms.append(term)
        else:
            const_terms.append(term)

    # ------------------------------------------------------------------
    # Process variable terms
    # ------------------------------------------------------------------
    if len(x_terms) > 1:
        coefs = [sp.Rational(term.coeff(x)) for term in x_terms]
        denominators = [coef.q for coef in coefs]
        common_denominator = _freduce(_mlcm, denominators)

        all_same = all(d == common_denominator for d in denominators)

        if not all_same:
            adjusted_numerators = [
                coef.p * (common_denominator // coef.q) for coef in coefs
            ]
            parts_a = [frac_x_latex(n, common_denominator) for n in adjusted_numerators]
            steps.append(("__latex__", join_latex(parts_a), new_terms.copy()))
        else:
            adjusted_numerators = [coef.p for coef in coefs]

        numerator_sum = sum(adjusted_numerators)
        numerator_str_list = [str(n) for n in adjusted_numerators]
        numerators_joined = " + ".join(numerator_str_list)
        numerators_joined = re.sub(r'\+\s*-', '- ', numerators_joined)

        if common_denominator > 1:
            grouped_numerator = fr"\frac{{({numerators_joined}) x}}{{{common_denominator}}}"
        else:
            grouped_numerator = f"({numerators_joined}) x"

        if not all_same:
            steps.append(("__latex__", grouped_numerator, new_terms.copy()))

        combined = sp.Rational(numerator_sum, common_denominator) * x
        new_terms = [combined] + const_terms
        steps.append(new_terms.copy())

    # ------------------------------------------------------------------
    # Process constant terms
    # ------------------------------------------------------------------
    elif len(const_terms) > 1:
        constant_coefs = [sp.Rational(t) for t in const_terms]
        constant_denominators = [c.q for c in constant_coefs]
        common_denominator_const = _freduce(_mlcm, constant_denominators)

        all_same_denominator = all(d == common_denominator_const for d in constant_denominators)

        if not all_same_denominator:
            adjusted_const_numerators = [
                c.p * (common_denominator_const // c.q) for c in constant_coefs
            ]
            parts_a = [frac_latex(n, common_denominator_const) for n in adjusted_const_numerators]
            steps.append(("__latex__", join_latex(parts_a), new_terms.copy()))
        else:
            adjusted_const_numerators = [c.p for c in constant_coefs]

        sum_of_const_numerators = sum(adjusted_const_numerators)
        const_numerator_strings = [str(n) for n in adjusted_const_numerators]
        const_numerator_expression = " + ".join(const_numerator_strings)
        const_numerator_expression = re.sub(r'\+\s*-', '- ', const_numerator_expression)

        if common_denominator_const > 1:
            grouped_const_numerator = fr"\frac{{{const_numerator_expression}}}{{{common_denominator_const}}}"
        else:
            grouped_const_numerator = f"({const_numerator_expression})"

        if not all_same_denominator:
            steps.append(("__latex__", grouped_const_numerator, new_terms.copy()))

        combined_constant = sp.Rational(sum_of_const_numerators, common_denominator_const)
        new_term_list = list(x_terms) + [combined_constant]
        steps.append(new_term_list.copy())

    return steps

