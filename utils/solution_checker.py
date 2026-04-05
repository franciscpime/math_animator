import re
import math
import sympy as sp
from math import lcm as _mlcm
from functools import reduce as _freduce
from models.step import Step
from parser.equation_parser import normalize_expression

x = sp.symbols("x")


def _fix_pm(expression):
    """Normalise double negatives and plus-minus combinations."""
    expression = re.sub(r"-\s*-\s*", "+ ", expression)
    expression = re.sub(r"\+\s*-\s*", "- ", expression)
    return expression.strip()


def sympy_stepwise(substituted_expression, sym_evaled, final_value):
    """
    Generate a list of (latex_str, explanation_or_None) tuples that walk
    through the arithmetic after substituting x = final_value.

    For a fractional final_value = p/q the sequence per term is:
      Step 1a  integer_coef * (p/q)  >> show product explicitly >> evaluate
      Step 1b  frac_coef * (p/q)    >> show product explicitly >> evaluate
      Step 1c  simplify any remaining unreduced constant fractions
      Step 2   convert isolated integers to the common denominator
      Step 3   add/subtract fractions pairwise

    For an integer final_value:
      Step 1   evaluate each product k*(val)
      Step 2   sum numeric pairs left to right
    """

    def _find_isolated_ints(expression):
        """Find integers outside LaTeX braces not immediately followed by / or {."""
        isolated_integers = []
        brace_level = 0
        i = 0

        while i < len(expression):
            current_char = expression[i]

            if current_char == "{":
                brace_level += 1
                i += 1
                continue
            if current_char == "}":
                brace_level -= 1
                i += 1
                continue
            if brace_level > 0:
                i += 1
                continue

            sign = 1
            start = i

            if current_char == "-" and i + 1 < len(expression) and expression[i + 1].isdigit():
                if i == 0 or expression[i - 1] in " +=({":
                    sign = -1
                    i += 1
                    current_char = expression[i]
                else:
                    i += 1
                    continue

            if current_char.isdigit():
                number_end = i
                while number_end < len(expression) and expression[number_end].isdigit():
                    number_end += 1
                if number_end < len(expression) and expression[number_end] in "/{":
                    i = number_end
                    continue
                try:
                    integer_value = sign * int(expression[i:number_end])
                except Exception:
                    i += 1
                    continue
                if integer_value != 0:
                    isolated_integers.append((start, number_end, integer_value))
                i = number_end
            else:
                i += 1

        return isolated_integers

    def _dens_in(expression):
        return [int(d) for d in re.findall(r"\\frac\{[^}]+\}\{(\d+)\}", expression)]

    result = [(substituted_expression, None)]
    current_expression = substituted_expression

    # ------------------------------------------------------------------
    # Integer solution path
    # ------------------------------------------------------------------
    if not (isinstance(final_value, sp.Rational) and final_value.q != 1):
        solution_as_int = int(final_value)
        solution_latex = sp.latex(final_value)
        solution_latex_escaped = re.escape(solution_latex)

        pattern = r"(-?\s*\d+)\s*\(\s*" + solution_latex_escaped + r"\s*\)"

        for _ in range(20):
            pattern_match = re.search(pattern, current_expression)
            if not pattern_match:
                break

            term_coefficient = int(pattern_match.group(1).replace(" ", ""))
            product = term_coefficient * solution_as_int
            before_match = current_expression[:pattern_match.start()]
            after_match = current_expression[pattern_match.end():]
            product_string = str(product)

            if before_match.rstrip() and before_match.rstrip()[-1] not in "+-=(,":
                if product >= 0:
                    product_string = "+ " + str(abs(product))
                else:
                    product_string = "- " + str(abs(product))

            updated_expression = _fix_pm(before_match + product_string + after_match)
            if updated_expression == current_expression:
                break
            result.append((updated_expression, None))
            current_expression = updated_expression

        final_expression = sp.latex(sp.simplify(sym_evaled))

        for _ in range(20):
            pattern_match = re.search(r"(-?\d+)\s*([+-])\s*(\d+)", current_expression)
            if not pattern_match:
                break

            first_number = int(pattern_match.group(1))
            arithmetic_operator = pattern_match.group(2)
            second_number = int(pattern_match.group(3))
            soma = first_number + second_number if arithmetic_operator == "+" else first_number - second_number

            new_expression = _fix_pm(
                current_expression[:pattern_match.start()]
                + str(soma)
                + current_expression[pattern_match.end():]
            )
            if new_expression == current_expression:
                break
            result.append((new_expression, None))
            current_expression = new_expression
            if current_expression == final_expression:
                break

        if current_expression != final_expression:
            result.append((final_expression, None))
        return result

    # ------------------------------------------------------------------
    # Fractional solution path
    # ------------------------------------------------------------------
    solution_numerator = final_value.p
    solution_denominator = final_value.q
    solution_latex = sp.latex(final_value)
    solution_latex_escaped = re.escape(solution_latex)

    if solution_numerator < 0:
        solution_numerator_str = f"({solution_numerator})"
    else:
        solution_numerator_str = str(solution_numerator)

    # Step 1a: integer * (p/q)
    int_times_frac_pattern = r"(-?\s*\d+)\s*\(\s*" + solution_latex_escaped + r"\s*\)"

    for _ in range(20):
        pattern_match = re.search(int_times_frac_pattern, current_expression)
        if not pattern_match:
            break

        term_coefficient = int(pattern_match.group(1).replace(" ", ""))
        numerator_product = term_coefficient * solution_numerator
        before_match = current_expression[:pattern_match.start()]
        after_match = current_expression[pattern_match.end():]

        frac_multiplication = (
            r"\frac{" + str(term_coefficient) + r" \cdot " + solution_numerator_str
            + r"}{" + str(solution_denominator) + r"}"
        )
        expr_with_multiplication = _fix_pm(before_match + frac_multiplication + after_match)

        if expr_with_multiplication != current_expression:
            result.append((expr_with_multiplication, None))
            current_expression = expr_with_multiplication

        numerator_evaluated = r"\frac{" + str(numerator_product) + r"}{" + str(solution_denominator) + r"}"
        fraction_reduced = sp.latex(sp.Rational(numerator_product, solution_denominator))
        simplification_is_integer = "\\frac" not in fraction_reduced

        if fraction_reduced == numerator_evaluated:
            expr_eval = _fix_pm(current_expression.replace(frac_multiplication, numerator_evaluated, 1))
            if expr_eval != current_expression:
                result.append((expr_eval, None))
                current_expression = expr_eval
        elif simplification_is_integer:
            expr_eval = _fix_pm(current_expression.replace(frac_multiplication, numerator_evaluated, 1))
            if expr_eval != current_expression:
                result.append((expr_eval, None))
                current_expression = expr_eval
            expr_reduced = _fix_pm(current_expression.replace(numerator_evaluated, fraction_reduced, 1))
            if expr_reduced != current_expression:
                result.append((expr_reduced, None))
                current_expression = expr_reduced
        else:
            expr_eval = _fix_pm(current_expression.replace(frac_multiplication, numerator_evaluated, 1))
            if expr_eval != current_expression:
                result.append((expr_eval, None))
                current_expression = expr_eval
            expr_reduced = _fix_pm(current_expression.replace(numerator_evaluated, fraction_reduced, 1))
            if expr_reduced != current_expression:
                result.append((expr_reduced, None))
                current_expression = expr_reduced

    # Step 1b: \frac{a}{b} * (p/q)
    fraction_times_fraction_pattern = (
        r"\\frac\{(\d+)\}\{(\d+)\}\s*\(\s*" + solution_latex_escaped + r"\s*\)"
    )

    for _ in range(20):
        pattern_match = re.search(fraction_times_fraction_pattern, current_expression)
        if not pattern_match:
            break

        frac_coef_numerator = int(pattern_match.group(1))
        frac_coef_denominator = int(pattern_match.group(2))
        frac_product_numerator = frac_coef_numerator * solution_numerator
        frac_product_denominator = frac_coef_denominator * solution_denominator
        before_match = current_expression[:pattern_match.start()]
        after_match = current_expression[pattern_match.end():]

        frac_product_shown = (
            r"\frac{" + str(frac_coef_numerator) + r" \cdot " + solution_numerator_str
            + r"}{" + str(frac_coef_denominator) + r" \cdot " + str(solution_denominator) + r"}"
        )
        expr_with_multiplication = _fix_pm(before_match + frac_product_shown + after_match)

        if expr_with_multiplication != current_expression:
            result.append((expr_with_multiplication, None))
            current_expression = expr_with_multiplication

        frac_numerator_evaluated = r"\frac{" + str(frac_product_numerator) + r"}{" + str(frac_product_denominator) + r"}"
        frac_fraction_reduced = sp.latex(sp.Rational(frac_product_numerator, frac_product_denominator))
        frac_simplification_is_integer = "\\frac" not in frac_fraction_reduced

        if frac_fraction_reduced == frac_numerator_evaluated:
            expr_eval = _fix_pm(current_expression.replace(frac_product_shown, frac_numerator_evaluated, 1))
            if expr_eval != current_expression:
                result.append((expr_eval, None))
                current_expression = expr_eval
        elif frac_simplification_is_integer:
            expr_reduced = _fix_pm(current_expression.replace(frac_product_shown, frac_fraction_reduced, 1))
            if expr_reduced != current_expression:
                result.append((expr_reduced, None))
                current_expression = expr_reduced
        else:
            expr_eval = _fix_pm(current_expression.replace(frac_product_shown, frac_numerator_evaluated, 1))
            if expr_eval != current_expression:
                result.append((expr_eval, None))
                current_expression = expr_eval
            expr_reduced = _fix_pm(current_expression.replace(frac_numerator_evaluated, frac_fraction_reduced, 1))
            if expr_reduced != current_expression:
                result.append((expr_reduced, None))
                current_expression = expr_reduced

    # Step 1c: simplify unreduced positive constant fractions
    constant_fraction_pattern = r"\\frac\{(\d+)\}\{(\d+)\}"

    for _ in range(20):
        has_reducible_fraction = False
        for pattern_match in re.finditer(constant_fraction_pattern, current_expression):
            const_frac_numerator = int(pattern_match.group(1))
            const_frac_denominator = int(pattern_match.group(2))
            common_divisor = math.gcd(const_frac_numerator, const_frac_denominator)

            if common_divisor > 1:
                const_frac_unreduced = r"\frac{" + str(const_frac_numerator) + r"}{" + str(const_frac_denominator) + r"}"
                if const_frac_denominator // common_divisor > 1:
                    const_frac_reduced = (
                        r"\frac{" + str(const_frac_numerator // common_divisor)
                        + r"}{" + str(const_frac_denominator // common_divisor) + r"}"
                    )
                else:
                    const_frac_reduced = str(const_frac_numerator // common_divisor)

                new_expression = _fix_pm(current_expression.replace(const_frac_unreduced, const_frac_reduced, 1))
                if new_expression != current_expression:
                    result.append((new_expression, None))
                    current_expression = new_expression
                    has_reducible_fraction = True
                    break

        if not has_reducible_fraction:
            break

    # Step 2: convert isolated integers to the common denominator
    found_denominators = _dens_in(current_expression)

    if found_denominators and _find_isolated_ints(current_expression):
        common_denominator = _freduce(_mlcm, found_denominators)
        result.append((current_expression, f"Reduce to common denominator ({common_denominator})"))

        for _ in range(30):
            found_integers = _find_isolated_ints(current_expression)
            if not found_integers:
                break

            start, end, integer_value = found_integers[0]
            converted_fraction = (
                r"\frac{" + str(integer_value * common_denominator)
                + r"}{" + str(common_denominator) + r"}"
            )
            new_expression = _fix_pm(current_expression[:start] + converted_fraction + current_expression[end:])
            if new_expression == current_expression:
                break
            result.append((new_expression, None))
            current_expression = new_expression

    # Step 3: add/subtract fractions pairwise
    fraction_pair_pattern = r"(\\frac\{(-?\d+)\}\{(\d+)\})\s*([+-])\s*(\\frac\{(-?\d+)\}\{(\d+)\})"

    for _ in range(30):
        pattern_match = re.search(fraction_pair_pattern, current_expression)
        if not pattern_match:
            break

        first_numerator = int(pattern_match.group(2))
        first_denominator = int(pattern_match.group(3))
        operator = pattern_match.group(4)
        second_numerator = int(pattern_match.group(6))
        second_denominator = int(pattern_match.group(7))

        before_match = current_expression[:pattern_match.start()].rstrip()

        if before_match.endswith("-"):
            first_numerator = -abs(first_numerator)
            current_expression = (
                current_expression[:len(before_match) - 1].rstrip()
                + " "
                + current_expression[len(before_match):]
            )
            pattern_match = re.search(fraction_pair_pattern, current_expression)
            if not pattern_match:
                break
            first_numerator = -abs(int(pattern_match.group(2)))
            first_denominator = int(pattern_match.group(3))
            operator = pattern_match.group(4)
            second_numerator = int(pattern_match.group(6))
            second_denominator = int(pattern_match.group(7))

        if first_denominator != second_denominator:
            from math import lcm as _lcm2
            common_denominator = _lcm2(first_denominator, second_denominator)
            first_num_conv = first_numerator * (common_denominator // first_denominator)
            second_num_conv = second_numerator * (common_denominator // second_denominator)

            first_frac_cd = r"\frac{" + str(first_num_conv) + r"}{" + str(common_denominator) + r"}"
            second_frac_cd = r"\frac{" + str(second_num_conv) + r"}{" + str(common_denominator) + r"}"

            new_expression = _fix_pm(
                current_expression[:pattern_match.start()]
                + first_frac_cd + " " + operator + " " + second_frac_cd
                + current_expression[pattern_match.end():]
            )
            if new_expression == current_expression:
                break
            result.append((new_expression, None))
            current_expression = new_expression
            continue

        if operator == "+":
            second_numerator_with_sign = second_numerator
            unevaluated_numerator_expression = f"{first_numerator} + {second_numerator}"
        else:
            second_numerator_with_sign = -second_numerator
            unevaluated_numerator_expression = f"{first_numerator} - {second_numerator}"

        grouped_fraction = r"\frac{" + unevaluated_numerator_expression + r"}{" + str(first_denominator) + r"}"
        expr_joined = _fix_pm(current_expression[:pattern_match.start()] + grouped_fraction + current_expression[pattern_match.end():])

        if expr_joined != current_expression:
            result.append((expr_joined, None))
            current_expression = expr_joined

        numerator_sum = first_numerator + second_numerator_with_sign
        frac_evaluated = r"\frac{" + str(numerator_sum) + r"}{" + str(first_denominator) + r"}"
        reduced_fraction = sp.latex(sp.Rational(numerator_sum, first_denominator))

        if frac_evaluated != grouped_fraction:
            new_expression_eval = _fix_pm(current_expression.replace(grouped_fraction, frac_evaluated, 1))
            if new_expression_eval != current_expression:
                result.append((new_expression_eval, None))
                current_expression = new_expression_eval

        if reduced_fraction != frac_evaluated:
            new_expression = _fix_pm(current_expression.replace(frac_evaluated, reduced_fraction, 1))
            if new_expression == current_expression:
                break
            result.append((new_expression, None))
            current_expression = new_expression
        else:
            if grouped_fraction in current_expression:
                new_expression = _fix_pm(current_expression.replace(grouped_fraction, reduced_fraction, 1))
            else:
                new_expression = current_expression
            if new_expression != current_expression:
                result.append((new_expression, None))
                current_expression = new_expression

    final_expression = sp.latex(sp.simplify(sym_evaled))
    if current_expression != final_expression:
        result.append((final_expression, None))

    deduplicated_steps = [result[0]]
    for item in result[1:]:
        step_latex, step_explanation = item
        if step_latex != deduplicated_steps[-1][0] or step_explanation is not None:
            deduplicated_steps.append(item)

    return deduplicated_steps


def check_solution(final_value, final_latex, equation, steps):
    """
    Verify the solution by substituting x = final_value back into the
    original equation and simplifying both sides step by step.

    Animation sequence:
      1. Show the original equation with a "Let's verify!" message.
      2. Announce the substitution value.
      3. Replace every x with the numeric value.
      4. Simplify the left side arithmetic step by step.
      5. Simplify the right side arithmetic step by step.
      6. Show a confirmation message (correct / does not satisfy).
    """
    left_str, right_str = equation.split("=")
    left_sym  = sp.sympify(normalize_expression(left_str.strip()),  evaluate=False)
    right_sym = sp.sympify(normalize_expression(right_str.strip()), evaluate=False)

    def _fracs_to_latex(expression):
        return re.sub(
            r"(-?\d+)/(\d+)",
            lambda m: r"\frac{" + m.group(1) + r"}{" + m.group(2) + r"}",
            expression,
        )

    left_display  = _fracs_to_latex(left_str.strip())
    right_display = _fracs_to_latex(right_str.strip())

    left_display  = re.sub(r"(\\frac\{\d+\}\{\d+\})(\s*)([a-zA-Z])", r"\1 \3", left_display)
    right_display = re.sub(r"(\\frac\{\d+\}\{\d+\})(\s*)([a-zA-Z])", r"\1 \3", right_display)

    equation_display = f"{left_display} = {right_display}"

    steps.append(Step(before=equation_display, after=equation_display, explanation="Let's verify!"))
    steps.append(Step(before=equation_display, after=equation_display, explanation=f"Now substitute x for {final_latex}"))

    solution_latex = sp.latex(final_value)

    def _sub_x(expression, x_value):
        return re.sub(
            r"(?<![a-zA-Z])x(?![a-zA-Z])",
            lambda _match: "(" + x_value + ")",
            expression,
        )

    left_substituted  = _sub_x(left_display,  solution_latex)
    right_substituted = _sub_x(right_display, solution_latex)

    steps.append(Step(before=equation_display, after=f"{left_substituted} = {right_substituted}"))

    def _subst_terms(sympy_expression, x_value):
        if isinstance(sympy_expression, sp.Add):
            new_terms = [term.xreplace({x: sp.UnevaluatedExpr(x_value)}) for term in sympy_expression.args]
            return sp.Add(*new_terms, evaluate=False)
        return sympy_expression.xreplace({x: sp.UnevaluatedExpr(x_value)})

    left_unevaluated  = _subst_terms(left_sym,  final_value)
    right_unevaluated = _subst_terms(right_sym, final_value)

    left_tuples  = sympy_stepwise(left_substituted,  left_unevaluated,  final_value)
    right_tuples = sympy_stepwise(right_substituted, right_unevaluated, final_value)

    def _extract(tuples):
        step_list, explanations = [], {}
        for i, (step_str, step_explanation) in enumerate(tuples):
            step_list.append(step_str)
            if step_explanation:
                explanations[i] = step_explanation
        return step_list, explanations

    left_steps_v,  left_explanations  = _extract(left_tuples)
    right_steps_v, right_explanations = _extract(right_tuples)

    current_left, current_right = left_substituted, right_substituted

    for i, left_after in enumerate(left_steps_v):
        current_explanation = left_explanations.get(i)
        equation_before = f"{current_left} = {current_right}"

        if current_explanation:
            steps.append(Step(before=equation_before, after=equation_before, explanation=current_explanation))

        equation_after = f"{left_after} = {current_right}"
        if equation_before != equation_after:
            steps.append(Step(before=equation_before, after=equation_after))

        current_left = left_after

    for i, right_after in enumerate(right_steps_v):
        current_explanation = right_explanations.get(i)
        equation_before = f"{current_left} = {current_right}"

        if current_explanation:
            steps.append(Step(before=equation_before, after=equation_before, explanation=current_explanation))

        equation_after = f"{current_left} = {right_after}"
        if equation_before != equation_after:
            steps.append(Step(before=equation_before, after=equation_after))

        current_right = right_after

    left_orig  = sp.sympify(normalize_expression(left_str.strip()),  evaluate=False)
    right_orig = sp.sympify(normalize_expression(right_str.strip()), evaluate=False)
    is_true = sp.simplify(left_orig.subs(x, final_value) - right_orig.subs(x, final_value)) == 0

    explanation = "The solution is correct!" if is_true else "The solution does not satisfy the equation."
    steps.append(Step(before=f"{current_left} = {current_right}", after=f"{current_left} = {current_right}", explanation=explanation))

