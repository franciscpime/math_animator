import re
import math
import sympy as sp
from math import lcm as _mlcm
from functools import reduce as _freduce
from models.step import Step
from parser.equation_parser import normalize_expression

x = sp.symbols("x")

'''
This function normalises double negatives and plus-minus combinations in a mathematical expression.

Example: - - x + - y  >>  + x - y
'''
def fix_sign_combinations(expression):
    expression = re.sub(r"-\s*-\s*", "+ ", expression)
    expression = re.sub(r"\+\s*-\s*", "- ", expression)

    return expression.strip()


'''
This function generates a stepwise solution using SymPy for the given expression.

For a fractional final_value = p/q the sequence per term is:
      Step 1a  integer_coef * (p/q)  >> show product explicitly >> evaluate
        >>> example: 2 * (3/4)  >>  (2*3)/4  >>  6/4  >>  3/2

      Step 1b  frac_coef * (p/q)    >> show product explicitly >> evaluate
        >>> example: 3/5 * (3/4)  >>  (3*3)/(5*4)  >>  9/20

      Step 1c  simplify any remaining unreduced constant fractions
        >>> example: 6/4  >>  3/2

      Step 2   convert isolated integers to the common denominator
        >>> example: 3 * (4/5)  >>  (3*5)/(1*5)  >>  15/5

      Step 3   add/subtract fractions pairwise
        >>> example: 2/3 + 1/4  >>  (2*4)/(3*4) + (1*3)/(4*3)  >>  8/12 + 3/12  >>  11/12

For an integer final_value:
      Step 1   evaluate each product k*(val)
        >>> example: 2 * (3)  >>  6

      Step 2   sum numeric pairs left to right
        >>> example: 2 + 3  >>  5
'''
def sympy_stepwise(substituted_expression, sym_evaled, final_value):
    
    """
    Find all integers in a LaTeX expression string that are:
        - Outside of any LaTeX braces  (e.g. not inside \frac{...}{...})
        - Not immediately followed by / or {  (which would make them part of a fraction)

    Returns a list of tuples: (start_index, end_index, integer_value)
    Example: '3 + \frac{1}{2}'  >>  [(0, 1, 3)]   -- only 3 is isolated; 1 and 2 are inside braces
    """
    def _find_isolated_ints(expression):

        # Will hold all found isolated integers as (start, end, value) tuples
        isolated_integers = []

        # Tracks how deep inside LaTeX braces we currently are
        # Example: '\frac{1}{2}'  >>  brace_level goes 0 >> 1 >> 0 >> 1 >> 0
        brace_level = 0

        # Current position in the expression string
        i = 0

        # Walk through every character in the expression one by one
        while i < len(expression):

            # Read the character at the current position
            current_char = expression[i]

            # Opening brace -- we are going one level deeper inside a LaTeX command
            # Example: '\frac{' -- after the { we are inside the numerator
            if current_char == "{":
                brace_level += 1
                i += 1
                continue

            # Closing brace -- we are coming one level back out
            # Example: '\frac{1}' -- after the } we are back outside the numerator
            if current_char == "}":
                brace_level -= 1
                i += 1
                continue

            # If we are inside any braces, skip this character entirely --
            # integers inside braces belong to \frac or similar and are not isolated
            if brace_level > 0:
                i += 1
                continue

            # Assume positive sign until we find a minus
            sign = 1

            # Remember where this potential number started (including any leading minus)
            start = i

            # Check if the current character is a minus sign followed by a digit --
            # this could be the start of a negative number
            if current_char == "-" and i + 1 < len(expression) and expression[i + 1].isdigit():

                # Only treat this minus as a negative sign if it appears at the very
                # start of the expression or right after an operator/opening symbol.
                # Example: '3 - 2'  >>  the '-' belongs to the operator, not the number
                #          '= -2'   >>  the '-' belongs to the number
                if i == 0 or expression[i - 1] in " +=({":
                    sign = -1
                    i += 1
                    # Move past the minus so current_char is now the first digit
                    current_char = expression[i]
                else:
                    # This minus is an operator between two numbers -- skip it
                    i += 1
                    continue

            # If the current character is a digit, we may have found an integer
            if current_char.isdigit():

                # Scan forward to find where the number ends
                number_end = i
                while number_end < len(expression) and expression[number_end].isdigit():
                    number_end += 1

                # If the number is immediately followed by / or {, it is part of a
                # fraction like '1/2' or '\frac{1}' -- skip it, it is not isolated
                if number_end < len(expression) and expression[number_end] in "/{":
                    i = number_end
                    continue

                # Try to parse the digit sequence as an integer
                try:
                    integer_value = sign * int(expression[i:number_end])
                except Exception:
                    # Could not parse -- move on to the next character
                    i += 1
                    continue

                # Only record non-zero integers -- zero is not useful for our purposes
                if integer_value != 0:
                    isolated_integers.append((start, number_end, integer_value))

                # Jump past all the digits we just consumed
                i = number_end

            else:
                # Not a digit, not a brace, not a minus -- just move on
                i += 1

        return isolated_integers


    '''
    This function extracts the denominators from a LaTeX fraction expression.

    Example: '\frac{1}{2}' >> [2]
    '''
    def extract_denominators(expression):
        return [int(denominator) for denominator in re.findall(r"\\frac\{[^}]+\}\{(\d+)\}", expression)]

    # Start the result list with the original substituted expression.
    # The None indicates there is no explanation text for this first entry.
    # Example: '2 (\frac{1}{2}) + 3'  >>  [('2 (\frac{1}{2}) + 3', None)]
    result = [(substituted_expression, None)]

    # Keep a running copy of the expression as we simplify it step by step.
    # This gets updated after every change so each step starts from the latest state.
    current_expression = substituted_expression

    # ------------------------------------------------------------------
    # Integer solution path
    # Used when the solution is a whole number (e.g. x = 3, x = -2)
    # ------------------------------------------------------------------

    # Check if the final value is NOT a proper fraction (denominator == 1 means it's a whole number)
    # Example: final_value = 3       >>  not a fraction  >>  enter integer path
    #          final_value = 1/2     >>  is a fraction   >>  skip to fractional path
    if not (isinstance(final_value, sp.Rational) and final_value.q != 1):

        # Convert the solution to a plain Python int for arithmetic
        # Example: sp.Integer(3)  >>  solution_as_int = 3
        solution_as_int = int(final_value)

        # Get the LaTeX string of the solution for display purposes
        # Example: sp.Integer(3)  >>  solution_latex = '3'
        solution_latex = sp.latex(final_value)

        # Escape any special regex characters in the LaTeX string so it can be
        # used safely inside a regex pattern
        # Example: '-3'  >>  solution_latex_escaped = '\\-3'
        solution_latex_escaped = re.escape(solution_latex)

        # Build the regex pattern that matches expressions like '2(3)' or '-4(3)'
        # i.e. an integer coefficient immediately followed by the solution in parentheses
        # Example: pattern matches '2(3)' in '2(3) + 5'
        pattern = r"(-?\s*\d+)\s*\(\s*" + solution_latex_escaped + r"\s*\)"

        # Repeatedly find and evaluate products of the form k*(integer) until none remain
        # The loop limit of 20 prevents infinite loops in unexpected edge cases
        for _ in range(20):

            # Search for the next occurrence of the pattern in the current expression
            pattern_match = re.search(pattern, current_expression)

            # No more matches -- all products have been evaluated, exit the loop
            if not pattern_match:
                break

            # Extract the integer coefficient from the match
            # Example: '2(3)'  >>  term_coefficient = 2
            term_coefficient = int(pattern_match.group(1).replace(" ", ""))

            # Multiply the coefficient by the solution to get the evaluated product
            # Example: 2 * 3  >>  product = 6
            product = term_coefficient * solution_as_int

            # Everything before the matched pattern -- used to reconstruct the expression
            # Example: '5 + 2(3) - 1'  >>  before_match = '5 + '
            before_match = current_expression[:pattern_match.start()]

            # Everything after the matched pattern
            # Example: '5 + 2(3) - 1'  >>  after_match = ' - 1'
            after_match = current_expression[pattern_match.end():]

            # Start with the product as a plain string
            # Example: product = 6  >>  product_string = '6'
            product_string = str(product)

            # If the product is not at the start of the expression and not right after
            # an operator, we need to add an explicit + or - sign before it
            # Example: '5 + 6'  >>  the 6 needs a '+' prefix so it reads correctly
            if before_match.rstrip() and before_match.rstrip()[-1] not in "+-=(,":

                # Positive product -- add an explicit '+' sign
                # Example: product = 6  >>  product_string = '+ 6'
                if product >= 0:
                    product_string = "+ " + str(abs(product))

                # Negative product -- add an explicit '-' sign
                # Example: product = -6  >>  product_string = '- 6'
                else:
                    product_string = "- " + str(abs(product))

            # Reconstruct the expression with the product replacing the matched pattern,
            # then clean up any double negatives or plus-minus combinations
            # Example: '5 + 2(3) - 1'  >>  '5 + 6 - 1'
            updated_expression = fix_sign_combinations(before_match + product_string + after_match)

            # If nothing changed, stop -- avoids infinite loops
            if updated_expression == current_expression:
                break

            # Record this step and update the current expression
            result.append((updated_expression, None))
            current_expression = updated_expression

        # Ask SymPy for the fully simplified final result to use as a target
        # Example: sym_evaled = 3 + 2  >>  final_expression = '5'
        final_expression = sp.latex(sp.simplify(sym_evaled))

        # Repeatedly sum pairs of integers until only one number remains
        # Example: '5 + 6 - 1'  >>  '11 - 1'  >>  '10'
        for _ in range(20):

            # Search for the next pair of integers connected by + or -
            # Example: '5 + 6 - 1'  >>  matches '5 + 6'
            pattern_match = re.search(r"(-?\d+)\s*([+-])\s*(\d+)", current_expression)

            # No more pairs to combine -- exit the loop
            if not pattern_match:
                break

            # Extract both numbers and the operator between them
            first_number       = int(pattern_match.group(1))
            arithmetic_operator = pattern_match.group(2)
            second_number      = int(pattern_match.group(3))

            # Compute the result of the operation
            # Example: 5 + 6  >>  soma = 11
            #          11 - 1  >>  soma = 10
            if arithmetic_operator == "+":
                soma = first_number + second_number
            else:
                soma = first_number - second_number

            # Replace the matched pair with the computed sum in the expression
            new_expression = fix_sign_combinations(
                current_expression[:pattern_match.start()]
                + str(soma)
                + current_expression[pattern_match.end():]
            )

            # If nothing changed, stop -- avoids infinite loops
            if new_expression == current_expression:
                break

            # Record this step and update the current expression
            result.append((new_expression, None))
            current_expression = new_expression

            # If we have already reached the final simplified form, stop early
            if current_expression == final_expression:
                break

        # If the current expression still differs from SymPy's final result,
        # jump straight to it as a fallback to guarantee correctness
        if current_expression != final_expression:
            result.append((final_expression, None))

        # Return early -- the integer path is complete
        return result

    # ------------------------------------------------------------------
    # Fractional solution path
    # Used when the solution is a proper fraction (e.g. x = 1/2, x = -3/4)
    # ------------------------------------------------------------------

    # Extract the numerator of the fractional solution
    # Example: final_value = 1/2  >>  solution_numerator = 1
    solution_numerator = final_value.p

    # Extract the denominator of the fractional solution
    # Example: final_value = 1/2  >>  solution_denominator = 2
    solution_denominator = final_value.q

    # Get the LaTeX string of the solution for display purposes
    # Example: Rational(1, 2)  >>  solution_latex = '\frac{1}{2}'
    solution_latex = sp.latex(final_value)

    # Escape any special regex characters so the LaTeX string is safe to use in patterns
    solution_latex_escaped = re.escape(solution_latex)

    # Wrap a negative numerator in parentheses for visual clarity in the display
    # Example: solution_numerator = -1  >>  solution_numerator_str = '(-1)'
    #          solution_numerator =  1  >>  solution_numerator_str = '1'
    if solution_numerator < 0:
        solution_numerator_str = f"({solution_numerator})"
    else:
        solution_numerator_str = str(solution_numerator)

    # ---------------------------------------------------------------
    # Step 1a: handle patterns like '3 * (p/q)' -- integer times a fraction
    # Example: '2(\frac{1}{2})'  >>  '\frac{2 \cdot 1}{2}'  >>  '\frac{2}{2}'  >>  '1'
    # ---------------------------------------------------------------

    # Build the regex pattern that matches an integer coefficient followed by the solution in parentheses
    # Example: matches '2(\frac{1}{2})' in '2(\frac{1}{2}) + 3'
    int_times_frac_pattern = r"(-?\s*\d+)\s*\(\s*" + solution_latex_escaped + r"\s*\)"

    # Repeatedly find and expand products of the form k*(p/q) until none remain
    for _ in range(20):

        # Search for the next occurrence of the pattern
        pattern_match = re.search(int_times_frac_pattern, current_expression)

        # No more matches -- all integer-times-fraction products have been handled
        if not pattern_match:
            break

        # Extract the integer coefficient from the match
        # Example: '2(\frac{1}{2})'  >>  term_coefficient = 2
        term_coefficient = int(pattern_match.group(1).replace(" ", ""))

        # Multiply the integer coefficient by the solution numerator
        # Example: 2 * 1  >>  numerator_product = 2
        numerator_product = term_coefficient * solution_numerator

        # Everything before the matched pattern
        before_match = current_expression[:pattern_match.start()]

        # Everything after the matched pattern
        after_match = current_expression[pattern_match.end():]

        # Build the LaTeX showing the multiplication explicitly before evaluating it
        # Example: '\frac{2 \cdot 1}{2}'
        frac_multiplication = (
            r"\frac{" + str(term_coefficient) + r" \cdot " + solution_numerator_str
            + r"}{" + str(solution_denominator) + r"}"
        )

        # Replace the matched pattern with the explicit multiplication form
        expr_with_multiplication = fix_sign_combinations(before_match + frac_multiplication + after_match)

        # Only record this step if something actually changed
        if expr_with_multiplication != current_expression:
            result.append((expr_with_multiplication, None))
            current_expression = expr_with_multiplication

        # Build the LaTeX with the numerator evaluated (but not yet reduced)
        # Example: '\frac{2}{2}'
        numerator_evaluated = r"\frac{" + str(numerator_product) + r"}{" + str(solution_denominator) + r"}"

        # Ask SymPy for the fully reduced form
        # Example: Rational(2, 2)  >>  fraction_reduced = '1'
        fraction_reduced = sp.latex(sp.Rational(numerator_product, solution_denominator))

        # Check whether the reduced form is a whole number (no \frac in the string)
        # Example: '1'  >>  simplification_is_integer = True
        #          '\frac{1}{2}'  >>  simplification_is_integer = False
        simplification_is_integer = "\\frac" not in fraction_reduced

        # Sub-case: already in lowest terms -- just show the evaluated numerator
        # Example: '\frac{1}{3}'  is already reduced  >>  show '\frac{1}{3}' and move on
        if fraction_reduced == numerator_evaluated:
            expr_eval = fix_sign_combinations(current_expression.replace(frac_multiplication, numerator_evaluated, 1))
            if expr_eval != current_expression:
                result.append((expr_eval, None))
                current_expression = expr_eval

        # Sub-case: the fraction reduces to a whole integer
        # Example: '\frac{2}{2}'  >>  show '\frac{2}{2}'  >>  then show '1'
        elif simplification_is_integer:
            expr_eval = fix_sign_combinations(current_expression.replace(frac_multiplication, numerator_evaluated, 1))

            if expr_eval != current_expression:
                result.append((expr_eval, None))
                current_expression = expr_eval

            expr_reduced = fix_sign_combinations(current_expression.replace(numerator_evaluated, fraction_reduced, 1))

            if expr_reduced != current_expression:
                result.append((expr_reduced, None))
                current_expression = expr_reduced

        # Sub-case: the fraction reduces to a simpler fraction
        # Example: '\frac{4}{6}'  >>  show '\frac{4}{6}'  >>  then show '\frac{2}{3}'
        else:
            expr_eval = fix_sign_combinations(current_expression.replace(frac_multiplication, numerator_evaluated, 1))

            if expr_eval != current_expression:
                result.append((expr_eval, None))
                current_expression = expr_eval

            expr_reduced = fix_sign_combinations(current_expression.replace(numerator_evaluated, fraction_reduced, 1))

            if expr_reduced != current_expression:
                result.append((expr_reduced, None))
                current_expression = expr_reduced

    # ---------------------------------------------------------------
    # Step 1b: handle patterns like '\frac{a}{b} * (p/q)' -- fraction times a fraction
    # Example: '\frac{3}{5}(\frac{1}{2})'  >>  '\frac{3 \cdot 1}{5 \cdot 2}'  >>  '\frac{3}{10}'
    # ---------------------------------------------------------------

    # Build the regex pattern that matches a LaTeX fraction coefficient followed by the solution in parentheses
    fraction_times_fraction_pattern = (
        r"\\frac\{(\d+)\}\{(\d+)\}\s*\(\s*" + solution_latex_escaped + r"\s*\)"
    )

    # Repeatedly find and expand products of the form (a/b)*(p/q) until none remain
    for _ in range(20):

        # Search for the next occurrence of the pattern
        pattern_match = re.search(fraction_times_fraction_pattern, current_expression)

        # No more matches -- all fraction-times-fraction products have been handled
        if not pattern_match:
            break

        # Extract the numerator of the coefficient fraction
        # Example: '\frac{3}{5}(\frac{1}{2})'  >>  frac_coef_numerator = 3
        frac_coef_numerator = int(pattern_match.group(1))

        # Extract the denominator of the coefficient fraction
        # Example: '\frac{3}{5}(\frac{1}{2})'  >>  frac_coef_denominator = 5
        frac_coef_denominator = int(pattern_match.group(2))

        # Multiply the two numerators together
        # Example: 3 * 1  >>  frac_product_numerator = 3
        frac_product_numerator = frac_coef_numerator * solution_numerator

        # Multiply the two denominators together
        # Example: 5 * 2  >>  frac_product_denominator = 10
        frac_product_denominator = frac_coef_denominator * solution_denominator

        # Everything before the matched pattern
        before_match = current_expression[:pattern_match.start()]

        # Everything after the matched pattern
        after_match = current_expression[pattern_match.end():]

        # Build the LaTeX showing both multiplications explicitly before evaluating
        # Example: '\frac{3 \cdot 1}{5 \cdot 2}'
        frac_product_shown = (
            r"\frac{" + str(frac_coef_numerator) + r" \cdot " + solution_numerator_str
            + r"}{" + str(frac_coef_denominator) + r" \cdot " + str(solution_denominator) + r"}"
        )

        # Replace the matched pattern with the explicit multiplication form
        expr_with_multiplication = fix_sign_combinations(before_match + frac_product_shown + after_match)

        # Only record this step if something actually changed
        if expr_with_multiplication != current_expression:
            result.append((expr_with_multiplication, None))
            current_expression = expr_with_multiplication

        # Build the LaTeX with both products evaluated (but not yet reduced)
        # Example: '\frac{3}{10}'
        frac_numerator_evaluated = r"\frac{" + str(frac_product_numerator) + r"}{" + str(frac_product_denominator) + r"}"

        # Ask SymPy for the fully reduced form
        # Example: Rational(3, 10)  >>  frac_fraction_reduced = '\frac{3}{10}'
        frac_fraction_reduced = sp.latex(sp.Rational(frac_product_numerator, frac_product_denominator))

        # Check whether the reduced form is a whole number
        frac_simplification_is_integer = "\\frac" not in frac_fraction_reduced

        # Sub-case: already in lowest terms -- just show the evaluated form
        if frac_fraction_reduced == frac_numerator_evaluated:
            expr_eval = fix_sign_combinations(current_expression.replace(frac_product_shown, frac_numerator_evaluated, 1))

            if expr_eval != current_expression:
                result.append((expr_eval, None))
                current_expression = expr_eval

        # Sub-case: the fraction reduces to a whole integer -- jump straight to it
        # Example: '\frac{4}{2}'  >>  '2'
        elif frac_simplification_is_integer:
            expr_reduced = fix_sign_combinations(current_expression.replace(frac_product_shown, frac_fraction_reduced, 1))

            if expr_reduced != current_expression:
                result.append((expr_reduced, None))
                current_expression = expr_reduced

        # Sub-case: the fraction reduces to a simpler fraction
        # Example: '\frac{6}{10}'  >>  show '\frac{6}{10}'  >>  then show '\frac{3}{5}'
        else:
            expr_eval = fix_sign_combinations(current_expression.replace(frac_product_shown, frac_numerator_evaluated, 1))

            if expr_eval != current_expression:
                result.append((expr_eval, None))
                current_expression = expr_eval

            expr_reduced = fix_sign_combinations(current_expression.replace(frac_numerator_evaluated, frac_fraction_reduced, 1))

            if expr_reduced != current_expression:
                result.append((expr_reduced, None))
                current_expression = expr_reduced

    # ---------------------------------------------------------------
    # Step 1c: simplify any unreduced positive constant fractions still present
    # These come from the original equation text which is preserved verbatim
    # Example: '\frac{6}{4}'  >>  '\frac{3}{2}'
    # ---------------------------------------------------------------

    # Build the regex pattern that matches any positive LaTeX fraction
    constant_fraction_pattern = r"\\frac\{(\d+)\}\{(\d+)\}"

    # Repeatedly scan for reducible fractions until none remain
    # The outer loop runs up to 20 times to handle multiple fractions
    for _ in range(20):

        # Flag to track whether we found and reduced any fraction in this pass
        has_reducible_fraction = False

        # Scan all fractions in the current expression
        for pattern_match in re.finditer(constant_fraction_pattern, current_expression):

            # Extract numerator and denominator of the current fraction
            const_frac_numerator   = int(pattern_match.group(1))
            const_frac_denominator = int(pattern_match.group(2))

            # Find the GCD to check whether the fraction can be reduced
            common_divisor = math.gcd(const_frac_numerator, const_frac_denominator)

            # Only proceed if the fraction is not already in lowest terms
            if common_divisor > 1:

                # Build the unreduced fraction string to replace
                # Example: '\frac{6}{4}'
                const_frac_unreduced = r"\frac{" + str(const_frac_numerator) + r"}{" + str(const_frac_denominator) + r"}"

                # If the reduced denominator is still greater than 1, keep it as a fraction
                # Example: '\frac{6}{4}'  >>  '\frac{3}{2}'
                if const_frac_denominator // common_divisor > 1:
                    const_frac_reduced = (
                        r"\frac{" + str(const_frac_numerator // common_divisor)
                        + r"}{" + str(const_frac_denominator // common_divisor) + r"}"
                    )

                # If the reduced denominator is 1, show the result as a plain integer
                # Example: '\frac{6}{3}'  >>  '2'
                else:
                    const_frac_reduced = str(const_frac_numerator // common_divisor)

                # Replace the unreduced fraction with the reduced form in the expression
                new_expression = fix_sign_combinations(current_expression.replace(const_frac_unreduced, const_frac_reduced, 1))

                # Only record this step if something actually changed
                if new_expression != current_expression:
                    result.append((new_expression, None))
                    current_expression = new_expression
                    has_reducible_fraction = True
                    # Break the inner loop and restart the outer loop to scan again from scratch
                    break

        # If no reducible fraction was found in this pass, all fractions are in lowest terms
        if not has_reducible_fraction:
            break

    # ---------------------------------------------------------------
    # Step 2: convert isolated integers to the common denominator so
    # they can be added to the fractions in Step 3
    # Example: '3 + \frac{1}{2}'  >>  '\frac{6}{2} + \frac{1}{2}'
    # ---------------------------------------------------------------

    # Find all denominators currently present in the expression
    # Example: '3 + \frac{1}{2} + \frac{1}{4}'  >>  found_denominators = [2, 4]
    found_denominators = extract_denominators(current_expression)

    # Only proceed if there are both fractions and isolated integers to convert
    if found_denominators and _find_isolated_ints(current_expression):

        # Find the least common multiple of all denominators
        # Example: lcm(2, 4)  >>  common_denominator = 4
        common_denominator = _freduce(_mlcm, found_denominators)

        # Emit an explanation step before converting -- tells the viewer what is about to happen
        result.append((current_expression, f"Reduce to common denominator ({common_denominator})"))

        # Convert one isolated integer at a time until none remain
        for _ in range(30):

            # Find all remaining isolated integers in the current expression
            found_integers = _find_isolated_ints(current_expression)

            # No more isolated integers to convert -- exit the loop
            if not found_integers:
                break

            # Take the first isolated integer found
            # Example: '3 + \frac{1}{2}'  >>  start=0, end=1, integer_value=3
            start, end, integer_value = found_integers[0]

            # Rewrite the integer as a fraction over the common denominator
            # Example: 3 with common_denominator=4  >>  '\frac{12}{4}'
            converted_fraction = (
                r"\frac{" + str(integer_value * common_denominator)
                + r"}{" + str(common_denominator) + r"}"
            )

            # Replace the integer with the converted fraction in the expression
            new_expression = fix_sign_combinations(
                current_expression[:start] + converted_fraction + current_expression[end:]
            )

            # If nothing changed, stop -- avoids infinite loops
            if new_expression == current_expression:
                break

            # Record this step and update the current expression
            result.append((new_expression, None))
            current_expression = new_expression

    # ---------------------------------------------------------------
    # Step 3: add or subtract fractions pairwise until only one remains
    # Example: '\frac{6}{4} + \frac{1}{4}'  >>  '\frac{6 + 1}{4}'  >>  '\frac{7}{4}'
    # ---------------------------------------------------------------

    # Build the regex pattern that matches two adjacent LaTeX fractions connected by + or -
    fraction_pair_pattern = r"(\\frac\{(-?\d+)\}\{(\d+)\})\s*([+-])\s*(\\frac\{(-?\d+)\}\{(\d+)\})"

    # Repeatedly combine pairs of fractions until only one fraction remains
    for _ in range(30):

        # Search for the next adjacent pair of fractions
        pattern_match = re.search(fraction_pair_pattern, current_expression)

        # No more pairs to combine -- exit the loop
        if not pattern_match:
            break

        # Extract the numerator of the first fraction
        # Example: '\frac{6}{4} + \frac{1}{4}'  >>  first_numerator = 6
        first_numerator   = int(pattern_match.group(2))

        # Extract the denominator of the first fraction
        # Example: first_denominator = 4
        first_denominator = int(pattern_match.group(3))

        # Extract the operator between the two fractions (+ or -)
        # Example: operator = '+'
        operator = pattern_match.group(4)

        # Extract the numerator of the second fraction
        # Example: second_numerator = 1
        second_numerator   = int(pattern_match.group(6))

        # Extract the denominator of the second fraction
        # Example: second_denominator = 4
        second_denominator = int(pattern_match.group(7))

        # Check what comes immediately before the matched pair
        before_match = current_expression[:pattern_match.start()].rstrip()

        # If a minus sign precedes the entire pair, absorb it into the first numerator
        # Example: '- \frac{6}{4} + \frac{1}{4}'  >>  first_numerator becomes -6
        if before_match.endswith("-"):

            # Make the first numerator negative
            first_numerator = -abs(first_numerator)

            # Remove the leading minus from the expression to avoid double negation
            current_expression = (
                current_expression[:len(before_match) - 1].rstrip()
                + " "
                + current_expression[len(before_match):]
            )

            # Re-search to get updated match positions after modifying the expression
            pattern_match = re.search(fraction_pair_pattern, current_expression)

            # If the pattern is no longer found after the edit, exit the loop
            if not pattern_match:
                break

            # Re-extract all values from the updated match
            first_numerator = -abs(int(pattern_match.group(2)))
            first_denominator = int(pattern_match.group(3))
            operator = pattern_match.group(4)
            second_numerator = int(pattern_match.group(6))
            second_denominator = int(pattern_match.group(7))

        # If the two fractions have different denominators, align them first
        # Example: '\frac{1}{2} + \frac{1}{3}'  >>  '\frac{3}{6} + \frac{2}{6}'
        if first_denominator != second_denominator:

            # Import lcm locally to find the least common denominator
            from math import lcm as _lcm2
            common_denominator = _lcm2(first_denominator, second_denominator)

            # Scale the first numerator up to the common denominator
            # Example: 1 * (6 // 2) = 3
            first_num_conv  = first_numerator  * (common_denominator // first_denominator)

            # Scale the second numerator up to the common denominator
            # Example: 1 * (6 // 3) = 2
            second_num_conv = second_numerator * (common_denominator // second_denominator)

            # Build the two converted fractions over the common denominator
            first_frac_cd  = r"\frac{" + str(first_num_conv)  + r"}{" + str(common_denominator) + r"}"
            second_frac_cd = r"\frac{" + str(second_num_conv) + r"}{" + str(common_denominator) + r"}"

            # Replace the original pair with the converted pair in the expression
            new_expression = fix_sign_combinations(
                current_expression[:pattern_match.start()]
                + first_frac_cd + " " + operator + " " + second_frac_cd
                + current_expression[pattern_match.end():]
            )

            # If nothing changed, stop -- avoids infinite loops
            if new_expression == current_expression:
                break

            # Record this step and update the current expression
            result.append((new_expression, None))
            current_expression = new_expression

            # Restart the loop to process the newly aligned fractions
            continue

        # Both fractions share the same denominator -- combine their numerators
        # Determine the sign of the second numerator based on the operator
        if operator == "+":
            # Adding: second numerator keeps its sign
            second_numerator_with_sign = second_numerator
            unevaluated_numerator_expression = f"{first_numerator} + {second_numerator}"
        else:
            # Subtracting: second numerator becomes negative
            second_numerator_with_sign = -second_numerator
            unevaluated_numerator_expression = f"{first_numerator} - {second_numerator}"

        # Build the grouped fraction showing the unevaluated numerator sum
        # Example: '\frac{6 + 1}{4}'
        grouped_fraction = r"\frac{" + unevaluated_numerator_expression + r"}{" + str(first_denominator) + r"}"

        # Replace the two separate fractions with the grouped unevaluated form
        expr_joined = fix_sign_combinations(
            current_expression[:pattern_match.start()] + grouped_fraction + current_expression[pattern_match.end():]
        )

        # Only record this step if something actually changed
        if expr_joined != current_expression:
            result.append((expr_joined, None))
            current_expression = expr_joined

        # Evaluate the numerator sum
        # Example: 6 + 1  >>  numerator_sum = 7
        numerator_sum = first_numerator + second_numerator_with_sign

        # Build the fraction with the evaluated numerator
        # Example: '\frac{7}{4}'
        frac_evaluated = r"\frac{" + str(numerator_sum) + r"}{" + str(first_denominator) + r"}"

        # Ask SymPy for the fully reduced form
        # Example: Rational(7, 4)  >>  reduced_fraction = '\frac{7}{4}'
        reduced_fraction = sp.latex(sp.Rational(numerator_sum, first_denominator))

        # Show the evaluated numerator as an intermediate step before reducing
        if frac_evaluated != grouped_fraction:
            new_expression_eval = fix_sign_combinations(current_expression.replace(grouped_fraction, frac_evaluated, 1))

            if new_expression_eval != current_expression:
                result.append((new_expression_eval, None))
                current_expression = new_expression_eval

        # If the fraction reduced further, show the reduced form
        # Example: '\frac{4}{6}'  >>  '\frac{2}{3}'
        if reduced_fraction != frac_evaluated:
            new_expression = fix_sign_combinations(current_expression.replace(frac_evaluated, reduced_fraction, 1))

            if new_expression == current_expression:
                break

            result.append((new_expression, None))
            current_expression = new_expression

        else:
            # Fraction was already in lowest terms -- replace the grouped form with the evaluated form
            if grouped_fraction in current_expression:
                new_expression = fix_sign_combinations(current_expression.replace(grouped_fraction, reduced_fraction, 1))

            else:
                new_expression = current_expression

            if new_expression != current_expression:
                result.append((new_expression, None))
                current_expression = new_expression

    # Ask SymPy for the definitive final result as a fallback safety net
    # Example: sym_evaled = Rational(7, 4)  >>  final_expression = '\frac{7}{4}'
    final_expression = sp.latex(sp.simplify(sym_evaled))

    # If the current expression still differs from the final, jump straight to it
    if current_expression != final_expression:
        result.append((final_expression, None))

    # Remove consecutive duplicate entries while keeping explanation steps
    # This prevents the animation from showing the same state twice in a row
    deduplicated_steps = [result[0]]

    for item in result[1:]:
        step_latex, step_explanation = item

        # Keep this entry if the LaTeX changed OR if it carries an explanation
        if step_latex != deduplicated_steps[-1][0] or step_explanation is not None:
            deduplicated_steps.append(item)

    # Return the clean list of (latex, explanation) tuples to the caller
    return deduplicated_steps


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
def check_solution(final_value, final_latex, equation, steps):
    
    left_str, right_str = equation.split("=")
    left_sym  = sp.sympify(normalize_expression(left_str.strip()), evaluate=False)
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

    steps.append(
        Step(
            before = f"x = {final_latex}",
            after = equation_display,
            explanation = "Let's verify!"
        )
    )

    steps.append(
        Step(
            before = equation_display,
            after = equation_display,
            explanation = f"Now substitute x for {final_latex}"
        )
    )

    solution_latex = sp.latex(final_value)

    def _sub_x(expression, x_value):
        return re.sub(
            r"(?<![a-zA-Z])x(?![a-zA-Z])",
            lambda _match: "(" + x_value + ")",
            expression,
        )

    left_substituted  = _sub_x(left_display,  solution_latex)
    right_substituted = _sub_x(right_display, solution_latex)

    steps.append(
        Step(
            before = equation_display,
            after = f"{left_substituted} = {right_substituted}"
        )
    )

    def _subst_terms(sympy_expression, x_value):

        if isinstance(sympy_expression, sp.Add):
            new_terms = []

            for term in sympy_expression.args:
                substituted_term = term.xreplace({x: sp.UnevaluatedExpr(x_value)})
                new_terms.append(substituted_term)

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
            steps.append(
                Step(
                    before = equation_before,
                    after = equation_before,
                    explanation = current_explanation
                )
            )

        equation_after = f"{left_after} = {current_right}"

        if equation_before != equation_after:
            steps.append(
                Step(
                    before = equation_before,
                    after = equation_after
                )
            )

        current_left = left_after

    for i, right_after in enumerate(right_steps_v):
        current_explanation = right_explanations.get(i)
        equation_before = f"{current_left} = {current_right}"

        if current_explanation:
            steps.append(
                Step(
                    before = equation_before, 
                    after = equation_before, 
                    explanation = current_explanation
                )
            )

        equation_after = f"{current_left} = {right_after}"

        if equation_before != equation_after:
            steps.append(
                Step(
                    before = equation_before, 
                    after = equation_after
                )
            )

        current_right = right_after

    left_orig  = sp.sympify(normalize_expression(left_str.strip()), evaluate=False)
    right_orig = sp.sympify(normalize_expression(right_str.strip()), evaluate=False)
    is_true = sp.simplify(left_orig.subs(x, final_value) - right_orig.subs(x, final_value)) == 0

    final_explanation = "The solution is correct!" if is_true else "The solution does not satisfy the equation."

    steps.append(
        Step(
            before = f"{current_left} = {current_right}",
            after = f"{current_left} = {current_right}",
            explanation = final_explanation
        )
    )

