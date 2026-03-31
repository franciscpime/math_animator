import sympy as sp
from sympy import lcm, product
import re
import math
from functools import reduce
from models.step import Step
from parser.equation_parser import (
    parse_equation,
    normalize_expression,
    safe_sympify,
    fix_implicit_mul,
    detect_decimals,
    detect_raw_fractions,
)
from math_utils.mmc import compute_mmc, apply_mmc
from utils.term_extractor import extract_terms, detailed_multiplication
from utils.equation_builder import build_equation, render_terms


# The symbolic variable used throughout all equation solving.
x = sp.symbols("x")


# =============================================================================
# HELPER: coefficient extraction
# =============================================================================

def _coef_rational(term):
    """Return the rational coefficient of a term containing x."""
    return sp.Rational(term.coeff(x))


# =============================================================================
# HELPER: LaTeX rendering of individual terms
# =============================================================================

def _frac_x_latex(numerator, denominator):
    """
    Return LaTeX for (num/den)*x, keeping the denominator explicit.

    Examples:
      _frac_x_latex(1, 2)   >> '\\frac{x}{2}'
      _frac_x_latex(-6, 2)  >> '- \\frac{6 x}{2}'
      _frac_x_latex(10, 1)  >> '10 x'
    """
    if denominator == 1:
        if numerator == 1:
            return "x"
        if numerator == -1:
            return "- x"
        return f"{numerator} x"

    if numerator == 1:
        return fr"\frac{{x}}{{{denominator}}}"
    if numerator == -1:
        return fr"- \frac{{x}}{{{denominator}}}"
    if numerator < 0:
        return fr"- \frac{{{abs(numerator)} x}}{{{denominator}}}"

    return fr"\frac{{{numerator} x}}{{{denominator}}}"


def _frac_latex(numerator, denominator):
    """
    Return the LaTeX string for numerator/denominator, always showing the denominator
    explicitly.

    Examples:
      _frac_latex(18, 2)   >> '\\frac{18}{2}'
      _frac_latex(-9, 2)   >> '- \\frac{9}{2}'
      _frac_latex(8, 1)    >> '8'
    """
    if denominator == 1:
        return str(numerator)
    if numerator < 0:
        return fr"- \frac{{{abs(numerator)}}}{{{denominator}}}"

    return fr"\frac{{{numerator}}}{{{denominator}}}"


def _join_latex(parts):
    """
    Join a list of LaTeX term strings with explicit +/- operators.
    Terms that start with '- ' are treated as negative and attached directly;
    all others are preceded by ' + '.

    Example:
      ['\\frac{10 x}{5}', '\\frac{4 x}{5}', '- \\frac{15 x}{5}']
      >> '\\frac{10 x}{5} + \\frac{4 x}{5} - \\frac{15 x}{5}'
    """
    result = parts[0]

    for part in parts[1:]:
        if part.startswith("- "):
            result += " " + part
        else:
            result += " + " + part
    return result


# =============================================================================
# CORE: stepwise term combination
# =============================================================================

def combine_terms_stepwise(terms):
    """
    Combine like terms step by step, emitting intermediate visual states so
    the animation can show every arithmetic operation explicitly.

    For variable terms (those containing x):
      Step A >> convert every coefficient to the common denominator
               e.g. 2x + 4/5x - 3x  >>  10x/5 + 4x/5 - 15x/5
      Step B >> group all numerators under the common denominator (unevaluated)
               e.g.  >>  (10 + 4 - 15)x / 5
      Step C >> emit the simplified SymPy result
               e.g.  >>  -x/5

    For constant terms (no x):
      Same three-step pattern applied to the numeric values.

    Return value is a list of entries, each being either:
      - a plain Python list of SymPy terms  (updated state, ready to render)
      - a tuple ("__latex__", latex_str, sympy_terms_list)
        which signals a visual-only intermediate step where the LaTeX is
        provided directly and sympy_terms_list holds the current SymPy state
        so the caller can build the right-hand side of the equation correctly.
    """
    from math import lcm as _mlcm
    from functools import reduce as _freduce

    new_terms = terms.copy()
    steps = []

    # Separate variable terms from constant terms.
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
        coefs = []
        for term in x_terms:
            coefs.append(sp.Rational(term.coeff(x)))

        denominators = []
        for coef in coefs:
            denominators.append(coef.q)

        common_denominator = _freduce(_mlcm, denominators)

        all_same = True
        for denominator in denominators:
            
            if denominator != common_denominator:
                all_same = False
                break

        if not all_same:
            # Step A: rewrite every term with the common denominator.
            adjusted_numerators = []

            for coefficient in coefs:
                adjusted_numerator = coefficient.p * (common_denominator // coefficient.q)
                adjusted_numerators.append(adjusted_numerator)

            parts_a = []

            for adjusted_numerator in adjusted_numerators:
                parts_a.append(_frac_x_latex(adjusted_numerator, common_denominator))

            steps.append(("__latex__", _join_latex(parts_a), new_terms.copy()))
        else:
            adjusted_numerators = []
            for coefficient in coefs:
                adjusted_numerators.append(coefficient.p)

        # Step B: group all numerators under the common denominator
        numerator_sum = 0

        for adjusted_numerator in adjusted_numerators:
            numerator_sum += adjusted_numerator

        numerator_str_list = []

        for adjusted_numerator in adjusted_numerators:
            numerator_str_list.append(str(adjusted_numerator))

        numerators_joined = " + ".join(numerator_str_list)
        numerators_joined = re.sub(r'\+\s*-', '- ', numerators_joined)

        if common_denominator > 1:
            grouped_numerator = fr"\frac{{({numerators_joined}) x}}{{{common_denominator}}}"
        else:
            grouped_numerator = f"({numerators_joined}) x"

        if not all_same:
            steps.append(("__latex__", grouped_numerator, new_terms.copy()))

        # Step C: emit the final combined term
        combined = sp.Rational(numerator_sum, common_denominator) * x

        new_terms = [combined]

        for const_term in const_terms:
            new_terms.append(const_term)

        steps.append(new_terms.copy())

    # ------------------------------------------------------------------
    # Process constant terms
    # ------------------------------------------------------------------
    elif len(const_terms) > 1:
        constant_coefs = []

        for const_term in const_terms:
            constant_coefs.append(sp.Rational(const_term))

        constant_denominators = []

        for constant_coef in constant_coefs:
            constant_denominators.append(constant_coef.q)

        common_denominator_const = _freduce(_mlcm, constant_denominators)

        all_same_denominator = True

        for current_denominator in constant_denominators:

            if current_denominator != common_denominator_const:
                all_same_denominator = False
                break

        if not all_same_denominator:
            # Step A: rewrite every constant with the common denominator.
            adjusted_const_numerators = []

            for constant_coef in constant_coefs:
                adjusted_numerator = constant_coef.p * (common_denominator_const // constant_coef.q)
                adjusted_const_numerators.append(adjusted_numerator)

            parts_a = []

            for adjusted_numerator in adjusted_const_numerators:
                parts_a.append(_frac_latex(adjusted_numerator, common_denominator_const))

            steps.append(("__latex__", _join_latex(parts_a), new_terms.copy()))
        else:
            adjusted_const_numerators = []

            for constant_coef in constant_coefs:
                adjusted_const_numerators.append(constant_coef.p)

        # Step B: group numerators
        sum_of_const_numerators = 0

        for numerator in adjusted_const_numerators:
            sum_of_const_numerators += numerator

        const_numerator_strings = []

        for numerator in adjusted_const_numerators:
            const_numerator_strings.append(str(numerator))

        const_numerator_expression = " + ".join(const_numerator_strings)
        const_numerator_expression = re.sub(r'\+\s*-', '- ', const_numerator_expression)

        if common_denominator_const > 1:
            grouped_const_numerator = fr"\frac{{{const_numerator_expression}}}{{{common_denominator_const}}}"
        else:
            grouped_const_numerator = f"({const_numerator_expression})"

        if not all_same_denominator:
            steps.append(("__latex__", grouped_const_numerator, new_terms.copy()))

        # Step C: final combined constant
        combined_constant = sp.Rational(sum_of_const_numerators, common_denominator_const)

        new_term_list = []
        if x_terms:

            for term in x_terms:
                new_term_list.append(term)

        new_term_list.append(combined_constant)

        steps.append(new_term_list.copy())

    return steps


# =============================================================================
# HELPERS: miscellaneous utilities kept for compatibility
# =============================================================================

def substitution_steps(expr, value):
    """Evaluate expr at x=value, returning intermediate SymPy expressions."""
    steps = []

    expr_sub = expr.subs(x, sp.Integer(value), evaluate=False)

    steps.append(expr_sub)
    expanded = sp.expand(expr_sub)

    if expanded != expr_sub:
        steps.append(expanded)

    final = sp.simplify(expanded)

    if final != expanded:
        steps.append(final)

    return steps


def common_divisor_of_constants(terms):
    """
    Return the largest rational divisor that evenly divides all constant terms.
    The result is a Rational so it can represent both integer and fractional
    common divisors.
    """
    if not terms:
        return sp.Integer(1)

    rationals = []

    for term in terms:
        rationals.append(sp.Rational(term))

    denominators = []

    for rational in rationals:
        denominators.append(rational.q)

    if denominators:
        mmc = lcm(*denominators)
    else:
        mmc = 1

    numerators_multiplied_lmc = []

    for rational in rationals:
        numerators_multiplied_lmc.append(int(rational * mmc))

    if numerators_multiplied_lmc:
        nums_int = []

        for numerator_multiplied_lmc in numerators_multiplied_lmc:
            nums_int.append(sp.Integer(numerator_multiplied_lmc))

        gcd_nums = abs(reduce(sp.gcd, nums_int))
    else:
        gcd_nums = 1

    return sp.Rational(gcd_nums, mmc)


def _decimal_str(solution):
    """
    Convert a rational solution to a decimal string rounded to 3 significant
    figures. Returns None when the solution is already an integer (no decimal
    approximation needed).
    """
    if isinstance(solution, sp.Integer) or (
        isinstance(solution, sp.Rational) and solution.q == 1
    ):
        return None

    decimal_value = float(solution)
    rounded = round(decimal_value, 3)

    if rounded == int(rounded):
        return str(int(rounded))

    return f"{rounded:.3f}".rstrip("0").rstrip(".")


# =============================================================================
# HELPERS: step generators for the initial fraction/decimal simplification
# =============================================================================

def _fraction_simplification_steps(num_str, den_str):
    """
    Return a list of LaTeX strings showing the reduction of a/b to its lowest
    terms. The list always starts with the unreduced form; if it is already in
    lowest terms the list contains only one element.

    Example:
      _fraction_simplification_steps('6', '4')
      >> ['\\frac{6}{4}', '\\frac{3}{2}']
    """
    numerator = int(num_str)
    denominator = int(den_str)

    steps = [r"\frac{" + num_str + r"}{" + den_str + r"}"]
    common_divisor = math.gcd(abs(numerator), denominator)

    if common_divisor > 1:
        steps.append(
            r"\frac{"
            + str(numerator // common_divisor)
            + r"}{"
            + str(denominator // common_divisor)
            + r"}"
        )

    return steps


def _decimal_simplification_steps(decimal_str):
    """
    Return a list of LaTeX strings walking through the conversion of a decimal
    to a fraction in lowest terms.

    Example:
      _decimal_simplification_steps('0.5')
      >> ['0.5', '\\frac{5}{10}', '\\frac{1}{2}']
    """
    decimal_normalized = decimal_str.replace(",", ".")

    if "." not in decimal_normalized:
        return [decimal_normalized]

    is_negative = decimal_normalized.startswith("-")

    decimal_abs = decimal_normalized.lstrip("-")
    decimal_places = len(decimal_abs.split(".")[1])

    denominator = 10 ** decimal_places
    numerator = int(decimal_abs.replace(".", ""))

    if is_negative:
        numerator = -numerator

    frac_unreduced = r"\frac{" + str(numerator) + r"}{" + str(denominator) + r"}"

    steps = [decimal_str, frac_unreduced]

    common_divisor = math.gcd(abs(numerator), denominator)

    if common_divisor > 1:
        steps.append(
            r"\frac{"
            + str(numerator // common_divisor)
            + r"}{"
            + str(denominator // common_divisor)
            + r"}"
        )
    return steps


# =============================================================================
# CORE: solve steps when the x-coefficient is a proper fraction
# =============================================================================

def _rational_coef_solve_steps(
        coef_rational,
        const_rational,
        final_left_latex,
        final_right_latex
    ):
    """
    Generate the animation steps for solving (p/q)*x = c when q > 1.

    The strategy is to first clear the denominator by multiplying both sides
    by q, then isolate x by dividing by p, showing every arithmetic operation
    as a separate step.

    Example for -x/5 = 25/2:
      [explanation]  Multiply both sides by 5
      >> -x = 25 * 5 / 2          (denominator cleared; product shown unevaluated)
      >> -x = 125/2               (numerator evaluated)
      [explanation]  Divide both sides by -1
      >> x = 125 / -2             (division shown as fraction)
      >> x = -125/2               (sign simplified by SymPy)

    Parameters
    ----------
    coef_rational     : sp.Rational  -- coefficient of x, e.g. Rational(-1, 5)
    const_rational    : sp.Rational  -- right-hand side,   e.g. Rational(25, 2)
    final_left_latex  : str          -- LaTeX of the left side before solving
    final_right_latex : str          -- LaTeX of the right side before solving

    Returns
    -------
    (list[Step], sp.Rational)  -- animation steps and the numeric solution
    """
    result_steps = []

    coef_numerator = coef_rational.p
    coef_denominator = coef_rational.q
    right_side_constant = const_rational

    eq_before_multiply = f"{final_left_latex} = {final_right_latex}"

    # Step 0: announce the multiplication.
    result_steps.append(
        Step(
            before=eq_before_multiply,
            after=eq_before_multiply,
            explanation=f"Multiply both sides by {coef_denominator}",
        )
    )

    # Step 1: multiply both sides by q to clear the denominator.
    # Left side becomes p*x (integer coefficient).
    # Right side: if c = cn/cd then c*q = (cn*q)/cd.
    left_without_den = sp.latex(sp.Integer(coef_numerator) * x)

    costante_numerator = right_side_constant.p
    costante_denominator = right_side_constant.q

    if costante_numerator < 0:
        cn_str = f"({costante_numerator})"
    else:
        cn_str = str(costante_numerator)

    if costante_denominator == 1:
        right_side_product_unevaluated = f"{cn_str} \\cdot {coef_denominator}"
    else:
        right_side_product_unevaluated = f"\\frac{{{cn_str} \\cdot {coef_denominator}}}{{{costante_denominator}}}"

    eq_after_multiply_unevaluated = f"{left_without_den} = {right_side_product_unevaluated}"

    result_steps.append(
        Step(
            before=eq_before_multiply,
            after=eq_after_multiply_unevaluated
        )
    )

    # Evaluate the product cn*coef_denominator to obtain a simpler right-hand side.
    numerator_product = costante_numerator * coef_denominator

    if costante_denominator == 1:
        right_side_product_evaluated = str(numerator_product)
    else:
        right_side_product_evaluated = sp.latex(
            sp.Rational(numerator_product, costante_denominator)
        )

    eq_after_multiply_evaluated = f"{left_without_den} = {right_side_product_evaluated}"

    if eq_after_multiply_evaluated != eq_after_multiply_unevaluated:
        result_steps.append(
            Step(
                before=eq_after_multiply_unevaluated,
                after=eq_after_multiply_evaluated
            )
        )
    else:
        eq_after_multiply_evaluated = eq_after_multiply_unevaluated

    # Step 2: announce the division, then show the result as a fraction.
    result_steps.append(
        Step(
            before=eq_after_multiply_evaluated,
            after=eq_after_multiply_evaluated,
            explanation=f"Divide both sides by {coef_numerator}",
        )
    )

    solution = sp.Rational(numerator_product, costante_denominator * coef_numerator)

    if costante_denominator == 1:
        right_side_divided = f"\\frac{{{numerator_product}}}{{{coef_numerator}}}"
    else:
        right_side_divided = f"\\frac{{{numerator_product}}}{{{costante_denominator * coef_numerator}}}"

    eq_after_divide_unevaluated = f"x = {right_side_divided}"

    result_steps.append(
        Step(
            before=eq_after_multiply_evaluated,
            after=eq_after_divide_unevaluated
        )
    )

    # Step 3: let SymPy simplify the fraction (handles sign cancellation etc.).
    eq_solution_simplified = f"x = {sp.latex(solution)}"

    if eq_solution_simplified != eq_after_divide_unevaluated:
        result_steps.append(
            Step(
                before=eq_after_divide_unevaluated,
                after=eq_solution_simplified
            )
        )

    return result_steps, solution


# =============================================================================
# MAIN: solve_linear
# =============================================================================

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

    _raw_decimals = detect_decimals(equation)
    _raw_fractions = detect_raw_fractions(equation)

    # ------------------------------------------------------------------
    # Build the initial display string, converting a/b notation to
    # \frac{a}{b} for proper LaTeX rendering.
    # ------------------------------------------------------------------
    def _eq_to_latex_display(eq):
        return re.sub(
            r"(?<!\\\\)(-?\d+)/(\d+)",
            lambda m: r"\frac{" + m.group(1) + r"}{" + m.group(2) + r"}",
            eq,
        )

    equation_display = _eq_to_latex_display(equation)
    current_eq_display = equation_display

    # ------------------------------------------------------------------
    # Pre-solve step 1: convert any decimal coefficients to fractions.
    # Each decimal is shown as an unreduced fraction first, then reduced.
    # ------------------------------------------------------------------
    for dec_str, _dec_val in _raw_decimals:

        dec_steps = _decimal_simplification_steps(dec_str)

        for i in range(1, len(dec_steps)):

            before_eq = current_eq_display
            after_eq  = before_eq.replace(dec_steps[i - 1], dec_steps[i], 1)

            if before_eq != after_eq:
                steps.append(
                    Step(
                        before=before_eq,
                        after=after_eq,
                        explanation="Convert decimal to fraction" if i == 1 else "Simplify fraction",
                    )
                )
                current_eq_display = after_eq

    # ------------------------------------------------------------------
    # Pre-solve step 2: simplify any unreduced fractions in the equation
    # (e.g. 6/4 >> 3/2) before we begin rearranging terms.
    # ------------------------------------------------------------------
    for num_s, den_s, _frac_val in _raw_fractions:
        frac_steps = _fraction_simplification_steps(num_s, den_s)
        frac_orig_str = num_s + "/" + den_s

        # Ensure a/b text is first shown as \frac{a}{b}.
        if frac_orig_str in current_eq_display:

            eq_with_frac = current_eq_display.replace(frac_orig_str, frac_steps[0], 1)

            if eq_with_frac != current_eq_display:
                steps.append(
                    Step(
                        before=current_eq_display,
                        after=eq_with_frac
                    )
                )
                current_eq_display = eq_with_frac

        # Then reduce if possible (e.g. \frac{6}{4} >> \frac{3}{2}).
        for i in range(1, len(frac_steps)):

            before_eq = current_eq_display
            after_eq  = before_eq.replace(frac_steps[i - 1], frac_steps[i], 1)

            if before_eq != after_eq:
                steps.append(
                    Step(
                        before=before_eq,
                        after=after_eq
                    )
                )
                current_eq_display = after_eq

    # ------------------------------------------------------------------
    # Rearrange: move all x-terms to the left, constants to the right.
    # ------------------------------------------------------------------
    left_terms  = extract_terms(left)
    right_terms = extract_terms(right)

    left_x,  left_const  = [], []
    right_x, right_const = [], []

    for term in left_terms:
        if term.has(x):
            left_x.append(term)
        else:
            left_const.append(term)

    for term in right_terms:
        if term.has(x):
            right_x.append(term)
        else:
            right_const.append(term)

    # Variable terms stay on the left; right-side variables are negated.
    # Constants move to the right; left-side constants are negated.
    variable_terms = []

    for term in left_x:
        variable_terms.append(term)

    for term in right_x:
        variable_terms.append(-term)

    constant_terms = []

    for term in right_const:
        constant_terms.append(term)

    for term in left_const:
        constant_terms.append(-term)

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
            after=new_eq
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
            # Intermediate visual step: the left-side LaTeX is supplied directly.
            # The right side is taken from the current equation state unchanged.
            # current_vars is NOT updated here; the following plain-list entry does that.
            _, var_latex, _state = entry
            before_eq  = build_equation(current_vars, constant_terms)
            const_side = before_eq.split("=")[1].strip()
            after_eq   = f"{var_latex} = {const_side}"

            if before_eq != after_eq:
                steps.append(
                    Step(
                        before=before_eq,
                        after=after_eq
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
            # Intermediate visual step: the right-side LaTeX is supplied directly.
            # current_consts is NOT updated here; the following plain-list entry does that.
            _, const_latex, _state = entry
            before_eq = build_equation(current_vars, current_consts)
            var_side  = before_eq.split("=")[0].strip()
            after_eq  = f"{var_side} = {const_latex}"

            if before_eq != after_eq:
                steps.append(
                    Step(
                        before=before_eq, 
                        after=after_eq
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
    final_left = current_vars[0]
    final_right = current_consts[0]
    coef = final_left.coeff(x)
    const = final_right

    def safe_gcd(a, b):
        return sp.gcd(sp.Rational(a), sp.Rational(b))

    # ==================================================================
    # INNER FUNCTION: _sympy_stepwise
    # ==================================================================
    def _sympy_stepwise(substituted_expression, sym_evaled, final_value):
        """
        Generate a list of (latex_str, explanation_or_None) tuples that walk
        through the arithmetic after substituting x = final_value.

        For a fractional final_value = p/q the sequence per term is:

          Step 1a  integer_coef * (p/q):
            >> \\frac{k \\cdot (p)}{q}          show the product explicitly
            >> result                            integer: jump directly;
                                                 fraction: show numerator then simplify

          Step 1b  frac_coef * (p/q):
            >> \\frac{a \\cdot (p)}{b \\cdot q}  show the product
            >> result                            same logic as 1a

          Step 1c  simplify any remaining unsimplified constant fractions
            e.g. \\frac{6}{4}  >>  \\frac{3}{2}
            (these come from the original equation text which is preserved verbatim)

          Step 2   convert isolated integers to the common denominator
            e.g. 3  >>  \\frac{6}{2}  when the expression also contains /2 terms

          Step 3   add/subtract fractions pairwise:
            >> \\frac{a op b}{d}   grouped numerator (unevaluated)
            >> \\frac{result}{d}   numerator evaluated
            >> simplified form     if the fraction is reducible

        For an integer final_value the sequence is:
          Step 1   evaluate each product k*(val)
          Step 2   sum numeric pairs left to right
        """
        import re as _re
        from math import lcm as _mlcm
        from functools import reduce as _freduce

        def _fix_pm(expression):
            """Normalise double negatives and plus-minus combinations."""
            expression = _re.sub(r"-\s*-\s*", "+ ", expression)
            expression = _re.sub(r"\+\s*-\s*", "- ", expression)
            return expression.strip()

        result = [(substituted_expression, None)]
        current_expression = substituted_expression

        # --------------------------------------------------------------
        # Integer solution path
        # --------------------------------------------------------------
        if not (isinstance(final_value, sp.Rational) and final_value.q != 1):
            solution_as_int = int(final_value)
            solution_latex = sp.latex(final_value)
            solution_latex_escaped = _re.escape(solution_latex)

            # Evaluate each product k*(integer) in the expression.
            pattern = r"(-?\s*\d+)\s*\(\s*" + solution_latex_escaped + r"\s*\)"

            for _ in range(20):
                pattern_match = _re.search(pattern, current_expression)

                if not pattern_match:
                    break

                term_coefficient = int(pattern_match.group(1).replace(" ", ""))
                product = term_coefficient * solution_as_int

                before_match = current_expression[:pattern_match.start()]
                after_match = current_expression[pattern_match.end():]

                product_string = str(product)

                # FIX Bug 2: build updated_expression in a single operation,
                # never overwriting before_match (which is needed unchanged).
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

            # Sum pairs of integers until a single number remains.
            final_expression = sp.latex(sp.simplify(sym_evaled))

            for _ in range(20):
                pattern_match = _re.search(r"(-?\d+)\s*([+-])\s*(\d+)", current_expression)

                if not pattern_match:
                    break

                first_number = int(pattern_match.group(1))
                arithmetic_operator = pattern_match.group(2)
                second_number = int(pattern_match.group(3))

                if arithmetic_operator == "+":
                    soma = first_number + second_number
                else:
                    soma = first_number - second_number
                
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

        # --------------------------------------------------------------
        # Fractional solution path
        # --------------------------------------------------------------
        solution_numerator = final_value.p
        solution_denominator = final_value.q

        solution_latex = sp.latex(final_value)
        solution_latex_escaped = _re.escape(solution_latex)

        # Wrap a negative numerator in parentheses for visual clarity.
        if solution_numerator < 0:
            solution_numerator_str = f"({solution_numerator})"
        else:
            solution_numerator_str = str(solution_numerator)

        # Step 1a: integer_coef * (p/q).
        int_times_frac_pattern = r"(-?\s*\d+)\s*\(\s*" + solution_latex_escaped + r"\s*\)"

        for _ in range(20):
            pattern_match = _re.search(int_times_frac_pattern, current_expression)

            if not pattern_match:
                break

            term_coefficient = int(pattern_match.group(1).replace(" ", ""))

            numerator_product = term_coefficient * solution_numerator
            before_match = current_expression[:pattern_match.start()]
            after_match = current_expression[pattern_match.end():]

            # Always show the multiplication symbol explicitly.
            frac_multiplication = (
                r"\frac{"
                + str(term_coefficient)
                + r" \cdot "
                + solution_numerator_str
                + r"}{"
                + str(solution_denominator)
                + r"}"
            )

            expr_with_multiplication = _fix_pm(before_match + frac_multiplication + after_match)

            if expr_with_multiplication != current_expression:
                result.append((expr_with_multiplication, None))
                current_expression = expr_with_multiplication

            numerator_evaluated = r"\frac{" + str(numerator_product) + r"}{" + str(solution_denominator) + r"}"

            fraction_reduced = sp.latex(
                sp.Rational(numerator_product, solution_denominator)
            )

            simplification_is_integer = "\\frac" not in fraction_reduced

            if fraction_reduced == numerator_evaluated:
                # Already in lowest terms; show the evaluated form and stop.
                expr_with_evaluated_numerator = _fix_pm(
                    current_expression.replace(frac_multiplication, numerator_evaluated, 1)
                )

                if expr_with_evaluated_numerator != current_expression:
                    result.append((expr_with_evaluated_numerator, None))
                    current_expression = expr_with_evaluated_numerator

            elif simplification_is_integer:
                # Numerator and denominator cancel completely; jump to integer.
                final_reduced_expr = _fix_pm(
                    current_expression.replace(frac_multiplication, fraction_reduced, 1)
                )

                if final_reduced_expr != current_expression:
                    result.append((final_reduced_expr, None))
                    current_expression = final_reduced_expr

            else:
                # Show the evaluated numerator first, then reduce the fraction.
                expr_with_evaluated_numerator = _fix_pm(
                    current_expression.replace(frac_multiplication, numerator_evaluated, 1)
                )

                if expr_with_evaluated_numerator != current_expression:
                    result.append((expr_with_evaluated_numerator, None))
                    current_expression = expr_with_evaluated_numerator

                final_reduced_expr = _fix_pm(
                    current_expression.replace(numerator_evaluated, fraction_reduced, 1)
                )

                if final_reduced_expr != current_expression:
                    result.append((final_reduced_expr, None))
                    current_expression = final_reduced_expr

        # Step 1b: frac_coef * (p/q).
        fraction_times_fraction_pattern = (
            r"\\frac\{(\d+)\}\{(\d+)\}\s*\(\s*"
            + solution_latex_escaped
            + r"\s*\)"
        )

        for _ in range(20):
            pattern_match = _re.search(fraction_times_fraction_pattern, current_expression)
            
            if not pattern_match:
                break

            frac_coef_numerator = int(pattern_match.group(1))
            frac_coef_denominator = int(pattern_match.group(2))

            frac_product_numerator = frac_coef_numerator * solution_numerator
            frac_product_denominator = frac_coef_denominator * solution_denominator

            before_match = current_expression[:pattern_match.start()]
            after_match = current_expression[pattern_match.end():]

            frac_product_shown = (
                r"\frac{"
                + str(frac_coef_numerator)
                + r" \cdot "
                + solution_numerator_str
                + r"}{"
                + str(frac_coef_denominator)
                + r" \cdot "
                + str(solution_denominator)
                + r"}"
            )

            expr_with_multiplication = _fix_pm(before_match + frac_product_shown + after_match)
            
            if expr_with_multiplication != current_expression:
                result.append((expr_with_multiplication, None))
                current_expression = expr_with_multiplication

            frac_numerator_evaluated = r"\frac{" + str(frac_product_numerator) + r"}{" + str(frac_product_denominator) + r"}"
            
            frac_fraction_reduced = sp.latex(
                sp.Rational(frac_product_numerator, frac_product_denominator)
            )

            frac_simplification_is_integer = "\\frac" not in frac_fraction_reduced

            if frac_fraction_reduced == frac_numerator_evaluated:
                expr_with_evaluated_numerator = _fix_pm(
                    current_expression.replace(frac_product_shown, frac_numerator_evaluated, 1)
                )
                
                if expr_with_evaluated_numerator != current_expression:
                    result.append((expr_with_evaluated_numerator, None))
                    current_expression = expr_with_evaluated_numerator
            
            elif frac_simplification_is_integer:
                final_reduced_expr = _fix_pm(
                    current_expression.replace(frac_product_shown, frac_fraction_reduced, 1)
                )
                
                if final_reduced_expr != current_expression:
                    result.append((final_reduced_expr, None))
                    current_expression = final_reduced_expr
            else:
                expr_with_evaluated_numerator = _fix_pm(
                    current_expression.replace(frac_product_shown, frac_numerator_evaluated, 1)
                )
                
                if expr_with_evaluated_numerator != current_expression:
                    result.append((expr_with_evaluated_numerator, None))
                    current_expression = expr_with_evaluated_numerator

                final_reduced_expr = _fix_pm(
                    current_expression.replace(frac_numerator_evaluated, frac_fraction_reduced, 1)
                )
                
                if final_reduced_expr != current_expression:
                    result.append((final_reduced_expr, None))
                    current_expression = final_reduced_expr

        # Step 1c: simplify any unreduced positive constant fractions still
        # present in the string, e.g. \frac{6}{4} >> \frac{3}{2}.
        # These originate from the original equation text which _check_solution
        # preserves verbatim so that the display matches the input exactly.
        constant_fraction_pattern = r"\\frac\{(\d+)\}\{(\d+)\}"
        for _ in range(20):
            has_reducible_fraction = False
            # Find all constant fractions in the current expression.
            for pattern_match in _re.finditer(constant_fraction_pattern, current_expression):

                const_frac_numerator = int(pattern_match.group(1))
                const_frac_denominator = int(pattern_match.group(2))
                common_divisor = math.gcd(const_frac_numerator, const_frac_denominator)

                if common_divisor > 1:
                    const_frac_unreduced = r"\frac{" + str(const_frac_numerator) + r"}{" + str(const_frac_denominator) + r"}"

                    if const_frac_denominator // common_divisor > 1:
                        const_frac_reduced = (
                            r"\frac{"
                            + str(const_frac_numerator // common_divisor)
                            + r"}{"
                            + str(const_frac_denominator // common_divisor)
                            + r"}"
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

        # Step 2: convert any remaining isolated integers to the common
        # denominator of all fractions currently present in the expression.
        def _find_isolated_ints(expression):
            """
            Find integers outside LaTeX braces that are not immediately
            followed by / or { (which would make them part of a fraction).
            """
            # FIX Bug 1: use 'i' consistently as the loop index throughout,
            # matching the initialisation 'i = 0' at the top of the function.
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
                    except:
                        i += 1
                        continue

                    if integer_value != 0:
                        isolated_integers.append((start, number_end, integer_value))
                    i = number_end
                else:
                    i += 1
            return isolated_integers


        def _dens_in(expression):
            # Extract all denominators from \\frac{...}{d} patterns.
            return [int(denominator) for denominator in _re.findall(r"\\frac\{[^}]+\}\{(\d+)\}", expression)]

        found_denominators = _dens_in(current_expression)

        if found_denominators and _find_isolated_ints(current_expression):
            from functools import reduce as _fr2

            common_denominator = _fr2(_mlcm, found_denominators)
            result.append((current_expression, f"Reduce to common denominator ({common_denominator})"))

            for _ in range(30):
                found_integers = _find_isolated_ints(current_expression)

                if not found_integers:
                    break

                start, end, integer_value = found_integers[0]

                converted_fraction = (
                    r"\frac{"
                    + str(integer_value * common_denominator)
                    + r"}{"
                    + str(common_denominator)
                    + r"}"
                )
                
                new_expression = _fix_pm(current_expression[:start] + converted_fraction + current_expression[end:])
                
                if new_expression == current_expression:
                    break

                result.append((new_expression, None))

                current_expression = new_expression

        # Step 3: add or subtract fractions pairwise until only one remains.
        fraction_pair_pattern = r"(\\frac\{(-?\d+)\}\{(\d+)\})\s*([+-])\s*(\\frac\{(-?\d+)\}\{(\d+)\})"

        for _ in range(30):
            pattern_match = _re.search(fraction_pair_pattern, current_expression)

            if not pattern_match:
                break

            first_numerator = int(pattern_match.group(2))
            first_denominator = int(pattern_match.group(3))

            operator = pattern_match.group(4)

            second_numerator = int(pattern_match.group(6))
            second_denominator = int(pattern_match.group(7))

            # Absorb a leading external minus sign into the first numerator.
            before_match = current_expression[:pattern_match.start()].rstrip()

            if before_match.endswith("-"):
                first_numerator = -abs(first_numerator)

                current_expression = (
                    current_expression[:len(before_match) - 1].rstrip()
                    + " "
                    + current_expression[len(before_match):]
                )
                
                pattern_match = _re.search(fraction_pair_pattern, current_expression)
                
                if not pattern_match:
                    break

                first_numerator = int(pattern_match.group(2))
                first_denominator =int(pattern_match.group(3))

                first_numerator = -abs(first_numerator)
                operator = pattern_match.group(4)

                second_numerator = int(pattern_match.group(6))
                second_denominator = int(pattern_match.group(7))

            # If denominators differ, align them first.
            if first_denominator != second_denominator:
                from math import lcm as _lcm2

                common_denominator = _lcm2(first_denominator, second_denominator)
                first_numerator_converted = first_numerator * (common_denominator // first_denominator)
                second_numerator_converted = second_numerator * (common_denominator // second_denominator)
                
                first_fraction_common_den = r"\frac{" + str(first_numerator_converted) + r"}{" + str(common_denominator) + r"}"
                second_fraction_common_den = r"\frac{" + str(second_numerator_converted) + r"}{" + str(common_denominator) + r"}"
                
                new_expression = _fix_pm(
                    current_expression[:pattern_match.start()]
                    + first_fraction_common_den
                    + " " + operator + " "
                    + second_fraction_common_den
                    + current_expression[pattern_match.end():]
                )

                if new_expression == current_expression:
                    break

                result.append((new_expression, None))
                current_expression = new_expression
                continue

            # Same denominator: show the unevaluated combined numerator first.
            if operator == "+":
                second_numerator_with_sign = second_numerator
                unevaluated_numerator_expression = f"{first_numerator} + {second_numerator}"
            else:
                second_numerator_with_sign = -second_numerator
                unevaluated_numerator_expression = f"{first_numerator} - {second_numerator}"
            
            grouped_fraction = r"\frac{" + unevaluated_numerator_expression + r"}{" + str(first_denominator) + r"}"
            
            expr_with_joined_fraction = _fix_pm(current_expression[:pattern_match.start()] + grouped_fraction + current_expression[pattern_match.end():])

            if expr_with_joined_fraction != current_expression:
                result.append((expr_with_joined_fraction, None))
                current_expression = expr_with_joined_fraction

            # Then evaluate the numerator as an explicit intermediate step.
            numerator_sum = first_numerator + second_numerator_with_sign
            frac_with_numerator_evaluated = r"\frac{" + str(numerator_sum) + r"}{" + str(first_denominator) + r"}"
            reduced_fraction = sp.latex(sp.Rational(numerator_sum, first_denominator))

            if frac_with_numerator_evaluated != grouped_fraction:
                new_expression_eval = _fix_pm(current_expression.replace(grouped_fraction, frac_with_numerator_evaluated, 1))

                if new_expression_eval != current_expression:
                    result.append((new_expression_eval, None))
                    current_expression = new_expression_eval

            # Finally simplify the fraction if it is reducible.
            if reduced_fraction != frac_with_numerator_evaluated:
                new_expression = _fix_pm(current_expression.replace(frac_with_numerator_evaluated, reduced_fraction, 1))

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

        # Remove consecutive duplicate entries while keeping explanations.
        deduplicated_steps = [result[0]]
        for item in result[1:]:
            step_latex, step_explanation = item
            if step_latex != deduplicated_steps[-1][0] or step_explanation is not None:
                deduplicated_steps.append(item)
        return deduplicated_steps

    # ==================================================================
    # INNER FUNCTION: _check_solution
    # ==================================================================
    def _check_solution(final_value, final_latex, equation, steps):
        """
        Verify the solution by substituting x = final_value back into the
        original equation and simplifying both sides step by step.

        Animation sequence:
          1. Show the original equation again with a "Let's verify!" message.
          2. Announce the substitution value.
          3. Replace every x with the numeric value.
          4. Simplify the left side arithmetic step by step.
          5. Simplify the right side arithmetic step by step.
          6. Show a confirmation message (correct / does not satisfy).
        """
        left_str,  right_str  = equation.split("=")
        left_sym   = sp.sympify(normalize_expression(left_str.strip()),  evaluate=False)
        right_sym  = sp.sympify(normalize_expression(right_str.strip()), evaluate=False)

        # Build the LaTeX display of the original equation.
        # Fractions are shown as \frac{a}{b} but NOT simplified, so that the
        # display matches the user's original input (e.g. 6/4 stays as \frac{6}{4}).
        def _fracs_to_latex(expression):
            return re.sub(
                r"(-?\d+)/(\d+)",
                lambda match: r"\frac{" + match.group(1) + r"}{" + match.group(2) + r"}",
                expression,
            )

        left_display  = _fracs_to_latex(left_str.strip())
        right_display = _fracs_to_latex(right_str.strip())

        # Add a space between \frac{a}{b} and a following variable letter.
        left_display  = re.sub(
            r"(\\frac\{\d+\}\{\d+\})(\s*)([a-zA-Z])", 
            r"\1 \3", 
            left_display
        )

        right_display = re.sub(
            r"(\\frac\{\d+\}\{\d+\})(\s*)([a-zA-Z])", 
            r"\1 \3", 
            right_display
        )

        equation_display = f"{left_display} = {right_display}"

        steps.append(
            Step(
                before=equation_display, 
                after=equation_display,
                explanation="Let's verify!",
            )
        )

        steps.append(
            Step(
                before=equation_display, 
                after=equation_display,
                explanation=f"Now substitute x for {final_latex}",
            )
        )

        # Replace every x occurrence in the LaTeX string with (value).
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
                before=equation_display,
                after=f"{left_substituted} = {right_substituted}",
            )
        )

        # Substitute into the SymPy expressions, keeping each Add term
        # separate so the stepwise evaluator can process them individually.
        def _subst_terms(sympy_expression, x_value):
            if isinstance(sympy_expression, sp.Add):
                new_terms = []
                for term in sympy_expression.args:
                    new_terms.append(term.xreplace({x: sp.UnevaluatedExpr(x_value)}))

                return sp.Add(*new_terms, evaluate=False)
            return sympy_expression.xreplace({x: sp.UnevaluatedExpr(x_value)})

        left_unevaluated  = _subst_terms(left_sym,  final_value)
        right_unevaluated = _subst_terms(right_sym, final_value)

        left_tuples  = _sympy_stepwise(left_substituted,  left_unevaluated,  final_value)
        right_tuples = _sympy_stepwise(right_substituted, right_unevaluated, final_value)

        def _extract(tuples):
            """Split (latex, explanation) tuples into a step list and an
            explanation map keyed by step index."""
            steps, explanations = [], {}
            for i, (step_str, step_explanation) in enumerate(tuples):
                steps.append(step_str)

                if step_explanation:
                    explanations[i] = step_explanation
            return steps, explanations

        left_steps_v,  left_explanations  = _extract(left_tuples)
        right_steps_v, right_explanations = _extract(right_tuples)

        # Animate the left side first, keeping the right side fixed.
        current_left, current_right = left_substituted, right_substituted

        for i, left_after in enumerate(left_steps_v):
            
            current_explanation   = left_explanations.get(i)
            equation_before = f"{current_left} = {current_right}"

            if current_explanation:
                steps.append(
                    Step(
                        before=equation_before, 
                        after=equation_before, 
                        explanation=current_explanation
                    )
                )

            equation_after = f"{left_after} = {current_right}"
            
            if equation_before != equation_after:
                steps.append(
                    Step(
                        before=equation_before, 
                        after=equation_after
                    )
                )

            current_left = left_after

        # Then animate the right side, keeping the left side fixed.
        for i, right_after in enumerate(right_steps_v):

            current_explanation   = right_explanations.get(i)
            equation_before = f"{current_left} = {current_right}"

            if current_explanation:
                steps.append(
                    Step(
                        before=equation_before, 
                        after=equation_before, 
                        explanation=current_explanation
                    )
                )

            equation_after = f"{current_left} = {right_after}"

            if equation_before != equation_after:
                steps.append(
                    Step(
                        before=equation_before, 
                        after=equation_after
                    )
                )

            current_right = right_after

        # Evaluate both sides numerically and confirm whether they are equal.
        left_orig  = sp.sympify(normalize_expression(left_str.strip()),  evaluate=False)
        right_orig = sp.sympify(normalize_expression(right_str.strip()), evaluate=False)

        is_true    = sp.simplify(
            left_orig.subs(x, final_value) - right_orig.subs(x, final_value)
        ) == 0

        if is_true:
            current_explanation = "The solution is correct!"
        else:
            current_explanation = "The solution does not satisfy the equation."

        steps.append(
            Step(
                before=f"{current_left} = {current_right}",
                after=f"{current_left} = {current_right}",
                explanation=current_explanation,
            )
        )

    # ------------------------------------------------------------------
    # Case 1: coefficient of x is already 1 -- no division needed.
    # ------------------------------------------------------------------
    if coef == 1:
        steps.append(
            Step(
                before=f"{sp.latex(final_left)} = {sp.latex(final_right)}",
                after=f"x = {sp.latex(const)}",
            )
        )

        final_value = const
        final_latex = sp.latex(final_value)
        decimal_approximation = _decimal_str(final_value)

        if decimal_approximation:
            solution_str = f"x = {final_latex}"

            steps.append(
                Step(
                    before=solution_str, after=solution_str,
                    explanation=f"x = {final_latex} \\approx {decimal_approximation}",
                )
            )

        _check_solution(final_value, final_latex, equation, steps)

    # ------------------------------------------------------------------
    # Case 2: coefficient of x requires a division step.
    # ------------------------------------------------------------------
    else:
        coefficient_rational  = sp.Rational(coef)
        constant_rational = sp.Rational(const)

        final_left_latex  = sp.latex(final_left)
        final_right_latex = sp.latex(final_right)

        # Sub-case 2a: fractional coefficient (e.g. -5/2 x = -12).
        # Delegate to the dedicated function that handles the multiply/divide sequence.
        if coefficient_rational.q > 1:
            extra_steps, solution = _rational_coef_solve_steps(
                coefficient_rational, constant_rational,
                final_left_latex, final_right_latex,
            )
            steps.extend(extra_steps)

        # Sub-case 2b: integer coefficient (e.g. 15x = 3).
        else:
            # Check whether a partial simplification step is possible before
            # the final division (e.g. 12x = 8  --div 4-->  3x = 2  --div 3-->  x = 2/3).
            common_divisor = safe_gcd(abs(constant_rational), abs(coefficient_rational))
            simplified_coef  = coefficient_rational  / common_divisor
            simplified_const = constant_rational / common_divisor

            has_intermediate_step = (
                common_divisor > 1
                and common_divisor != abs(coefficient_rational)
                and (isinstance(simplified_coef,  sp.Integer) or simplified_coef.q  == 1)
                and (isinstance(simplified_const, sp.Integer) or simplified_const.q == 1)
                and simplified_coef != 1
            )

            if has_intermediate_step:
                left_after_division  = simplified_coef * x
                right_after_division = simplified_const

                steps.append(
                    Step(
                        before=f"{final_left_latex} = {final_right_latex}",
                        after=f"{final_left_latex} = {final_right_latex}",
                        explanation=f"Divide both sides by {sp.latex(common_divisor)}",
                    )
                )

                steps.append(
                    Step(
                        before=f"{final_left_latex} = {final_right_latex}",
                        after=f"{sp.latex(left_after_division)} = {sp.latex(right_after_division)}",
                    )
                )

                steps.append(
                    Step(
                        before=f"{sp.latex(left_after_division)} = {sp.latex(right_after_division)}",
                        after=f"{sp.latex(left_after_division)} = {sp.latex(right_after_division)}",
                        explanation=f"Divide both sides by {sp.latex(simplified_coef)}",
                    )
                )

                steps.append(
                    Step(
                    before=f"{sp.latex(left_after_division)} = {sp.latex(right_after_division)}",
                    after=f"x = {sp.latex(sp.Rational(constant_rational, coefficient_rational))}",
                ))

            else:
                # Direct division: show x = const/coef unreduced first,
                # then simplify (e.g. 15x = 3  >>  x = 3/15  >>  x = 1/5).
                solution = sp.Rational(constant_rational, coefficient_rational)
                unreduced_fraction  = (
                    r"\frac{" + str(int(constant_rational)) + r"}{" + str(int(coefficient_rational)) + r"}"
                )

                reduced_fraction = sp.latex(solution)

                steps.append(
                    Step(
                        before=f"{final_left_latex} = {final_right_latex}",
                        after=f"{final_left_latex} = {final_right_latex}",
                        explanation=f"Divide both sides by {sp.latex(coefficient_rational)}",
                    )
                )

                steps.append(
                    Step(
                        before=f"{final_left_latex} = {final_right_latex}",
                        after=f"x = {unreduced_fraction}",
                    )
                )

                if reduced_fraction != unreduced_fraction:
                    steps.append(
                        Step(
                            before=f"x = {unreduced_fraction}",
                            after=f"x = {reduced_fraction}",
                        )
                    )

            solution = sp.Rational(constant_rational, coefficient_rational)

        final_value = solution
        final_latex = sp.latex(final_value)

        decimal_approximation = _decimal_str(final_value)
        
        if decimal_approximation:
            solution_str = f"x = {final_latex}"
            steps.append(
                Step(
                    before=solution_str, after=solution_str,
                    explanation=f"x = {final_latex} \\approx {decimal_approximation}",
                )
            )

        _check_solution(final_value, final_latex, equation, steps)

    return steps
