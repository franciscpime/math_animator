import sympy as sp
from sympy import lcm
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

        dens = []
        for coef in coefs:
            dens.append(coef.q)

        common_denominator = _freduce(_mlcm, dens)

        all_same = True
        for d in dens:
            if d != common_denominator:
                all_same = False
                break

        if not all_same:
            # Step A: rewrite every term with the common denominator.
            ns = []
            for c in coefs:
                n = c.p * (common_denominator // c.q)
                ns.append(n)

            parts_a = []
            for n in ns:
                parts_a.append(_frac_x_latex(n, common_denominator))

            steps.append(("__latex__", _join_latex(parts_a), new_terms.copy()))
        else:
            ns = []
            for c in coefs:
                ns.append(c.p)

        # Step B: group all numerators under the common denominator
        sum_num = 0
        for n in ns:
            sum_num += n

        num_parts = []
        for n in ns:
            num_parts.append(str(n))

        num_expr = " + ".join(num_parts)
        num_expr = re.sub(r'\+\s*-', '- ', num_expr)

        if common_denominator > 1:
            latex_b = fr"\frac{{({num_expr}) x}}{{{common_denominator}}}"
        else:
            latex_b = f"({num_expr}) x"

        if not all_same:
            steps.append(("__latex__", latex_b, new_terms.copy()))

        # Step C: emit the final combined term

        combined = sp.Rational(sum_num, common_denominator) * x

        new_terms = [combined]
        for t in const_terms:
            new_terms.append(t)

        steps.append(new_terms.copy())

    # ------------------------------------------------------------------
    # Process constant terms
    # ------------------------------------------------------------------
    elif len(const_terms) > 1:
        coefs_c = []
        for t in const_terms:
            coefs_c.append(sp.Rational(t))

        dens_c = []
        for c in coefs_c:
            dens_c.append(c.q)

        den_comum_c = _freduce(_mlcm, dens_c)

        all_same_c = True
        for d in dens_c:
            if d != den_comum_c:
                all_same_c = False
                break

        if not all_same_c:
            # Step A: rewrite every constant with the common denominator.
            ns_c = []
            for c in coefs_c:
                n = c.p * (den_comum_c // c.q)
                ns_c.append(n)

            parts_a = []
            for n in ns_c:
                parts_a.append(_frac_latex(n, den_comum_c))

            steps.append(("__latex__", _join_latex(parts_a), new_terms.copy()))
        else:
            ns_c = []
            for c in coefs_c:
                ns_c.append(c.p)

        # Step B: group numerators
        sum_num_c = 0
        for n in ns_c:
            sum_num_c += n

        num_parts_c = []
        for n in ns_c:
            num_parts_c.append(str(n))

        num_expr_c = " + ".join(num_parts_c)
        num_expr_c = re.sub(r'\+\s*-', '- ', num_expr_c)

        if den_comum_c > 1:
            latex_b_c = fr"\frac{{{num_expr_c}}}{{{den_comum_c}}}"
        else:
            latex_b_c = f"({num_expr_c})"

        if not all_same_c:
            steps.append(("__latex__", latex_b_c, new_terms.copy()))

        # Step C: final combined constant
        combined_c = sp.Rational(sum_num_c, den_comum_c)

        new_terms_c = []
        if x_terms:
            for t in x_terms:
                new_terms_c.append(t)

        new_terms_c.append(combined_c)

        steps.append(new_terms_c.copy())

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
    
    rats = []
    for t in terms:
        rats.append(sp.Rational(t))

    denoms = []
    for r in rats:
        denoms.append(r.q)

    if denoms:
        mmc = lcm(*denoms)
    else:
        mmc = 1

    nums = []
    for r in rats:
        nums.append(int(r * mmc))

    if nums:
        nums_int = []
        for n in nums:
            nums_int.append(sp.Integer(n))

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
    
    val = float(solution)
    rounded = round(val, 3)

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
    num, den = int(num_str), int(den_str)

    steps = [r"\frac{" + num_str + r"}{" + den_str + r"}"]
    g = math.gcd(abs(num), den)

    if g > 1:
        steps.append(r"\frac{" + str(num // g) + r"}{" + str(den // g) + r"}")

    return steps


def _decimal_simplification_steps(decimal_str):
    """
    Return a list of LaTeX strings walking through the conversion of a decimal
    to a fraction in lowest terms.

    Example:
      _decimal_simplification_steps('0.5')
      >> ['0.5', '\\frac{5}{10}', '\\frac{1}{2}']
    """
    s = decimal_str.replace(",", ".")

    if "." not in s:
        return [s]
    
    is_neg = s.startswith("-")
    s_abs  = s.lstrip("-")
    n_dec = len(s_abs.split(".")[1])
    den = 10 ** n_dec
    num = int(s_abs.replace(".", ""))

    if is_neg:
        num = -num

    frac_unreduced = r"\frac{" + str(num) + r"}{" + str(den) + r"}"

    steps = [decimal_str, frac_unreduced]

    g = math.gcd(abs(num), den)

    if g > 1:
        steps.append(r"\frac{" + str(num // g) + r"}{" + str(den // g) + r"}")
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

    coef_numerator = coef_rational.p   # numerator   of the coefficient (may be negative)
    coef_denominator = coef_rational.q   # denominator of the coefficient (always positive)
    right_side_constant = const_rational    # right-hand side as a Rational

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
        right_side_product_evaluated = sp.latex(sp.Rational(numerator_product, costante_denominator))

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

    for t in left_terms:
        if t.has(x):
            left_x.append(t)
        else:
            left_const.append(t)

    for t in right_terms:
        if t.has(x):
            right_x.append(t)
        else:
            right_const.append(t)

    # Variable terms stay on the left; right-side variables are negated.
    # Constants move to the right; left-side constants are negated.
    variable_terms = []

    for t in left_x:
        variable_terms.append(t)

    for t in right_x:
        variable_terms.append(-t)

    constant_terms = []

    for t in right_const:
        constant_terms.append(t)

    for t in left_const:
        constant_terms.append(-t)

    new_eq = build_equation(variable_terms, constant_terms)

    steps.append(
        Step(
            before=equation_display, after=equation_display,
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
                steps.append(Step(before=before_eq, after=after_eq))
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
                steps.append(Step(before=before_eq, after=after_eq))
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
    def _sympy_stepwise(subst_str, sym_evaled, final_value):
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

        def _fix_pm(s):
            """Normalise double negatives and plus-minus combinations."""
            s = _re.sub(r"-\s*-\s*", "+ ", s)
            s = _re.sub(r"\+\s*-\s*", "- ", s)
            return s.strip()

        result = [(subst_str, None)]
        cur = subst_str

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
                m = _re.search(pattern, cur)

                if not m: 
                    break

                term_coefficient = int(m.group(1).replace(" ", ""))
                prod = term_coefficient * solution_as_int
                before_m = cur[:m.start()]
                after_m = cur[m.end():]
                prod_str = str(prod)

                if before_m.rstrip() and before_m.rstrip()[-1] not in "+-=(,":
                    prod_str = ("+ " if prod >= 0 else "- ") + str(abs(prod))
                    before_m = _fix_pm(before_m + prod_str)
                s1 = _fix_pm(before_m + after_m)

                if s1 == cur: 
                    break

                result.append((s1, None)); cur = s1

            # Sum pairs of integers until a single number remains.
            final_str = sp.latex(sp.simplify(sym_evaled))

            for _ in range(20):
                m = _re.search(r"(-?\d+)\s*([+-])\s*(\d+)", cur)
                
                if not m: 
                    break

                a, op, b = int(m.group(1)), m.group(2), int(m.group(3))
                soma = a + b if op == "+" else a - b
                novo = _fix_pm(cur[:m.start()] + str(soma) + cur[m.end():])

                if novo == cur: 
                    break

                result.append((novo, None)); cur = novo

                if cur == final_str: 
                    break

            if cur != final_str:
                result.append((final_str, None))
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
        pat_coef_frac = r"(-?\s*\d+)\s*\(\s*" + solution_latex_escaped + r"\s*\)"

        for _ in range(20):
            m = _re.search(pat_coef_frac, cur)

            if not m: 
                break

            term_coefficient = int(m.group(1).replace(" ", ""))
            numerator_product = term_coefficient * solution_numerator
            before_m = cur[:m.start()]
            after_m = cur[m.end():]

            # Always show the multiplication symbol explicitly.
            frac_prod = (
                r"\frac{" + str(term_coefficient) + r" \cdot " + solution_numerator_str
                + r"}{" + str(solution_denominator) + r"}"
            )

            s1 = _fix_pm(before_m + frac_prod + after_m)

            if s1 != cur: 
                result.append((s1, None)); cur = s1

            numerator_evaluated = r"\frac{" + str(numerator_product) + r"}{" + str(solution_denominator) + r"}"
            fraction_reduced = sp.latex(sp.Rational(numerator_product, solution_denominator))
            simplification_is_integer = "\\frac" not in fraction_reduced

            if fraction_reduced == numerator_evaluated:
                # Already in lowest terms; show the evaluated form and stop.
                s2 = _fix_pm(cur.replace(frac_prod, numerator_evaluated, 1))

                if s2 != cur: 
                    result.append((s2, None)); cur = s2
            elif simplification_is_integer:
                # Numerator and denominator cancel completely; jump to integer.
                s3 = _fix_pm(cur.replace(frac_prod, fraction_reduced, 1))

                if s3 != cur: 
                    result.append((s3, None)); cur = s3
            else:
                # Show the evaluated numerator first, then reduce the fraction.
                s2 = _fix_pm(cur.replace(frac_prod, numerator_evaluated, 1))

                if s2 != cur: 
                    result.append((s2, None)); cur = s2

                s3 = _fix_pm(cur.replace(numerator_evaluated, fraction_reduced, 1))

                if s3 != cur: 
                    result.append((s3, None)); cur = s3

        # Step 1b: frac_coef * (p/q).
        pat_frac_frac = r"\\frac\{(\d+)\}\{(\d+)\}\s*\(\s*" + solution_latex_escaped + r"\s*\)"

        for _ in range(20):
            m = _re.search(pat_frac_frac, cur)
            if not m: break
            frac_coef_numerator = int(m.group(1))
            frac_coef_denominator = int(m.group(2))
            frac_product_numerator   = frac_coef_numerator * solution_numerator
            frac_product_denominator   =  frac_coef_denominator * solution_denominator
            before_m = cur[:m.start()]
            after_m  = cur[m.end():]

            frac_product_shown = (
                r"\frac{" + str(frac_coef_numerator) + r" \cdot " + solution_numerator_str
                + r"}{" + str( frac_coef_denominator) + r" \cdot " + str(solution_denominator) + r"}"
            )
            s1 = _fix_pm(before_m + frac_product_shown + after_m)
            if s1 != cur: result.append((s1, None)); cur = s1

            frac_numerator_evaluated   = r"\frac{" + str(frac_product_numerator) + r"}{" + str(frac_product_denominator) + r"}"
            frac_fraction_reduced   = sp.latex(sp.Rational(frac_product_numerator, frac_product_denominator))
            frac_simplification_is_integer = "\\frac" not in frac_fraction_reduced

            if frac_fraction_reduced == frac_numerator_evaluated:
                s2 = _fix_pm(cur.replace(frac_product_shown, frac_numerator_evaluated, 1))
                if s2 != cur: result.append((s2, None)); cur = s2
            elif frac_simplification_is_integer:
                s3 = _fix_pm(cur.replace(frac_product_shown, frac_fraction_reduced, 1))
                if s3 != cur: result.append((s3, None)); cur = s3
            else:
                s2 = _fix_pm(cur.replace(frac_product_shown, frac_numerator_evaluated, 1))
                if s2 != cur: result.append((s2, None)); cur = s2
                s3 = _fix_pm(cur.replace(frac_numerator_evaluated, frac_fraction_reduced, 1))
                if s3 != cur: result.append((s3, None)); cur = s3

        # Step 1c: simplify any unreduced positive constant fractions still
        # present in the string, e.g. \frac{6}{4} >> \frac{3}{2}.
        # These originate from the original equation text which _check_solution
        # preserves verbatim so that the display matches the input exactly.
        pat_const_frac = r"\\frac\{(\d+)\}\{(\d+)\}"
        for _ in range(20):
            found_any = False
            for m in _re.finditer(pat_const_frac, cur):

                const_frac_numerator = int(m.group(1))
                const_frac_denominator = int(m.group(2))
                g = math.gcd(const_frac_numerator,  const_frac_denominator)
                if g > 1:
                    const_frac_unreduced   = r"\frac{" + str(const_frac_numerator) + r"}{" + str( const_frac_denominator) + r"}"
                    const_frac_reduced = (
                        r"\frac{" + str(const_frac_numerator // g) + r"}{" + str( const_frac_denominator // g) + r"}"
                        if  const_frac_denominator // g > 1 else str(const_frac_numerator // g)
                    )
                    novo = _fix_pm(cur.replace(const_frac_unreduced, const_frac_reduced, 1))
                    if novo != cur:
                        result.append((novo, None)); cur = novo
                        found_any = True
                        break
            if not found_any:
                break

        # Step 2: convert any remaining isolated integers to the common
        # denominator of all fractions currently present in the expression.
        def _find_isolated_ints(s):
            """
            Find integers outside LaTeX braces that are not immediately
            followed by / or { (which would make them part of a fraction).
            """
            found = []; depth = 0; i = 0
            while i < len(s):
                c = s[i]
                if c == "{": depth += 1; i += 1; continue
                if c == "}": depth -= 1; i += 1; continue
                if depth > 0: i += 1; continue
                sign  = 1
                start = i
                if c == "-" and i + 1 < len(s) and s[i + 1].isdigit():
                    if i == 0 or s[i - 1] in " +=({":
                        sign = -1; i += 1; c = s[i]
                    else:
                        i += 1; continue
                if c.isdigit():
                    j = i
                    while j < len(s) and s[j].isdigit(): j += 1
                    if j < len(s) and s[j] in "/{": i = j; continue
                    try: n = sign * int(s[i:j])
                    except: i += 1; continue
                    if n != 0: found.append((start, j, n))
                    i = j
                else:
                    i += 1
            return found

        def _dens_in(s):
            """Extract all denominators from \\frac{...}{d} patterns."""
            return [int(d) for d in _re.findall(r"\\frac\{[^}]+\}\{(\d+)\}", s)]

        dens = _dens_in(cur)
        if dens and _find_isolated_ints(cur):
            from functools import reduce as _fr2
            common_denominator = _fr2(_mlcm, dens)
            result.append((cur, f"Reduce to common denominator ({common_denominator})"))
            for _ in range(30):
                iso = _find_isolated_ints(cur)
                if not iso: break
                start, end, n = iso[0]
                frac = r"\frac{" + str(n * common_denominator) + r"}{" + str(common_denominator) + r"}"
                novo = _fix_pm(cur[:start] + frac + cur[end:])
                if novo == cur: break
                result.append((novo, None)); cur = novo

        # Step 3: add or subtract fractions pairwise until only one remains.
        fp = r"(\\frac\{(-?\d+)\}\{(\d+)\})\s*([+-])\s*(\\frac\{(-?\d+)\}\{(\d+)\})"
        for _ in range(30):
            m = _re.search(fp, cur)
            if not m: break
            na, da = int(m.group(2)), int(m.group(3))
            op     = m.group(4)
            nb, db = int(m.group(6)), int(m.group(7))

            # Absorb a leading external minus sign into the first numerator.
            prefix = cur[:m.start()].rstrip()
            if prefix.endswith("-"):
                na  = -abs(na)
                cur = cur[:len(prefix) - 1].rstrip() + " " + cur[len(prefix):]
                m   = _re.search(fp, cur)
                if not m: break
                na, da = int(m.group(2)), int(m.group(3))
                na = -abs(na)
                op = m.group(4)
                nb, db = int(m.group(6)), int(m.group(7))

            # If denominators differ, align them first.
            if da != db:
                from math import lcm as _lcm2
                dc     = _lcm2(da, db)
                na_new = na * (dc // da)
                nb_new = nb * (dc // db)
                f_a    = r"\frac{" + str(na_new) + r"}{" + str(dc) + r"}"
                f_b    = r"\frac{" + str(nb_new) + r"}{" + str(dc) + r"}"
                novo   = _fix_pm(
                    cur[:m.start()] + f_a + " " + op + " " + f_b + cur[m.end():]
                )
                if novo == cur: break
                result.append((novo, None)); cur = novo
                continue

            # Same denominator: show the unevaluated combined numerator first.
            nb_s        = nb if op == "+" else -nb
            num_expr    = f"{na} + {nb}" if op == "+" else f"{na} - {nb}"
            frac_joined = r"\frac{" + num_expr + r"}{" + str(da) + r"}"
            novo_j = _fix_pm(cur[:m.start()] + frac_joined + cur[m.end():])
            if novo_j != cur:
                result.append((novo_j, None)); cur = novo_j

            # Then evaluate the numerator as an explicit intermediate step.
            soma_n         = na + nb_s
            frac_evaluated = r"\frac{" + str(soma_n) + r"}{" + str(da) + r"}"
            frac_result    = sp.latex(sp.Rational(soma_n, da))

            if frac_evaluated != frac_joined:
                novo_eval = _fix_pm(cur.replace(frac_joined, frac_evaluated, 1))
                if novo_eval != cur:
                    result.append((novo_eval, None)); cur = novo_eval

            # Finally simplify the fraction if it is reducible.
            if frac_result != frac_evaluated:
                novo = _fix_pm(cur.replace(frac_evaluated, frac_result, 1))
                if novo == cur: break
                result.append((novo, None)); cur = novo
            else:
                novo = _fix_pm(
                    cur.replace(frac_joined, frac_result, 1) if frac_joined in cur else cur
                )
                if novo != cur:
                    result.append((novo, None)); cur = novo

        final_str = sp.latex(sp.simplify(sym_evaled))
        if cur != final_str:
            result.append((final_str, None))

        # Remove consecutive duplicate entries while keeping explanations.
        deduped = [result[0]]
        for item in result[1:]:
            s, e = item
            if s != deduped[-1][0] or e is not None:
                deduped.append(item)
        return deduped

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
        def _fracs_to_latex(s):
            return re.sub(
                r"(-?\d+)/(\d+)",
                lambda m: r"\frac{" + m.group(1) + r"}{" + m.group(2) + r"}",
                s,
            )

        left_disp  = _fracs_to_latex(left_str.strip())
        right_disp = _fracs_to_latex(right_str.strip())
        # Add a space between \frac{a}{b} and a following variable letter.
        left_disp  = re.sub(r"(\\frac\{\d+\}\{\d+\})(\s*)([a-zA-Z])", r"\1 \3", left_disp)
        right_disp = re.sub(r"(\\frac\{\d+\}\{\d+\})(\s*)([a-zA-Z])", r"\1 \3", right_disp)
        equation_latex = f"{left_disp} = {right_disp}"

        steps.append(Step(
            before=equation_latex, after=equation_latex,
            explanation="Let's verify!",
        ))
        steps.append(Step(
            before=equation_latex, after=equation_latex,
            explanation=f"Now substitute x for {final_latex}",
        ))

        # Replace every x occurrence in the LaTeX string with (value).
        solution_latex = sp.latex(final_value)

        def _sub_x(latex_str, val):
            return re.sub(
                r"(?<![a-zA-Z])x(?![a-zA-Z])",
                lambda _m: "(" + val + ")",
                latex_str,
            )

        left_subst  = _sub_x(left_disp,  solution_latex)
        right_subst = _sub_x(right_disp, solution_latex)

        steps.append(Step(
            before=equation_latex,
            after=f"{left_subst} = {right_subst}",
        ))

        # Substitute into the SymPy expressions, keeping each Add term
        # separate so the stepwise evaluator can process them individually.
        def _subst_terms(sym, val):
            if isinstance(sym, sp.Add):
                new_t = [t.xreplace({x: sp.UnevaluatedExpr(val)}) for t in sym.args]
                return sp.Add(*new_t, evaluate=False)
            return sym.xreplace({x: sp.UnevaluatedExpr(val)})

        left_evaled  = _subst_terms(left_sym,  final_value)
        right_evaled = _subst_terms(right_sym, final_value)

        left_tuples  = _sympy_stepwise(left_subst,  left_evaled,  final_value)
        right_tuples = _sympy_stepwise(right_subst, right_evaled, final_value)

        def _extract(tuples):
            """Split (latex, explanation) tuples into a step list and an
            explanation map keyed by step index."""
            sl, ex = [], {}
            for i, (s, e) in enumerate(tuples):
                sl.append(s)
                if e: ex[i] = e
            return sl, ex

        left_steps_v,  left_expls  = _extract(left_tuples)
        right_steps_v, right_expls = _extract(right_tuples)

        # Animate the left side first, keeping the right side fixed.
        cur_left, cur_right = left_subst, right_subst

        for i, l_after in enumerate(left_steps_v):
            expl   = left_expls.get(i)
            before = f"{cur_left} = {cur_right}"
            if expl:
                steps.append(Step(before=before, after=before, explanation=expl))
            after = f"{l_after} = {cur_right}"
            if before != after:
                steps.append(Step(before=before, after=after))
            cur_left = l_after

        # Then animate the right side, keeping the left side fixed.
        for i, r_after in enumerate(right_steps_v):
            expl   = right_expls.get(i)
            before = f"{cur_left} = {cur_right}"
            if expl:
                steps.append(Step(before=before, after=before, explanation=expl))
            after = f"{cur_left} = {r_after}"
            if before != after:
                steps.append(Step(before=before, after=after))
            cur_right = r_after

        # Evaluate both sides numerically and confirm whether they are equal.
        left_orig  = sp.sympify(normalize_expression(left_str.strip()),  evaluate=False)
        right_orig = sp.sympify(normalize_expression(right_str.strip()), evaluate=False)
        is_true    = sp.simplify(
            left_orig.subs(x, final_value) - right_orig.subs(x, final_value)
        ) == 0
        explanation = (
            "The solution is correct!"
            if is_true
            else "The solution does not satisfy the equation."
        )

        steps.append(Step(
            before=f"{cur_left} = {cur_right}",
            after=f"{cur_left} = {cur_right}",
            explanation=explanation,
        ))

    # ------------------------------------------------------------------
    # Case 1: coefficient of x is already 1 -- no division needed.
    # ------------------------------------------------------------------
    if coef == 1:
        steps.append(Step(
            before=f"{sp.latex(final_left)} = {sp.latex(final_right)}",
            after=f"x = {sp.latex(const)}",
        ))
        final_value = const
        final_latex = sp.latex(final_value)
        dec = _decimal_str(final_value)
        if dec:
            sol_latex = f"x = {final_latex}"
            steps.append(Step(
                before=sol_latex, after=sol_latex,
                explanation=f"x = {final_latex} \\approx {dec}",
            ))
        _check_solution(final_value, final_latex, equation, steps)

    # ------------------------------------------------------------------
    # Case 2: coefficient of x requires a division step.
    # ------------------------------------------------------------------
    else:
        coef_rat  = sp.Rational(coef)
        const_rat = sp.Rational(const)

        final_left_latex  = sp.latex(final_left)
        final_right_latex = sp.latex(final_right)

        # Sub-case 2a: fractional coefficient (e.g. -5/2 x = -12).
        # Delegate to the dedicated function that handles the multiply/divide sequence.
        if coef_rat.q > 1:
            extra_steps, solution = _rational_coef_solve_steps(
                coef_rat, const_rat,
                final_left_latex, final_right_latex,
            )
            steps.extend(extra_steps)

        # Sub-case 2b: integer coefficient (e.g. 15x = 3).
        else:
            # Check whether a partial simplification step is possible before
            # the final division (e.g. 12x = 8  --div 4-->  3x = 2  --div 3-->  x = 2/3).
            divisor_intermedio = safe_gcd(abs(const_rat), abs(coef_rat))
            coef_simpl  = coef_rat  / divisor_intermedio
            const_simpl = const_rat / divisor_intermedio
            mostrar_intermedio = (
                divisor_intermedio > 1
                and divisor_intermedio != abs(coef_rat)
                and (isinstance(coef_simpl,  sp.Integer) or coef_simpl.q  == 1)
                and (isinstance(const_simpl, sp.Integer) or const_simpl.q == 1)
                and coef_simpl != 1
            )

            if mostrar_intermedio:
                left_int  = coef_simpl * x
                right_int = const_simpl
                steps.append(Step(
                    before=f"{final_left_latex} = {final_right_latex}",
                    after=f"{final_left_latex} = {final_right_latex}",
                    explanation=f"Divide both sides by {sp.latex(divisor_intermedio)}",
                ))
                steps.append(Step(
                    before=f"{final_left_latex} = {final_right_latex}",
                    after=f"{sp.latex(left_int)} = {sp.latex(right_int)}",
                ))
                steps.append(Step(
                    before=f"{sp.latex(left_int)} = {sp.latex(right_int)}",
                    after=f"{sp.latex(left_int)} = {sp.latex(right_int)}",
                    explanation=f"Divide both sides by {sp.latex(coef_simpl)}",
                ))
                steps.append(Step(
                    before=f"{sp.latex(left_int)} = {sp.latex(right_int)}",
                    after=f"x = {sp.latex(sp.Rational(const_rat, coef_rat))}",
                ))
            else:
                # Direct division: show x = const/coef unreduced first,
                # then simplify (e.g. 15x = 3  >>  x = 3/15  >>  x = 1/5).
                solution = sp.Rational(const_rat, coef_rat)
                frac_unreduced  = (
                    r"\frac{" + str(int(const_rat)) + r"}{" + str(int(coef_rat)) + r"}"
                )
                frac_simplified = sp.latex(solution)
                steps.append(Step(
                    before=f"{final_left_latex} = {final_right_latex}",
                    after=f"{final_left_latex} = {final_right_latex}",
                    explanation=f"Divide both sides by {sp.latex(coef_rat)}",
                ))
                steps.append(Step(
                    before=f"{final_left_latex} = {final_right_latex}",
                    after=f"x = {frac_unreduced}",
                ))
                if frac_simplified != frac_unreduced:
                    steps.append(Step(
                        before=f"x = {frac_unreduced}",
                        after=f"x = {frac_simplified}",
                    ))

            solution = sp.Rational(const_rat, coef_rat)

        final_value = solution
        final_latex = sp.latex(final_value)

        dec = _decimal_str(final_value)
        if dec:
            sol_latex = f"x = {final_latex}"
            steps.append(Step(
                before=sol_latex, after=sol_latex,
                explanation=f"x = {final_latex} \\approx {dec}",
            ))

        _check_solution(final_value, final_latex, equation, steps)

    return steps

