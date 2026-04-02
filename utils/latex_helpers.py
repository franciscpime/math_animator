import re
import sympy as sp

x = sp.symbols("x")


def eq_to_latex_display(eq):
    """
    Convert every plain a/b fraction in an equation string to a LaTeX
    \\frac{a}{b} so the equation renders correctly in the animation.

    Example:
      '1/2 x + 3 = 7/4'  >>  '\\frac{1}{2} x + 3 = \\frac{7}{4}'
    """
    return re.sub(
        r"(?<!\\\\)(-?\d+)/(\d+)",
        lambda m: r"\frac{" + m.group(1) + r"}{" + m.group(2) + r"}",
        eq,
    )


def coef_rational(term):
    """Return the rational coefficient of a term containing x."""
    return sp.Rational(term.coeff(x))


def frac_x_latex(numerator, denominator):
    """
    Return LaTeX for (numerator/denominator)*x, keeping the denominator explicit.

    Examples:
      frac_x_latex(1, 2)   >> '\\frac{x}{2}'
      frac_x_latex(-6, 2)  >> '- \\frac{6 x}{2}'
      frac_x_latex(10, 1)  >> '10 x'
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


def frac_latex(numerator, denominator):
    """
    Return the LaTeX string for numerator/denominator, always showing the
    denominator explicitly.

    Examples:
      frac_latex(18, 2)   >> '\\frac{18}{2}'
      frac_latex(-9, 2)   >> '- \\frac{9}{2}'
      frac_latex(8, 1)    >> '8'
    """
    if denominator == 1:
        return str(numerator)

    if numerator < 0:
        return fr"- \frac{{{abs(numerator)}}}{{{denominator}}}"

    return fr"\frac{{{numerator}}}{{{denominator}}}"


def join_latex(parts):
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


def decimal_str(solution):
    """
    Convert a rational solution to a decimal string rounded to 3 significant
    figures. Returns None when the solution is already an integer.
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

