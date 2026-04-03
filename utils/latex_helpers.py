import re
import sympy as sp

x = sp.symbols("x")


'''
This function converts an equation string to a LaTeX display format.

Example: '1/2 x + 3 = 7/4'  >>  '\\frac{1}{2} x + 3 = \\frac{7}{4}'
'''
def equation_to_latex_display(equation):
    return re.sub(
        r"(?<!\\\\)(-?\d+)/(\d+)",
        lambda m: r"\frac{" + m.group(1) + r"}{" + m.group(2) + r"}",
        equation,
    )


'''
This function converts a term containing x into a LaTeX display format.

Examples:

      frac_x_latex(1, 1)   >> '\\frac{x}{1}'
      frac_x_latex(-1, 1)  >> '- \\frac{x}{1}'
      frac_x_latex(0, 1)  >> '0'
      frac_x_latex(2, 1)  >> '\\frac{2}{1}'
      frac_x_latex(1, 4)  >> '- \\frac{x}{4}'
      frac_x_latex(-1, 4)  >> '- \\frac{x}{4}'
      frac_x_latex(-3, 2)   >> '- \\frac{3}{2}'
'''
def frac_x_latex(numerator, denominator):

    # If the denominator is 1
    if denominator == 1:
        # and the numerator is also 1 >> 1/1x = x
        if numerator == 1:
            return "x"
        # and the numerator is also -1 >> 1/1x = -x
        if numerator == -1:
            return "- x"
        # and the numerator is also 0 >> 0/1x = 0
        if numerator == 0:
            return "0"
        # otherwise return the integer >> 2/1x = 2
        return f"{numerator} x"

    # If the denominator is 1 >> 1/4x = x/4
    if numerator == 1:
        return fr"\frac{{x}}{{{denominator}}}"
    # If the numerator is also -1 >> -1/4x = -x/4
    if numerator == -1:
        return fr"- \frac{{x}}{{{denominator}}}"
    # If the numerator is negative >> -3/2x
    if numerator < 0:
        return fr"- \frac{{{abs(numerator)} x}}{{{denominator}}}"

    # Otherwise, return the fraction >> 3/2x
    return fr"\frac{{{numerator} x}}{{{denominator}}}"


'''
This function converts a fraction without x into a LaTeX display format.

Examples:
      frac_latex(3, 1)   >> '\\frac{3}{1}'
      frac_latex(-1, 4)  >> '- \\frac{1}{4}'
      frac_latex(3, 4)   >> '\\frac{3}{4}'
'''
def frac_latex(numerator, denominator):
    # If the denominator is 1 >> 3/1 = 3
    if denominator == 1:
        return str(numerator)

    # If the numerator is -1 >> -1/4 = -1/4
    if numerator < 0:
        return fr"- \frac{{{abs(numerator)}}}{{{denominator}}}"

    # Otherwise >> 3/4 = 3/4
    return fr"\frac{{{numerator}}}{{{denominator}}}"


'''
This function: 
    - Joins LaTeX terms with explicit +/- operators.
    - Terms that start with '- ' are treated as negative and attached directly.
    - All others are preceded by ' + '.

Example:
      ['\\frac{10 x}{5}', '\\frac{4 x}{5}', '- \\frac{15 x}{5}']
      >> '\\frac{10 x}{5} + \\frac{4 x}{5} - \\frac{15 x}{5}'
'''
def join_latex(parts):
    # Start with the first term as the base of the result string
    result = parts[0]

    # Iterate through the remaining parts
    for part in parts[1:]:
        if part.startswith("- "):
            result += " " + part
        else:
            result += " + " + part

    return result


'''
Converts a rational solution to a decimal string rounded to 3 decimal
places. Returns None when the solution is already a whole number, since
no approximation is needed in that case.

Examples:
    solution = 3         >>  None        (whole number, no approximation needed)
    solution = 2         >>  None        (whole number, no approximation needed)
    solution = 1/3       >>  "0.333"     (rounded to 3 decimal places)
    solution = 1/2       >>  "0.5"       (trailing zeros stripped: "0.500" >> "0.5")
    solution = 3/1       >>  None        (denominator is 1, counts as whole number)
'''
def decimal_str(solution):
    # Two cases count as whole numbers:
    #   - sp.Integer directly                    (e.g. Integer(3))
    #   - sp.Rational whose denominator is 1     (e.g. Rational(3, 1) == 3)
    if isinstance(solution, sp.Integer) or (
        isinstance(solution, sp.Rational) and solution.q == 1
    ):
        return None

    decimal_value = float(solution)
    rounded = round(decimal_value, 3)

    if rounded == int(rounded):
        return str(int(rounded))

    return f"{rounded:.3f}".rstrip("0").rstrip(".")

