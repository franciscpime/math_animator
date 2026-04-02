import math
import sympy as sp


def fraction_simplification_steps(num_str, den_str):
    """
    Return a list of LaTeX strings showing the reduction of a/b to its lowest
    terms. The list always starts with the unreduced form; if it is already in
    lowest terms the list contains only one element.

    Example:
      fraction_simplification_steps('6', '4')
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


def decimal_simplification_steps(decimal_str):
    """
    Return a list of LaTeX strings walking through the conversion of a decimal
    to a fraction in lowest terms.

    Example:
      decimal_simplification_steps('0.5')
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

