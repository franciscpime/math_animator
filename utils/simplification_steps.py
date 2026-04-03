import math
import sympy as sp


'''
This function returns the steps to simplify a fraction.

Example:
      fraction_simplification_steps('6', '4')
      >> ['\\frac{6}{4}', '\\frac{3}{2}']
'''
def fraction_simplification_steps(num_str: str, den_str: str) -> list[str]:

    numerator = int(num_str)
    denominator = int(den_str)

    # Start with the unreduced fraction
    steps = [r"\frac{" + num_str + r"}{" + den_str + r"}"]

    # Find the greatest common divisor (GCD) of the numerator and denominator
    common_divisor = math.gcd(abs(numerator), denominator)

    # If the GCD is greater than 1, we can reduce the fraction
    # Example: 6/4 >> 3/2
    # Otherwise, example: 6/5 remains 6/5
    if common_divisor > 1:
        steps.append(
            r"\frac{"
            + str(numerator // common_divisor)
            + r"}{"
            + str(denominator // common_divisor)
            + r"}"
        )

    return steps


'''
This function returns the steps to convert a decimal to a fraction.

Example:
      decimal_simplification_steps('0.5')
      >> ['0.5', '\\frac{5}{10}', '\\frac{1}{2}']
'''
def decimal_simplification_steps(decimal_str: str) -> list[str]:

    # Normalize the decimal string, example: '0,5' >> '0.5'
    decimal_normalized = decimal_str.replace(",", ".")

    # If the decimal is a whole number
    if "." not in decimal_normalized:
        return [decimal_normalized]

    # Check if the decimal is negative (True if it starts with '-')
    is_negative = decimal_normalized.startswith("-")

    # Get the absolute value of the decimal
    decimal_abs = decimal_normalized.lstrip("-")

    # Get the number of decimal places
    decimal_places = len(decimal_abs.split(".")[1])
    denominator = 10 ** decimal_places

    # Get the numerator by removing the decimal point
    numerator = int(decimal_abs.replace(".", ""))

    if is_negative:
        numerator = -numerator

    # Create the unreduced fraction
    frac_unreduced = r"\frac{" + str(numerator) + r"}{" + str(denominator) + r"}"
    steps = [decimal_str, frac_unreduced]

    # Find the greatest common divisor (GCD) of the numerator and denominator
    common_divisor = math.gcd(abs(numerator), denominator)

    # If the GCD is greater than 1, we can reduce the fraction
    # Example: 6/4 >> 3/2
    # Otherwise, example: 6/5 remains 6/5
    if common_divisor > 1:
        steps.append(
            r"\frac{"
            + str(numerator // common_divisor)
            + r"}{"
            + str(denominator // common_divisor)
            + r"}"
        )

    return steps

