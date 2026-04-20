import sympy as sp
from models.step import Step
from math_utils.gcd_utils import safe_gcd

'''
This function generates the animation steps for solving linear equations of the form n*x = c, 
where n is a non-zero integer.

Two sub-cases are handled:

    Sub-case A: two-stage division (intermediate simplification possible):
      When the GCD of |c| and |n| is greater than 1 but does not equal |n|,
      and both simplified values are still whole numbers, we first divide by
      the GCD to reduce both sides, then divide by the remaining coefficient.

      Example: 12x = 8
        Divide both sides by 4  >>  3x = 2
        Divide both sides by 3  >>  x = 2/3

    Sub-case B -- direct division:
      In all other cases we divide directly by n, showing the unreduced
      fraction first and then the SymPy-simplified result.

      Example: 15x = 3
        Divide both sides by 15  >>  x = 3/15  >>  x = 1/5

Parameters
----------
coefficient_rational  -- coefficient of x (integer, q == 1)
constant_rational     -- right-hand side constant
final_left_latex      -- LaTeX of the left side before solving
final_right_latex     -- LaTeX of the right side before solving
'''
def integer_coef_solve_steps(
        coefficient_rational: sp.Rational,
        constant_rational: sp.Rational,
        final_left_latex: str,
        final_right_latex: str
    ):

    x = sp.symbols("x")

    result_steps = []

    # Find the greatest common divisor of |constant| and |coefficient|
    # Example: coefficient = 12, constant = 8 >> common_divisor = 4
    common_divisor = safe_gcd(
        abs(constant_rational),
        abs(coefficient_rational)
    )

    # Divide the coefficient by the common divisor to get the simplified version
    # Example: coefficient = 12, common_divisor = 4 >> simplified_coef = 3
    simplified_coef = coefficient_rational / common_divisor

    # Divide the constant by the common divisor to get the simplified version
    # Example: constant=8, common_divisor=4  >>  simplified_const = 2
    simplified_const = constant_rational / common_divisor

    # Check whether the simplified coefficient is still a whole number
    # Example: simplified_coef = 3 >> coef_is_whole = True
    #          simplified_coef = 3/2 >> coef_is_whole = False
    coef_is_whole = isinstance(simplified_coef, sp.Integer) or simplified_coef.q == 1

    # Check whether the simplified constant is still a whole number
    # Example: simplified_const = 2 >> const_is_whole = True
    #          simplified_const = 2/3 >> const_is_whole = False
    const_is_whole = isinstance(simplified_const, sp.Integer) or simplified_const.q == 1

    # Determine whether a two-stage division is worth showing.
    # All five conditions must be true:
    #   1. common_divisor > 1              -- there is actually something to simplify first
    #   2. common_divisor != |coefficient| -- the first division does not already isolate x
    #   3. coef_is_whole                   -- simplified coefficient is a whole number
    #   4. const_is_whole                  -- simplified constant is a whole number
    #   5. simplified_coef != 1            -- after first division, x is still not isolated
    # Example: 12x = 8  >>  common_divisor = 4, simplified_coef = 3, simplified_const = 2
    #          all conditions met >> has_intermediate_step = True
    has_intermediate_step = (
        common_divisor > 1
        and common_divisor != abs(coefficient_rational)
        and coef_is_whole
        and const_is_whole
        and simplified_coef != 1
    )

    if has_intermediate_step:

        # Sub-case A

        # Build the left side after the first division (by common_divisor)
        # Example: simplified_coef = 3, x  >>  left_after_division = 3x
        left_after_division = simplified_coef * x

        # Build the right side after the first division (by common_divisor)
        # Example: simplified_const = 2 >> right_after_division = 2
        right_after_division = simplified_const

        # Step 1: announce the first division (by the common divisor)
        result_steps.append(
            Step(
                # Show the equation unchanged and the explanation announcement
                before = f"{final_left_latex} = {final_right_latex}",
                after = f"{final_left_latex} = {final_right_latex}",
                explanation = f"Divide both sides by {sp.latex(common_divisor)}",
            )
        )

        # Step 2: show the equation after dividing by the common divisor
        # Example: 12x = 8 >> 3x = 2
        result_steps.append(
            Step(
                before = f"{final_left_latex} = {final_right_latex}",
                after = f"{sp.latex(left_after_division)} = {sp.latex(right_after_division)}",
            )
        )

        # Step 3: announce the second division (by the remaining coefficient)
        result_steps.append(
            Step(
                # Show the equation unchanged -- this step is just the explanation announcement
                before = f"{sp.latex(left_after_division)} = {sp.latex(right_after_division)}",
                after = f"{sp.latex(left_after_division)} = {sp.latex(right_after_division)}",
                explanation = f"Divide both sides by {sp.latex(simplified_coef)}"
            )
        )

        result_steps.append(
            Step(
                # Show the equation unchanged -- this step is just the explanation announcement
                before = f"{sp.latex(left_after_division)} = {sp.latex(right_after_division)}",
                after = f"\\frac{{{sp.latex(left_after_division)}}}{{{simplified_coef}}} = \\frac{{{sp.latex(right_after_division)}}}{{{simplified_coef}}}"
            )
        )


        # Compute the exact final solution as a SymPy Rational
        # Example: Rational(8, 12)  >>  2/3
        solution = sp.Rational(constant_rational, coefficient_rational)

        # Step 4: show the final isolated x
        # Example: '3x = 2'  >>  'x = 2/3'
        result_steps.append(
            Step(
                before = f"\\frac{{{sp.latex(left_after_division)}}}{{{simplified_coef}}} = \\frac{{{sp.latex(right_after_division)}}}{{{simplified_coef}}}",
                after = f"x = {sp.latex(solution)}",
            )
        )

    else:

        # Sub-case B

        # Compute the exact solution as a SymPy Rational
        # Example: Rational(3, 15)  >>  1/5
        solution = sp.Rational(constant_rational, coefficient_rational)

        # Build the unreduced fraction string to show before SymPy simplifies it
        # Example: constant = 3, coefficient = 15  >>  '\frac{3}{15}'
        unreduced_fraction = (
            r"\frac{"
            + str(int(constant_rational))
            + r"}{"
            + str(int(coefficient_rational))
            + r"}"
        )

        # Let SymPy produce the fully reduced form
        # Example: Rational(3, 15)  >>  '\frac{1}{5}'
        reduced_fraction = sp.latex(solution)

        # Step 1: announce the division
        result_steps.append(
            Step(
                # Show the equation unchanged and the explanation announcement
                before = f"{final_left_latex} = {final_right_latex}",
                after = f"{final_left_latex} = {final_right_latex}",
                explanation = f"Divide both sides by {sp.latex(coefficient_rational)}",
            )
        )

        result_steps.append(
            Step(
                # Show the equation unchanged and the explanation announcement
                before = f"{final_left_latex} = {final_right_latex}",
                after = f"\\frac{{{sp.latex(coefficient_rational)}}}{{{sp.latex(coefficient_rational)}}} x" 
                        f"= {final_right_latex} : {sp.latex(coefficient_rational)}"
            )
        )

        
        result_steps.append(
            Step(
                # Show the equation unchanged and the explanation announcement
                before = f"\\frac{{{sp.latex(coefficient_rational)}}}{{{sp.latex(coefficient_rational)}}} x" 
                        f"= {final_right_latex} : {sp.latex(coefficient_rational)}",
                after = f"x = {final_right_latex} \\cdot \\frac{1}{{{sp.latex(coefficient_rational)}}}"
            )
        )
       
        numerador = constant_rational.p
        denominador = constant_rational.q

        result_steps.append(
            Step(
                # Show the equation unchanged and the explanation announcement
                before = f"x = {final_right_latex} \\cdot \\frac{1}{{{sp.latex(coefficient_rational)}}}",
                after = f"x = \\frac{{{numerador} \\cdot {1}}}{{{denominador} \\cdot {sp.latex(coefficient_rational)}}}"
            )
        )

        result_steps.append(
            Step(
                # Show the equation unchanged and the explanation announcement
                before = f"x = \\frac{{{numerador} \\cdot {1}}}{{{denominador} \\cdot {sp.latex(coefficient_rational)}}}",
                after = f"x = \\frac{{{numerador * 1}}}{{{denominador * coefficient_rational}}}"
            )
        )

        result_steps.append(
            Step(
                before = f"x = \\frac{{{numerador * 1}}}{{{denominador * coefficient_rational}}}",
                after = f"x = {solution}"
            )
        )

        

    # Return the list of animation steps and the exact numeric solution
    return result_steps, solution