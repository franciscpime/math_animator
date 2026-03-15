import sympy as sp

from models.step import EquationStep

# Define the symbolic variable x used in the equation
x = sp.symbols("x")

def solve_quadratic(
        polynomial, 
        equation, 
        mmc, 
        scaled_expression, 
        is_factorized, 
        m, n, o, p
    ):

    # This list will store each transformation step of the solution
    steps = []

    if is_factorized:
        steps.append(
            EquationStep(
                before = equation,
                after = equation
            )
        )

        steps.append(
            EquationStep(
                before = equation,
                after = "(m+n)(o+p) = 0"
            )
        )

        steps.append(
            EquationStep(
                before = "(m+n)(o+p) = 0",
                after = "(m)(o)+(m)(p)+(n)(o)+(n)(p) = 0"
            )
        )
    
        steps.append(
            EquationStep(
                before = "(m)(o)+(m)(p)+(n)(o)+(n)(p) = 0",
                after = f"({sp.latex(m)})({sp.latex(o)})+({sp.latex(m)})({sp.latex(p)})+({sp.latex(n)})({sp.latex(o)})+({sp.latex(n)})({sp.latex(p)}) = 0"
            )
        )

        mo = m * o 
        mp = m * p 
        no = n * o 
        np = n * p 

        expr = sp.Add(mo, mp, no, np, evaluate=False)

        steps.append(
            EquationStep(
                before = f"({sp.latex(m)})({sp.latex(o)})+({sp.latex(m)})({sp.latex(p)})+({sp.latex(n)})({sp.latex(o)})+({sp.latex(n)})({sp.latex(p)}) = 0",
                after = f"{sp.latex(expr)} = 0"
            )
        )

        simplified = sp.simplify(expr)

        steps.append(
            EquationStep(
                before = f"{sp.latex(expr)} = 0",
                after = f"{sp.latex(simplified)} = 0"
            )
        )

        scaled_expression = simplified

    # Extract coefficients a, b and c from ax^2 + bx + c
    a, b, c = polynomial.all_coeffs()

    abc_text = f"a={sp.latex(a)},\\; b={sp.latex(b)},\\; c={sp.latex(c)}"

    # Compute the discriminant delta = b^2 − 4ac
    delta = b**2 - 4*a*c

    # Example evolving through the steps:
    # 2x² + 3x − 2 = 0
    if not is_factorized:
        steps.append(
            EquationStep(
                before = equation,
                after = equation,
                explanation = "Quadratic function"
            )
        )

        # If denominators exist, multiply both sides by the least common multiple (mmc)
        # Example:
        # x^2/2 + 3x/2 - 1 = 0  >> multiply by 2 >>  x^2 + 3x − 2 = 0
        if mmc != 1:
            steps.append(
                EquationStep(
                    before = equation,
                    after = equation,
                    explanation = f"Multiply both sides by {mmc}"
                )
            )

            # After multiplying, we obtain the scaled polynomial equation
            steps.append(
                EquationStep(
                    before = equation,
                    after = scaled_expression
                )
            )
    
        else:
            # If no scaling is required, the equation already has integer coefficients
            # Example: 2x^2 + 3x − 2 = 0
            steps.append(
                EquationStep(
                    before = equation,
                    after = scaled_expression
                )
            )

    # Identify the coefficients of the quadratic equation
    # Example: a = 2, b = 3, c = -2
    steps.append(
        EquationStep(
            before = scaled_expression,
            after = scaled_expression,
            explanation = abc_text
        )
    )

    # Introduce the discriminant formula
    # delta = b^2 − 4ac
    steps.append(
        EquationStep(
            before = scaled_expression,
            after = "\\Delta=b^2 - 4(a)(c)"
        )
    )

    # Substitute the coefficients into the formula
    # Example: delta = 3^2 − 4(2)(−2)
    steps.append(
        EquationStep(
            before = "\\Delta=b^2 - 4(a)(c)",
            after = f"\\Delta=({sp.latex(b)})^2 - 4({sp.latex(a)})({sp.latex(c)})"
        )
    )

    if a == 0 or c == 0:

        # Special simplification if a or c is zero
        steps.append(
            EquationStep(
                before = f"\\Delta={sp.latex(b)}^2 - 4({sp.latex(a)})({sp.latex(c)})",
                after = f"\\Delta={sp.latex(b**2)} + {sp.latex(-4 * a * c)}"
            )
        )

        steps.append(
            EquationStep(
                before = f"\\Delta={sp.latex(b)}^2 - 4({sp.latex(a)})({sp.latex(c)})",
                after = f"\\Delta={sp.latex(b**2)}"
            )
        )

    else:
        # Expand the multiplication
        # Example: delta = 3^2 + (-8)(-2)
        steps.append(
            EquationStep(
                before = f"\\Delta={sp.latex(b)}^2 - 4({sp.latex(a)})({sp.latex(c)})",
                after = f"\\Delta={sp.latex(b**2)} + ({sp.latex(-4 * a)})({sp.latex(c)})"
            )
        )

    expression1 = sp.Add(b**2, -4*a*c, evaluate=False)
    expression2 = sp.Add(-4*a*c, evaluate=False)

    if b != 0:
    
        # Combine the terms
        # Example: delta = 9 + 16
        steps.append(
            EquationStep(
                before = f"\\Delta={sp.latex(b**2)} + ({sp.latex(-4 * a)})({sp.latex(c)})",
                after = f"\\Delta={sp.latex(expression1)}"
            )
        )
    else: 
        steps.append(
            EquationStep(
                before = f"\\Delta={sp.latex(b**2)} + ({sp.latex(-4 * a)})({sp.latex(c)})",
                after = f"\\Delta={sp.latex(expression2)}"
            )   
        )

    # Final value of the discriminant
    # Example: delta = 25
    steps.append(
        EquationStep(
            before = f"\\Delta={sp.latex(b**2)} + {sp.latex(-4*a*c)}",
            after = f"\\Delta={sp.latex(delta)}"
        )
    )

    if delta < 0:

        # If delta < 0, there are no real solutions
        steps.append(
            EquationStep(
                before = f"\\Delta={sp.latex(delta)}",
                after = "\\Delta < 0"
            )
        )

        steps.append(
            EquationStep(
                before = "\\Delta < 0",
                after = "\\text{There are no real solutions}"
            )
        )

        return steps

    else:

        # Apply the quadratic formula
        # x = (-b ± sqrt(delta)) / 2a
        steps.append(
            EquationStep(
                before = f"\\Delta={sp.latex(delta)}",
                after = "x=\\frac{-b\\pm\\sqrt{\\Delta}}{2(a)}"
            )
        )

        latex_b = sp.latex(-b)
        latex_delta = sp.latex(delta)
        latex_sqrt_delta = sp.latex(sp.sqrt(delta))
        latex_den = sp.latex(2 * a)

        # Substitute numeric values into the formula
        # Example: x = (-3 ± sqrt(25)) / 4
        steps.append(
            EquationStep(
                before = "x=\\frac{-b\\pm\\sqrt{\\Delta}}{2a}",
                after = f"x=\\frac{{{latex_b}\\pm\\sqrt{{{latex_delta}}}}}{{2({sp.latex(a)})}}"
            )
        )

        # Simplify the square root
        # Example: sqrt(25) = 5
        steps.append(
            EquationStep(
                before = f"x=\\frac{{{latex_b}\\pm\\sqrt{{{latex_delta}}}}}{{2({sp.latex(a)})}}",
                after = f"x=\\frac{{{latex_b}\\pm {latex_sqrt_delta}}}{{{latex_den}}}"
        ))

        den = 2 * a
        
        if b == 0:
            num1 = sp.Add(sp.sqrt(delta), evaluate=False)
            num2 = sp.Add(-sp.sqrt(delta), evaluate=False)
        else:
            num1 = sp.Add(-b, sp.sqrt(delta), evaluate=False)
            num2 = sp.Add(-b, -sp.sqrt(delta), evaluate=False)
        

        latex_num1 = sp.latex(num1)
        latex_num2 = sp.latex(num2)

        # Separate the two possible solutions
        # Example:
        # x = (-3 + 5)/4  or  x = (-3 − 5)/4
        steps.append(
            EquationStep(
                before = f"x=\\frac{{{latex_b}\\pm {latex_sqrt_delta}}}{{{latex_den}}}",
                after = f"x=\\frac{{{latex_num1}}}{{{latex_den}}} \\quad \\lor \\quad "
                        f"x=\\frac{{{latex_num2}}}{{{latex_den}}}"
        ))

        fact_num1 = sp.factor(num1)
        fact_num2 = sp.factor(num2)

        # Keep the original numerator form first
        latex_num1_raw = sp.latex(num1)
        latex_num2_raw = sp.latex(num2)

        # Step: show the two separate fractions
        steps.append(
            EquationStep(
                before = f"x=\\frac{{{latex_b}\\pm {latex_sqrt_delta}}}{{{latex_den}}}",
                after = f"x=\\frac{{{latex_num1_raw}}}{{{latex_den}}} \\quad \\lor \\quad "
                        f"x=\\frac{{{latex_num2_raw}}}{{{latex_den}}}"
            )
        )

        # Step: factor the numerators
        steps.append(
            EquationStep(
                before = f"x=\\frac{{{latex_num1_raw}}}{{{latex_den}}} \\quad \\lor \\quad "
                        f"x=\\frac{{{latex_num2_raw}}}{{{latex_den}}}",
                after = f"x=\\frac{{{sp.latex(fact_num1)}}}{{{latex_den}}} \\quad \\lor \\quad "
                        f"x=\\frac{{{sp.latex(fact_num2)}}}{{{latex_den}}}"
            )
        )

        sol1 = sp.simplify(fact_num1 / den)
        sol2 = sp.simplify(fact_num2 / den)

        # Final simplified solutions
        steps.append(
            EquationStep(
                before = f"x=\\frac{{{sp.latex(fact_num1)}}}{{{latex_den}}} \\quad \\lor \\quad "
                        f"x=\\frac{{{sp.latex(fact_num2)}}}{{{latex_den}}}",
                after = f"x={sp.latex(sol1)} \\quad \\lor \\quad x={sp.latex(sol2)}"
            )
        )

        solution = sp.solve(polynomial, x)

        # Present the final solution set
        steps.append(
            EquationStep(
                before = f"x={sp.latex(sol1)} \\quad \\lor \\quad x={sp.latex(sol2)}",
                after = f"Solution: {sp.latex(solution)}"
            )
        )

        return steps
    