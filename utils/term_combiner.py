import sympy as sp

x = sp.symbols("x")


def combine_terms_stepwise(terms):

    new_terms = terms.copy()

    steps = []

    while len(new_terms) > 1:

        term1 = new_terms[-2]
        term2 = new_terms[-1]

        if term1.has(x):

            coef1 = term1.coeff(x)
            coef2 = term2.coeff(x)

            new_term = (coef1 + coef2) * x

        else:

            new_term = term1 + term2

        new_terms.pop()
        new_terms.pop()

        new_terms.append(new_term)

        steps.append(new_terms.copy())

    return steps