from solver.utils.term_renderer import render_terms


def build_equation(left_terms, right_terms):

    left_side = render_terms(left_terms)

    right_side = render_terms(right_terms)

    return f"{left_side} = {right_side}"