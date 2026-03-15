from solver.equation_solver import solve_equation
from animation.renderer import EquationRenderer


class AnimationController:

    def solve_and_animate(self, scene, equation: str):

        steps = solve_equation(equation)

        renderer = EquationRenderer(scene)

        renderer.animate(steps)