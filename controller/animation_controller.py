from animation.equation_renderer import EquationRenderer
from solvers.linear_solver import solve_linear
from utils.guess_checker import guess

class AnimationController:
    def __init__(self, scene):
        self.renderer = EquationRenderer(scene)

    def run(self, equation, mode, x_value=None):
        if mode == "solve":
            steps = solve_linear(equation)
        elif mode == "guess":
            steps = guess(equation, x_value)
        
        for i, step in enumerate(steps):
            print("-" * 30)
            print(f"Step {i}: before={repr(step.before)}")
            print(f"after={repr(step.after)}")
            print(f"explanation={repr(step.explanation)}")
        
        self.renderer.animate(steps)

