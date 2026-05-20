from animation.equation_renderer import EquationRenderer
from solvers.linear_solver import solve_linear

class AnimationController:
    def __init__(self, scene):
        self.renderer = EquationRenderer(scene)

    def run(self, equation, mode):
        if mode == "solve":
            steps = solve_linear(equation)
        elif mode == "guess":
            pass  
        
        for i, step in enumerate(steps):
            print("-" * 30)
            print(f"Step {i}: before={repr(step.before)}")
            print(f"after={repr(step.after)}")
            print(f"explanation={repr(step.explanation)}")
        
        self.renderer.animate(steps)

