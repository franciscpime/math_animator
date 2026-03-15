from manim import *
from controller.animation_controller import AnimationController


class EquationScene(Scene):
    """
    Main Manim scene responsible for rendering the animation.

    This scene delegates all logic to the AnimationController.
    The controller will:
        1. Analize and solve the equation
        2. Generate step-by-step transformations
        3. Render those steps using the renderer

    To test a different equation, just change >> equation = "..."
    """

    def construct(self):
        # Controller orchestrates the solving + animation pipeline
        controller = AnimationController()

        # Equation to solve and animate
        equation = "2x + 4 = 8"

        # Start the full pipeline
        controller.solve_and_animate(self, equation)

        # Pause at the end so the final result stays on screen
        self.wait(2)
