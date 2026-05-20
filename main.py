from manim import *
from controller.animation_controller import AnimationController

class SolveScene(Scene):
    def construct(self):
        equation = input("Equation: ")
        mode = input("Mode (solve/guess): ").strip().lower()
        controller = AnimationController(self)
        controller.run(equation, mode)

        