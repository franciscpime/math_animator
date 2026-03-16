from manim import *
from controller.animation_controller import AnimationController

class SolveScene(Scene):

    def construct(self):

        equation = input("Equation: ")

        controller = AnimationController(self)

        controller.run(equation)

