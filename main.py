from manim import *
from controller.animation_controller import AnimationController

class SolveScene(Scene):

    def construct(self):

        equation = "1/2 + 4 = 0"

        controller = AnimationController(self)

        controller.run(equation)