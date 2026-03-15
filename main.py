from manim import *
from controller.animation_controller import AnimationController


class SolveScene(Scene):

    def construct(self):

        equation = "(x+2)(x+3)=0"

        controller = AnimationController(self)

        controller.run(equation)