from manim import *
from controller.animation_controller import AnimationController

class SolveScene(Scene):
    def construct(self):
        equation = input("Equation: ")
        mode = input("Mode (solve/guess): ").strip().lower()
        if mode == "guess":
            x_value = input("x = ").strip()  
        else: 
            x_value = None
        controller = AnimationController(self)
        controller.run(equation, mode, x_value)

