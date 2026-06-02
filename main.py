from manim import *
from controller.animation_controller import AnimationController
from utils.ai_parser import parse_natural_language

class SolveScene(Scene):
    def construct(self):
        input_mode = input("Input mode (math/natural): ").strip().lower()
        
        if input_mode == "natural":
            user_input = input("Describe the equation: ")
            equation = parse_natural_language(user_input)
            print(f"Parsed equation: {equation}")
        else:
            equation = input("Equation: ")

        mode = input("Mode (solve/guess): ").strip().lower()

        if mode == "guess":
            x_value = input("x = ").strip()
        else:
            x_value = None

        controller = AnimationController(self)
        controller.run(equation, mode, x_value)

        