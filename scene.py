import os
from manim import *
from controller.animation_controller import AnimationController  # FIX G: ligado ao controller real
 
EXPRESSION = os.environ.get("MATH_ANIMATOR_EXPR", "x+1=0")
 
 
# ---------------------------
# RENDER NORMAL
# ---------------------------
class SolveScene(Scene):
    def construct(self):
        equation = os.environ.get("MATH_ANIMATOR_EXPR", "x+1=0")
 
        # FIX G: usa AnimationController em vez do SolveSceneLogic placeholder
        controller = AnimationController(self)
        controller.run(equation)
 
 
# ---------------------------
# MODO INTERATIVO
# ---------------------------
class SolveSceneInteractive(Scene):
    def construct(self):
        while True:
            expr = input(">>> nova expressão (ou 'exit'): ")
 
            if expr == "exit":
                break
 
            self.clear()
 
            controller = AnimationController(self)
            controller.run(expr)
 