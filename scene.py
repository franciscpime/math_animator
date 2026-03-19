import os
from manim import *

# vai buscar expressão do main.py
EXPRESSION = os.environ.get("MATH_ANIMATOR_EXPR", "x+1")


# ---------------------------
# A TUA LÓGICA (liga ao teu projeto)
# ---------------------------
class SolveSceneLogic:
    def build(self, scene, expr_str):
        # aqui ligas ao teu engine
        # exemplo simples:
        tex = MathTex(expr_str)

        scene.play(Write(tex))
        scene.wait()

        # depois substituis isto pelos teus Steps:
        # steps = solve(expr_str)
        # for step in steps:
        #     scene.play(step.to_animation())


# ---------------------------
# RENDER NORMAL
# ---------------------------
class SolveScene(Scene):
    def construct(self):
        SolveSceneLogic().build(self, EXPRESSION)


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

            SolveSceneLogic().build(self, expr)