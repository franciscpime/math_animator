import sympy as sp
from manim import *
from utils.latex_formatter import to_latex


class EquationRenderer:

    def __init__(self, scene: Scene):
        self.scene = scene
        self.tex = None
        self.explanation_tex = None


    def animate(self, steps):

        first = to_latex(steps[0].before)

        self.tex = MathTex(first).scale(1.4)
        self.scene.play(Write(self.tex))

        for step in steps:

            after = to_latex(step.after)

            new_tex = MathTex(after).scale(1.4)
            new_tex.move_to(self.tex)

            self.scene.play(
                ReplacementTransform(self.tex, new_tex)
            )

            self.tex = new_tex

            if step.explanation:

                explanation = Tex(step.explanation).scale(0.9)
                explanation.next_to(self.tex, DOWN)

                self.scene.play(Write(explanation))

                self.explanation_tex = explanation

            self.scene.wait(1)