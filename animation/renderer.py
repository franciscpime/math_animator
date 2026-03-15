import sympy as sp
from manim import *
from models.step import EquationStep


class EquationRenderer:

    def __init__(self, scene: Scene):
        self.scene = scene
        self.tex = None
        self.explanation_tex = None


    def animate(self, steps):

        def to_latex(expr):
            # if already string, return
            if isinstance(expr, str):
                return expr
        
            # if sympy expression, convert to latex
            if isinstance(expr, sp.Basic):
                return sp.latex(expr)
        
            return str(expr)

        # Primeira equação
        first = to_latex(steps[0].before)

        self.tex = MathTex(first).scale(1.4)
        self.scene.play(Write(self.tex))

        for step in steps[1:]:

            if self.explanation_tex:
                self.scene.play(FadeOut(self.explanation_tex))
                self.explanation_tex = None

            after = to_latex(step.after)

            new_tex = MathTex(after).scale(1.4)
            new_tex.move_to(self.tex)

            self.scene.play(
                ReplacementTransform(self.tex, new_tex)
            )

            self.tex = new_tex

            # explanation
            if step.explanation:

                explanation = Tex(step.explanation).scale(0.9)
                explanation.next_to(self.tex, DOWN)

                self.scene.play(Write(explanation))

                self.explanation_tex = explanation

            self.scene.wait(1)
