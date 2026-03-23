import sympy as sp
from manim import *
from utils.latex_formatter import to_latex
import re


class EquationRenderer:
    def __init__(self, scene: Scene):
        self.scene = scene
        self.tex = None
        self.explanation_tex = None

    def _format_explanation(self, text: str) -> str:
        """Envolve comandos LaTeX em $...$ para que o Tex() os compile correctamente."""
        return re.sub(
            r'(\\[a-zA-Z]+(?:\{[^}]*\})*(?:\{[^}]*\})?)',
            r'$\1$',
            text
        )

    def animate(self, steps):
        first = to_latex(steps[0].before)
        self.tex = MathTex(first).scale(1.4)
        self.scene.play(Write(self.tex))

        for step in steps:
            before = to_latex(step.before)
            after  = to_latex(step.after)

            # Só anima a transição se houver mudança visível
            if before != after:
                new_tex = MathTex(after).scale(1.4)
                new_tex.move_to(self.tex)
                self.scene.play(ReplacementTransform(self.tex, new_tex))
                self.tex = new_tex
                self.scene.wait(1)

            if step.explanation:
                explanation_text = self._format_explanation(step.explanation)

                if step.explanation == "Vamos verificar!":
                    # Fazer ReplacementTransform para a equação original (after)
                    # antes de mostrar o texto — assim não há flash abrupto
                    eq_tex = MathTex(after).scale(1.4)
                    eq_tex.move_to(self.tex)
                    self.scene.play(ReplacementTransform(self.tex, eq_tex))
                    self.tex = eq_tex
                    self.scene.wait(0.5)
                    # Fazer FadeOut da equação e mostrar só o texto centrado
                    self.scene.play(FadeOut(self.tex))
                    verificar = Tex(explanation_text).scale(1.2)
                    self.scene.play(Write(verificar))
                    self.scene.wait(1.5)
                    self.scene.play(FadeOut(verificar))
                    # Repor a equação original
                    self.scene.play(FadeIn(self.tex))
                    self.scene.wait(0.5)
                else:
                    explanation = Tex(explanation_text).scale(0.9)
                    explanation.next_to(self.tex, DOWN)
                    self.scene.play(Write(explanation))
                    self.scene.wait(1)
                    self.scene.play(FadeOut(explanation))
                    self.scene.wait(1)