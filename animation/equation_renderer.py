import sympy as sp
from manim import *
from utils.latex_formatter import to_latex
import re


class EquationRenderer:
    def __init__(self, scene: Scene):
        self.scene = scene
        self.tex = None

    def _format_explanation(self, text: str) -> str:
        """Envolve comandos LaTeX em $...$ para compilação correcta."""
        return re.sub(
            r'(\\[a-zA-Z]+(?:\{[^}]*\})*(?:\{[^}]*\})?)',
            r'$\1$',
            text
        )

    def _make_tex(self, latex_str: str) -> MathTex:
        """Cria um MathTex redimensionado para caber no ecrã."""
        tex = MathTex(latex_str)
        # Escalar para caber na largura do ecrã com margem
        max_width  = config.frame_width  * 0.9
        max_height = config.frame_height * 0.25
        scale = min(
            max_width  / tex.width  if tex.width  > 0 else 1,
            max_height / tex.height if tex.height > 0 else 1,
            1.4  # escala máxima
        )
        return tex.scale(scale)

    def animate(self, steps):
        first = to_latex(steps[0].before)
        self.tex = self._make_tex(first)
        self.scene.play(Write(self.tex))

        for step in steps:
            before = to_latex(step.before)
            after  = to_latex(step.after)

            # Só anima a transição se houver mudança visível
            if before != after:
                new_tex = self._make_tex(after)
                new_tex.move_to(self.tex)
                self.scene.play(ReplacementTransform(self.tex, new_tex))
                self.tex = new_tex
                self.scene.wait(1)

            if step.explanation:
                explanation_text = self._format_explanation(step.explanation)

                if step.explanation == "Vamos verificar!":
                    # Transição para equação original antes do texto centrado
                    eq_tex = self._make_tex(after)
                    eq_tex.move_to(self.tex)
                    self.scene.play(ReplacementTransform(self.tex, eq_tex))
                    self.tex = eq_tex
                    self.scene.wait(0.5)
                    self.scene.play(FadeOut(self.tex))
                    verificar = Tex(explanation_text).scale(1.2)
                    self.scene.play(Write(verificar))
                    self.scene.wait(1.5)
                    self.scene.play(FadeOut(verificar))
                    self.scene.play(FadeIn(self.tex))
                    self.scene.wait(0.5)
                else:
                    explanation = Tex(explanation_text).scale(0.9)
                    explanation.next_to(self.tex, DOWN)
                    self.scene.play(Write(explanation))
                    self.scene.wait(1)
                    self.scene.play(FadeOut(explanation))
                    self.scene.wait(1)
                    