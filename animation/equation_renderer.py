import sympy as sp
from manim import *
from utils.latex_formatter import to_latex
import re


class EquationRenderer:
    def __init__(self, scene: Scene):
        self.scene = scene
        self.tex = None

    def _format_explanation(self, text: str) -> str:
        """Envolve comandos LaTeX em $...$ para compilação correcta pelo Tex()."""
        return re.sub(
            r'(\\[a-zA-Z]+(?:\{[^}]*\})*(?:\{[^}]*\})?)',
            r'$\1$',
            text
        )

    def _make_tex(self, latex_str: str) -> MathTex:
        """Cria um MathTex redimensionado para caber no ecrã."""
        latex_str = re.sub(r'  +', ' ', latex_str)
        tex = MathTex(latex_str)
        max_width  = config.frame_width  * 0.9
        max_height = config.frame_height * 0.25
        scale = min(
            max_width  / tex.width  if tex.width  > 0 else 1,
            max_height / tex.height if tex.height > 0 else 1,
            1.4
        )
        return tex.scale(scale)

    def _make_explanation(self, text: str) -> Mobject:
        """
        Cria o mobject de explicação adequado consoante o conteúdo.

        Regras:
          - Só LaTeX puro (ex: 'x = \\frac{24}{5} \\approx 4.8') → MathTex
          - Texto misto com LaTeX embutido (ex: 'Substituir x por \\frac{24}{5}')
            → Tex com _format_explanation (envolve LaTeX em $...$)
          - Texto simples → Tex
        """
        # Detectar se é LaTeX puro: começa com \, x = ..., ou só tem LaTeX
        pure_latex = bool(re.match(r'^[x\s\\$=\d\{\}\^_\+\-\*/\.]+$', text.strip()))

        if pure_latex and re.search(r'\\approx|\\frac', text):
            # LaTeX puro — usar MathTex
            return self._make_tex(text).scale(0.7)
        elif re.search(r'\\[a-zA-Z]', text):
            # Texto misto com comandos LaTeX — usar Tex com $...$
            formatted = self._format_explanation(text)
            return Tex(formatted).scale(0.9)
        else:
            # Texto simples
            return Tex(text).scale(0.9)

    def animate(self, steps):
        first = to_latex(steps[0].before)
        self.tex = self._make_tex(first)
        self.scene.play(Write(self.tex))

        for step in steps:
            before = to_latex(step.before)
            after  = to_latex(step.after)

            if before != after:
                new_tex = self._make_tex(after)
                new_tex.move_to(self.tex)
                self.scene.play(ReplacementTransform(self.tex, new_tex))
                self.tex = new_tex
                self.scene.wait(1)

            if step.explanation:
                if step.explanation == "Vamos verificar!":
                    eq_tex = self._make_tex(after)
                    eq_tex.move_to(self.tex)
                    self.scene.play(ReplacementTransform(self.tex, eq_tex))
                    self.tex = eq_tex
                    self.scene.wait(0.5)
                    self.scene.play(FadeOut(self.tex))
                    verificar = Tex(step.explanation).scale(1.2)
                    self.scene.play(Write(verificar))
                    self.scene.wait(1.5)
                    self.scene.play(FadeOut(verificar))
                    self.scene.play(FadeIn(self.tex))
                    self.scene.wait(0.5)
                else:
                    explanation = self._make_explanation(step.explanation)
                    explanation.next_to(self.tex, DOWN)
                    self.scene.play(Write(explanation))
                    self.scene.wait(1.5)
                    self.scene.play(FadeOut(explanation))
                    self.scene.wait(1)

                    