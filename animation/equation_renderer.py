import sympy as sp
from manim import *
from utils.latex_formatter import to_latex
# FIX E: removido import morto de expand_multiplications (nunca era usado)
import re
 
 
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
 
            before = step.before
            after = step.after
 
            # FIX F: usa sp.sympify em vez de int() para suportar decimais e frações
            expanded = re.sub(
                r'(\d+(?:\.\d+)?)\((\-?\d+(?:[./]\d+)?)\)',
                lambda m: str(sp.sympify(m.group(1)) * sp.sympify(m.group(2))),
                before
            )
 
            if expanded != before:
                mid_tex = MathTex(to_latex(expanded)).scale(1.4)
                mid_tex.move_to(self.tex)
 
                self.scene.play(ReplacementTransform(self.tex, mid_tex))
                self.tex = mid_tex
                self.scene.wait(1)
 
            # passo normal
            new_tex = MathTex(to_latex(after)).scale(1.4)
            new_tex.move_to(self.tex)
 
            self.scene.play(ReplacementTransform(self.tex, new_tex))
            self.tex = new_tex
 
            if step.explanation:
                explanation = Tex(step.explanation).scale(0.9)
                explanation.next_to(self.tex, DOWN)
 
                self.scene.play(Write(explanation))
                self.scene.wait(1)
 
                self.scene.play(FadeOut(explanation))
 
            self.scene.wait(1)