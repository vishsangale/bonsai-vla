from manim import *

class Patchification(Scene):
    def construct(self):
        # TODO: Implement animation showing an image breaking down into a grid of patches
        title = Text("ViT: Patchification")
        self.play(Write(title))
        self.wait(1)
        self.play(FadeOut(title))
