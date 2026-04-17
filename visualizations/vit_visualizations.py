from manim import *
import numpy as np

class PatchificationScene(Scene):
    def construct(self):
        # 1. Load and display input image
        image = ImageMobject("visualizations/assets/cifar_sample.png")
        image.height = 4
        image.to_edge(LEFT, buff=1)
        
        title = Text("1. Patchification", font_size=36).to_edge(UP)
        self.play(FadeIn(image), Write(title))
        self.wait()

        # 2. Create grid of patches
        grid_size = 4 # 4x4 grid
        patch_width = image.width / grid_size
        patch_height = image.height / grid_size
        
        # Create actual image patches using crop
        patches = Group()
        for i in range(grid_size):
            for j in range(grid_size):
                # Manim's ImageMobject has a pixel-based cropping or we can use multiple ImageMobjects
                # A robust way is to use the same image and set display_pixel_coords
                p = ImageMobject("visualizations/assets/cifar_sample.png")
                p.height = patch_height
                
                # Calculate pixel coordinates for cropping (CIFAR image is 512x512 now)
                # PIL (0,0) is top-left.
                pixel_w = 512 // grid_size
                pixel_h = 512 // grid_size
                left = j * pixel_w
                top = i * pixel_h
                right = left + pixel_w
                bottom = top + pixel_h
                
                # Crop the underlying numpy array
                p.pixel_array = p.pixel_array[top:bottom, left:right]
                
                # Position relative to top-left of image
                p.move_to(image.get_corner(UL) + np.array([j*patch_width + patch_width/2, -i*patch_height - patch_height/2, 0]))
                patches.add(p)
        
        # Create outlines
        outlines = VGroup(*[
            Square(side_length=patch_width).move_to(p.get_center()).set_stroke(WHITE, opacity=0.3)
            for p in patches
        ])
        
        self.play(Create(outlines))
        self.wait()

        # 3. Animate patches moving and FLATTENING to sequence
        sequence = Group()
        for i, p in enumerate(patches):
            # Target position in a sequence
            target = p.copy()
            
            # Non-uniform scaling to simulate flattening into a 1D vector
            # We make it very thin (width=0.2) but keep some height (height=2.0)
            target.stretch_to_fit_width(0.15)
            target.stretch_to_fit_height(2.0)
            
            # Arrange in a single horizontal row
            target.move_to(LEFT * 4 + RIGHT * i * 0.4 + DOWN * 1)
            sequence.add(target)

        # Transformation animation
        self.play(
            LaggedStart(
                *[Transform(patches[i], sequence[i]) for i in range(len(patches))],
                lag_ratio=0.1
            ),
            image.animate.set_opacity(0.1),
            FadeOut(outlines)
        )
        self.wait()

        # Group sequence for easier labeling
        seq_rect = SurroundingRectangle(sequence, color=BLUE, buff=0.2)
        seq_label = Text("Flattened Patch Embeddings (Sequence of 1D Vectors)", font_size=24).next_to(seq_rect, UP)
        
        self.play(Create(seq_rect), Write(seq_label))
        self.wait(3)

class TransformerBlockScene(Scene):
    def construct(self):
        title = Text("2. Transformer Encoder Block (Pre-norm)", font_size=36).to_edge(UP)
        self.add(title)

        # Draw the block container
        rect = RoundedRectangle(height=6, width=4, corner_radius=0.2)
        rect.set_stroke(BLUE, width=2)
        self.play(Create(rect))

        # Internal components
        ln1 = Rectangle(height=0.6, width=2.5, color=GREEN_B).set_fill(GREEN_B, opacity=0.2)
        ln1_text = Text("LayerNorm", font_size=20).move_to(ln1)
        ln1_group = VGroup(ln1, ln1_text).shift(UP * 2)

        msa = Rectangle(height=0.8, width=3, color=ORANGE).set_fill(ORANGE, opacity=0.2)
        msa_text = Text("Multi-Head Attention", font_size=20).move_to(msa)
        msa_group = VGroup(msa, msa_text).next_to(ln1_group, DOWN, buff=0.5)

        ln2 = Rectangle(height=0.6, width=2.5, color=GREEN_B).set_fill(GREEN_B, opacity=0.2)
        ln2_text = Text("LayerNorm", font_size=20).move_to(ln2)
        ln2_group = VGroup(ln2, ln2_text).next_to(msa_group, DOWN, buff=0.8)

        mlp = Rectangle(height=1.2, width=3, color=PURPLE).set_fill(PURPLE, opacity=0.2)
        mlp_text = Text("MLP (Feed Forward)", font_size=20).move_to(mlp)
        mlp_group = VGroup(mlp, mlp_text).next_to(ln2_group, DOWN, buff=0.5)

        # Draw components
        self.play(Create(ln1_group), Create(msa_group), Create(ln2_group), Create(mlp_group))

        # Draw main flow arrows
        arrow1 = Arrow(start=UP*3.5, end=ln1_group.get_top(), buff=0.1)
        arrow2 = Arrow(start=ln1_group.get_bottom(), end=msa_group.get_top(), buff=0.1)
        arrow3 = Arrow(start=msa_group.get_bottom(), end=ln2_group.get_top(), buff=0.1)
        arrow4 = Arrow(start=ln2_group.get_bottom(), end=mlp_group.get_top(), buff=0.1)
        arrow5 = Arrow(start=mlp_group.get_bottom(), end=DOWN*3.5, buff=0.1)

        # Draw Skip Connections
        skip1_path = VGroup(
            Line(UP*3.2, UP*3.2 + LEFT*2),
            Line(UP*3.2 + LEFT*2, msa_group.get_bottom() + DOWN*0.2 + LEFT*2),
            Line(msa_group.get_bottom() + DOWN*0.2 + LEFT*2, msa_group.get_bottom() + DOWN*0.2),
        )
        add1 = Circle(radius=0.2, color=WHITE).move_to(msa_group.get_bottom() + DOWN*0.2)
        add1_text = Text("+", font_size=24).move_to(add1)
        
        skip2_path = VGroup(
            Line(msa_group.get_bottom() + DOWN*0.5, msa_group.get_bottom() + DOWN*0.5 + LEFT*2),
            Line(msa_group.get_bottom() + DOWN*0.5 + LEFT*2, mlp_group.get_bottom() + DOWN*0.2 + LEFT*2),
            Line(mlp_group.get_bottom() + DOWN*0.2 + LEFT*2, mlp_group.get_bottom() + DOWN*0.2),
        )
        add2 = Circle(radius=0.2, color=WHITE).move_to(mlp_group.get_bottom() + DOWN*0.2)
        add2_text = Text("+", font_size=24).move_to(add2)

        self.play(Create(arrow1), Create(arrow2), Create(arrow3), Create(arrow4), Create(arrow5))
        self.play(Create(skip1_path), Create(add1), Write(add1_text))
        self.play(Create(skip2_path), Create(add2), Write(add2_text))
        
        self.wait(3)

class ArchitectureOverview(Scene):
    def construct(self):
        title = Text("Vision Transformer (ViT) Architecture", font_size=36).to_edge(UP)
        self.add(title)

        # Components as boxes
        img_box = Rectangle(height=1.5, width=1.5, color=WHITE).shift(LEFT * 5)
        img_label = Text("Input Image", font_size=18).next_to(img_box, DOWN)
        
        patch_box = Rectangle(height=1.2, width=1.5, color=BLUE).next_to(img_box, RIGHT, buff=0.8)
        patch_label = Text("Patch + Position\nEmbeddings", font_size=16).next_to(patch_box, DOWN)

        trans_stack = VGroup(*[
            Rectangle(height=3, width=1, color=ORANGE).shift(RIGHT * (i*0.2))
            for i in range(3)
        ]).next_to(patch_box, RIGHT, buff=1)
        trans_label = Text("Transformer\nEncoder Stack", font_size=16).next_to(trans_stack, DOWN)

        mlp_head = Rectangle(height=1.5, width=1, color=PURPLE).next_to(trans_stack, RIGHT, buff=1)
        mlp_label = Text("MLP Head", font_size=16).next_to(mlp_head, DOWN)

        output = Text("Prediction", font_size=20, color=YELLOW).next_to(mlp_head, RIGHT, buff=0.8)

        # Arrows
        a1 = Arrow(img_box.get_right(), patch_box.get_left())
        a2 = Arrow(patch_box.get_right(), trans_stack.get_left())
        a3 = Arrow(trans_stack.get_right(), mlp_head.get_left())
        a4 = Arrow(mlp_head.get_right(), output.get_left())

        self.play(Create(img_box), Write(img_label))
        self.play(Create(a1), Create(patch_box), Write(patch_label))
        self.play(Create(a2), Create(trans_stack), Write(trans_label))
        self.play(Create(a3), Create(mlp_head), Write(mlp_label))
        self.play(Create(a4), Write(output))
        
        self.wait(5)
