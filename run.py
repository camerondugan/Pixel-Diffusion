from diffusers import DiffusionPipeline

pipeline = DiffusionPipeline.from_pretrained("CameronDugan/PixelDiffusion").to("cuda")

image = pipeline().images[0]
image.save("example.png")
