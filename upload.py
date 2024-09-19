from model import model

model.load_config("pixel-diffusion")
print(model)
model.save_pretrained("pixel-diffusion2")
model.save_config("pixel-diffusion2")
