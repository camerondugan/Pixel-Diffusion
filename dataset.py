from datasets import load_dataset
from torchvision import transforms

from training_config import config

config.dataset_name = "jainr3/diffusiondb-pixelart"
dataset = load_dataset(config.dataset_name, "2k_first_1k", split="train")
preprocess = transforms.Compose(
    [
        transforms.Resize((config.image_size, config.image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ]
)


def transform(examples):
    images = [preprocess(image.convert("RGB")) for image in examples["image"]]
    return {"images": images}


dataset.set_transform(transform)
