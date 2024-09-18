from datasets import load_dataset

from training_config import config

config.dataset_name = "jainr3/diffusiondb-pixelart"
dataset = load_dataset(config.dataset_name, "2k_first_1k", split="train")
