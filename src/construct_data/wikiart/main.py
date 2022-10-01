import torch
import torchvision
from PIL import Image

def get_torch(filename: str) -> torch.tensor:
    img = Image.open(filename)
    img_torch = torchvision.transforms.functional.to_tensor(img)
    return img_torch

if __name__ == '__main__':
    filename = "data/wikiart/Abstract_Expressionism/aaron-siskind_acolman-1-1955.jpg"
    filename = "data/wikiart/Baroque/adriaen-brouwer_a-boor-asleep.jpg"
    img_torch = get_torch(filename)
    print(img_torch)

