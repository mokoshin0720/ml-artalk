import torch
import torchvision
from PIL import Image

def get_torch(art_style: str, filename: str) -> torch.tensor:
    BASE_DIR = 'data/wikiart/'
    dir = BASE_DIR + art_style + '/' + filename + '.jpg'

    img = Image.open(dir)
    img_torch = torchvision.transforms.functional.to_tensor(img)

    return img_torch

if __name__ == '__main__':
    art_style = 'Baroque'
    filename = 'adriaen-brouwer_a-boor-asleep'
    img_torch = get_torch(art_style, filename)
    print(img_torch)

