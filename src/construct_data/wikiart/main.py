import torchvision
from PIL import Image

if __name__ == '__main__':
    filename = "data/wikiart/Abstract_Expressionism/aaron-siskind_acolman-1-1955.jpg"
    filename = "data/wikiart/Baroque/adriaen-brouwer_a-boor-asleep.jpg"
    img = Image.open(filename)
    img_torch = torchvision.transforms.functional.to_tensor(img)
    print(img_torch)

