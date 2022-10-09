from genericpath import isfile
from PIL import Image
import os

from sklearn.multiclass import OutputCodeClassifier

def resize_image(image, size):
    return image.resize((size, size), Image.ANTIALIAS)

def resize_images(image_dir, output_dir, size):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    images = os.listdir(image_dir)
    num_images = len(images)

    for i, image in enumerate(images):
        with open(os.path.join(image_dir, image), 'r+b') as f:
            with Image.open(f) as img:
                img = resize_image(img, size)
                img.save(os.path.join(output_dir, image), img.format)

        if (i+1) % 100 == 0:
            print("[{}/{}] Resized the images and saved into '{}'.".format(i+1, num_images, output_dir))

if __name__ == '__main__':
    wikiart_dir = 'data/wikiart/'
    output_dir = 'data/resized/'
    image_size = 256

    for name in os.listdir(wikiart_dir):
        path = wikiart_dir + name
        if os.path.isfile(path): continue
        resize_images(path, output_dir, image_size)