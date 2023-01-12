import cv2
import numpy as np
from PIL import Image
import pandas as pd
import os

def generate_masked_img(original_img, mask_pos):
    mask_pos = np.logical_not(mask_pos)
    
    blur_img = cv2.blur(original_img, ksize=(10, 10))
    img_height, img_width, _ = original_img.shape

    mask_i = Image.fromarray(np.uint8(mask_pos))
    mask_array = np.asarray(mask_i.resize((img_width, img_height)))

    indices = np.where(mask_array)

    for i in range(len(indices[0])):
        w = indices[0][i]
        h = indices[1][i]

        try:
            original_img[w][h] = blur_img[w][h]
        except Exception as e:
            continue

    return original_img

if __name__ == '__main__':
    original_dir = 'data/resized/'
    mask_dir = 'data/image_info/train/mask/'
    
    train_df = pd.read_csv('data/train.csv')
    mask_files = os.listdir(mask_dir)
    
    for idx, row in train_df.iterrows():
        if idx == 5: break
        
        id = row['id']
        filename = row['painting']
        img = Image.open(os.path.join(original_dir + filename + '.jpg'))
        img.save('origin.png')
        
        mask_pos = np.loadtxt(mask_dir + str(id) + '.txt')
        
        trans_img = generate_masked_img(np.array(img), mask_pos)
        
        i = Image.fromarray(trans_img)
        i.save('blur.png')