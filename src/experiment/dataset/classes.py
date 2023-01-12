import os
from PIL import Image
import torch.utils.data as data
import nltk
import torch
from experiment.utils.vocab import Vocabulary
import pandas as pd
import numpy as np
import cv2

class WikiartDataset(data.Dataset):
    def __init__(self, root_dir, wikiart_df, vocab, transform=None):
        self.root_dir = root_dir
        self.wikiart_df = wikiart_df
        self.vocab = vocab
        self.transform = transform

    def __getitem__(self, index):
        vocab = self.vocab
        wikiart_df = self.wikiart_df

        filename = wikiart_df.at[index, 'painting'] + '.jpg'
        caption = wikiart_df.at[index, 'utterance']

        image = Image.open(os.path.join(self.root_dir, filename)).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        
        tokens = nltk.tokenize.word_tokenize(str(caption).lower())
        caption = []
        caption.append(vocab('<sos>'))
        caption.extend([vocab(token) for token in tokens])
        caption.append(vocab('<eos>'))

        target = torch.Tensor(caption)

        return filename, image, target

    def __len__(self):
        return len(self.wikiart_df)
    
class WikiartDatasetWithObject(data.Dataset):
    def __init__(self, root_dir, wikiart_df, object_dir, mask_dir, vocab, transform=None):
        self.root_dir = root_dir
        self.wikiart_df = wikiart_df
        self.object_dir = object_dir
        self.mask_dir = mask_dir
        self.vocab = vocab
        self.transform = transform

    def __getitem__(self, index):
        vocab = self.vocab
        wikiart_df = self.wikiart_df
        
        id = wikiart_df.at[index, 'id']
        object_txt = self.object_dir + '/' + str(id) + '.txt'
        mask_txt = self.mask_dir + '/' + str(id) + '.txt'

        filename = wikiart_df.at[index, 'painting'] + '.jpg'
        caption = wikiart_df.at[index, 'utterance']

        image = Image.open(os.path.join(self.root_dir, filename)).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
            
        noun_list = np.genfromtxt(object_txt, dtype='str')
        mask_list = np.loadtxt(mask_txt)
        
        trans_img = generate_masked_img(np.array(image), mask_list)
        
        object_list = []
        if noun_list.size == 1:
            object_list = [vocab(str(noun_list))]
        else:
            for i in range(noun_list.size):
                object_list.append(vocab(noun_list[i]))

        tokens = nltk.tokenize.word_tokenize(str(caption).lower())
        caption = []
        caption.append(vocab('<sos>'))
        caption.extend([vocab(token) for token in tokens])
        caption.append(vocab('<eos>'))

        target = torch.Tensor(caption)

        return filename, image, object_list, target

    def __len__(self):
        return len(self.wikiart_df)
    

def calc_disjunction(target_list):
    if len(target_list) == 0 :
        return eval(target_list[0])
    else:
        result = eval(target_list[0])
        
        for target in target_list[1:]:
            result = np.logical_or(result, eval(target))
        return result
    
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