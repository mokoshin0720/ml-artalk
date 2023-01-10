import os
from PIL import Image
import torch.utils.data as data
import nltk
import torch
from experiment.utils.vocab import Vocabulary
import pandas as pd
import numpy as np
from pprint import pprint

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
        object_txt = self.object_dir + '/' + str(index) + '.txt'
        mask_txt = self.mask_dir + '/' + str(index) + '.txt'

        filename = wikiart_df.at[index, 'painting'] + '.jpg'
        caption = wikiart_df.at[index, 'utterance']

        image = Image.open(os.path.join(self.root_dir, filename)).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
            
        if os.path.isfile(object_txt):
            with open(object_txt, 'r') as f:
                obj_str = f.read()
                noun_list = eval(obj_str)
                f.close()
            
            with open(mask_txt, 'r') as f:
                print('画像加工処理を加える', flush=True)
                mask_str = f.read()
                print(type(mask_str))
                print(mask_str)
                mask_list = eval(mask_str)
                f.close()
        else:
            print('noun_listを空にする', flush=True)
            noun_list = []
        
        object_list = []
        for noun in noun_list:
            object_list.append(vocab(noun))

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