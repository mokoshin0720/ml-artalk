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
        
        # TODO: ☟indexじゃなくて、idである必要がある
        id = wikiart_df.at[index, 'id']
        object_txt = self.object_dir + '/' + str(id) + '.txt'
        mask_txt = self.mask_dir + '/' + str(id) + '.txt'

        filename = wikiart_df.at[index, 'painting'] + '.jpg'
        caption = wikiart_df.at[index, 'utterance']

        image = Image.open(os.path.join(self.root_dir, filename)).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
            
        noun_list = np.genfromtxt(object_txt, dtype='str')
        mask_list = np.loadtxt(mask_txt) # TODO: mask_listを使って使って画像加工を行う
        object_list = []
        if noun_list.size == 1:
            object_list = vocab(str(noun_list))
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