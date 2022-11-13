import os
from PIL import Image
import torch.utils.data as data
import nltk
import torch
from experiment.utils.vocab import Vocabulary

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

        return image, target

    def __len__(self):
        return len(self.wikiart_df)
    
class WikiartDatasetWithObject(data.Dataset):
    def __init__(self, root_dir, wikiart_df, idx2object_df, vocab, transform=None):
        self.root_dir = root_dir
        self.wikiart_df = wikiart_df
        self.idx2object_df = idx2object_df
        self.vocab = vocab
        self.transform = transform

    def __getitem__(self, index):
        vocab = self.vocab
        wikiart_df = self.wikiart_df
        idx2object_df = self.idx2object_df

        filename = wikiart_df.at[index, 'painting'] + '.jpg'
        caption = wikiart_df.at[index, 'utterance']

        image = Image.open(os.path.join(self.root_dir, filename)).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        
        noun_list = list(idx2object_df[idx2object_df['sentence_id'] == index+1]['object'])
        object_list = []
        for noun in noun_list:
            object_list.append(vocab(noun))

        tokens = nltk.tokenize.word_tokenize(str(caption).lower())
        caption = []
        caption.append(vocab('<sos>'))
        caption.extend([vocab(token) for token in tokens])
        caption.append(vocab('<eos>'))

        target = torch.Tensor(caption)

        return image, object_list, target

    def __len__(self):
        return len(self.wikiart_df)
