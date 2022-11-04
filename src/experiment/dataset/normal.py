import os
from torchvision import transforms
from pprint import pprint
from PIL import Image
import torch.utils.data as data
import nltk
import torch
import pandas as pd
import pickle
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
        caption.append(vocab('<start>'))
        caption.extend([vocab(token) for token in tokens])
        caption.append(vocab('<end>'))

        target = torch.Tensor(caption)

        return image, target

    def __len__(self):
        return len(self.wikiart_df)

def get_dataset(conf: dict, is_train: bool):
    transform = transforms.Compose([ 
        transforms.RandomCrop(conf['crop_size']),
        transforms.RandomHorizontalFlip(), 
        transforms.ToTensor(), 
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    wikiart_df = conf['train_df'] if is_train else conf['test_df']

    return WikiartDataset(
            root_dir=conf['image_dir'],
            wikiart_df=wikiart_df,
            vocab=conf['vocab'],
            transform=transform
        )

if __name__ == '__main__':
    nltk.download('punkt')

    with open('data/vocab.pkl', 'rb') as f:
        vocab = pickle.load(f)

    wikiart_df = pd.read_csv('data/artemis_mini.csv')

    dataset = WikiartDataset(
        'data/resized/',
        wikiart_df,
        vocab,
    )

    for i in range(len(dataset)):
        img, caption = dataset[i]
        pprint(img)
        pprint(caption)