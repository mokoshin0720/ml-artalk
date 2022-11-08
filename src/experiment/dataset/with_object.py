import os
from torchvision import transforms
from PIL import Image
import torch.utils.data as data
import nltk
import torch
import pandas as pd
import pickle
from experiment.utils.vocab import Vocabulary

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
        caption.append(vocab('<start>'))
        caption.extend([vocab(token) for token in tokens])
        caption.append(vocab('<end>'))

        target = torch.Tensor(caption)

        return image, object_list, target

    def __len__(self):
        return len(self.wikiart_df)

def get_dataset(conf: dict, is_train: bool):
    transform = transforms.Compose([ 
        transforms.RandomCrop(conf['crop_size']),
        transforms.RandomHorizontalFlip(), 
        transforms.ToTensor(), 
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    wikiart_df = conf['train_df'] if is_train else conf['test_df']

    return WikiartDatasetWithObject(
            root_dir=conf['image_dir'],
            wikiart_df=wikiart_df,
            idx2object_df=conf['idx2obj_df'],
            vocab=conf['vocab'],
            transform=transform
        )

    def __len__(self):
        return len(self.wikiart_df)

if __name__ == '__main__':
    nltk.download('punkt')

    with open('data/vocab.pkl', 'rb') as f:
        vocab = pickle.load(f)

    wikiart_df = pd.read_csv('data/artemis_mini.csv')
    idx2object_df = pd.read_csv('data/idx2object.csv')

    dataset = WikiartDatasetWithObject(
        'data/resized/',
        wikiart_df,
        idx2object_df,
        vocab,
    )

    for i in range(len(dataset)):
        img, input_object, caption = dataset[i]
        pprint(img)
        pprint(input_object)
        pprint(caption)
