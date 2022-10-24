import pickle
import pandas as pd
from torchvision import transforms
from experiment.dataloader.normal import get_loader

def get_normal_input_data():
    num_workers=0
    crop_size = 224
    vocab_path = 'data/vocab.pkl'
    image_dir ='data/resized'
    caption_csv = 'data/artemis_dataset.csv'
    wikiart_df = pd.read_csv(caption_csv)
    batch_size = 4

    with open(vocab_path, 'rb') as f:
        vocab = pickle.load(f)

    transform = transforms.Compose([ 
        transforms.RandomCrop(crop_size),
        transforms.RandomHorizontalFlip(), 
        transforms.ToTensor(), 
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    data_loader = get_loader(
        image_dir,
        wikiart_df,
        vocab,
        transform,
        batch_size,
        shuffle=True,
        num_workers=num_workers,
    )

    for _, (images, captions, lengths) in enumerate(data_loader):
        return images, captions, lengths

def get_vocab_size():
    vocab_path = 'data/vocab.pkl'

    with open(vocab_path, 'rb') as f:
        vocab = pickle.load(f)
    return len(vocab)