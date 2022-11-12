import pickle
import pandas as pd
from torchvision import transforms
import experiment.dataloader.normal as normal_loader
import experiment.dataloader.with_object as with_object_loader
from experiment.dataset.normal import get_dataset

def get_normal_input_data(conf: dict):
    dataset = get_dataset(conf, is_train=True)
    
    data_loader = normal_loader.get_loader(
        dataset=dataset,
        batch_size=conf['batch_size'],
        shuffle=conf['shuffle'],
        num_workers=conf['num_workers']
    )

    for images, captions, lengths in data_loader:
        return images, captions, lengths

def get_data_with_object():
    image_dir = 'data/resized'
    caption_csv = 'data/artemis_dataset.csv'
    vocab_path = 'data/vocab.pkl'
    wikiart_df = pd.read_csv(caption_csv)
    idx2object_df = 'data/idx2object.csv'
    idx2object_df = pd.read_csv(idx2object_df)
    batch_size = 4
    num_workers = 0
    crop_size = 224

    with open(vocab_path, 'rb') as f:
        vocab = pickle.load(f)

    transform = transforms.Compose([
        transforms.RandomCrop(crop_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    data_loader = with_object_loader.get_loader(
        image_dir,
        wikiart_df,
        idx2object_df,
        vocab,
        transform,
        batch_size,
        shuffle=True,
        num_workers=num_workers,
    )

    for images, input_objects, captions, lengths in data_loader:
        return images, input_objects, captions, lengths

def get_vocab_size():
    vocab_path = 'data/vocab.pkl'

    with open(vocab_path, 'rb') as f:
        vocab = pickle.load(f)
    return len(vocab)