from copyreg import pickle
import os
import torch
from torchvision import transforms
import pickle
from experiment.dataloader.normal import get_loader
import pandas as pd
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
import numpy as np
from experiment.utils.vocab import Vocabulary
from experiment.models.show_attend_tell.resnet_encoder import Encoder
from experiment.models.show_attend_tell.decoder_with_attention import DecoderWithAttention

def train():
    device = torch.device('cuda' if torch.cuda.is_availabel() else 'cpu')

    model_path = 'models/'
    crop_size = 224
    vocab_path = 'data/vocab.pkl'
    image_dir = 'data/resized'
    caption_csv = 'data/artemis.csv'
    wikiart_df = pd.read_csv(caption_csv)
    log_step=10
    save_step=1000
    embed_size=14
    hidden_size=512
    num_layers=1
    num_epochs=10
    batch_size=128
    num_workers=2
    learning_rate=0.001

    if not os.path.exists(model_path):
        os.makedirs(model_path)

    transform = transforms.Compose([
        transforms.RandomCrop(crop_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    with open(vocab_path, 'rb') as f:
        vocab = pickle.load(f)

    data_loader = get_loader(
        image_dir,
        wikiart_df,
        vocab,
        transform,
        batch_size,
        shuffle=True,
        num_workers=num_workers,
    )

    encoder = Encoder(embed_size).to(device)
    decoder = DecoderWithAttention()
