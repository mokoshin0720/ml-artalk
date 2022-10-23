from copyreg import pickle
import os
import torch
from torchvision import transforms
import pickle
from dataloader import get_loader
import pandas as pd
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
import numpy as np
from vocab import Vocabulary
from experiment.models.cnn_lstm.with_word_object import Encoder, Decoder

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model_path = 'models/'
    crop_size = 224
    vocab_path = 'data/vocab.pkl'
    image_dir = 'data/resized'
    caption_csv = 'data/artemis_mini.csv'
    idx2object_df = 'data/idx2object.csv'
    log_step = 5
    save_step = 50
    embed_size = 256
    hidden_size = 512
    num_layers = 1
    num_epochs = 10
    batch_size = 4
    num_workers = 0
    learning_rate = 0.001

    if not os.path.exists(model_path):
        os.makedirs(model_path)

    transform = transforms.Compose([
        transforms.RandomCrop(crop_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    with open(vocab_path, 'rb') as f:
        vocab = pickle.load(f)

    wikiart_df = pd.read_csv(caption_csv)
    idx2object_df = pd.read_csv(idx2object_df)

    data_loader = get_loader(
        image_dir,
        wikiart_df,
        idx2object_df,
        vocab,
        transform,
        batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    encoder = Encoder(len(vocab), embed_size).to(device)
    decoder = Decoder(embed_size, hidden_size, len(vocab), num_layers)

    criterion = nn.CrossEntropyLoss()
    params = list(decoder.parameters()) + list(encoder.linear.parameters()) + list(encoder.bn.parameters())
    optimizer = torch.optim.Adam(params, lr=learning_rate)

    total_step = len(data_loader)

    for epoch in range(num_epochs):
        for i, (images, input_objects, captions, lengths) in enumerate(data_loader):
            images = images.to(device)
            input_objects = input_objects.to(device)
            captions = captions.to(device)
            targets = pack_padded_sequence(captions, lengths, batch_first=True, enforce_sorted=False)[0]

            features = encoder.forward(images, input_objects)
            outputs = decoder.forward(features, captions, lengths)
            loss = criterion(outputs, targets)

            decoder.zero_grad()
            encoder.zero_grad()
            loss.backward()
            optimizer.step()

            if i % log_step == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Perplexity: {:5.4f}'
                    .format(epoch, num_epochs, i, total_step, loss.item(), np.exp(loss.item()))) 
    
            if (i+1) % save_step == 0:
                torch.save(decoder.state_dict(), os.path.join(
                    model_path, 'decoder-{}-{}.ckpt'.format(epoch+1, i+1)))
                torch.save(encoder.state_dict(), os.path.join(
                    model_path, 'encoder-{}-{}.ckpt'.format(epoch+1, i+1)))