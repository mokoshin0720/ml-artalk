from base64 import decode
import torch
import torch.nn as nn
import pandas as pd
from torchvision import transforms
import pickle
from experiment.utils.vocab import Vocabulary
from experiment.models.show_attend_tell.decoder_with_attention import DecoderWithAttention
from experiment.models.show_attend_tell.resnet_encoder import Encoder
from experiment.dataloader.normal import get_loader
from experiment.train.utils import adjust_learning_rate
from experiment.train.show_attend_tell.train import train

def main():
    vocab_path = 'data/vocab.pkl'
    image_dir ='data/resized'
    caption_csv = 'data/artemis_dataset.csv'
    crop_size = 224
    wikiart_df = pd.read_csv(caption_csv)
    

    embed_dim = 512
    attention_dim = 512
    decoder_dim = 512
    dropout = 0.5
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    # cudnn.benchmark = True

    start_epoch = 0
    epochs = 120
    epochs_since_improvement = 0
    batch_size = 32
    encoder_lr = 1e-4
    decoder_lr = 4e-4
    grad_clip = 5.
    alpha_c = 1.
    best_bleu4 = 0.
    print_freq = 100
    num_workers=0
    fine_tune_encoder = False

    with open(vocab_path, 'rb') as f:
        vocab = pickle.load(f)

    encoder = Encoder()
    encoder.fine_tune(fine_tune_encoder)
    encoder_optimizer = torch.optim.Adam(
        params=filter(lambda p: p.requires_grad, encoder.parameters()),
        lr=encoder_lr
    ) if fine_tune_encoder else None

    decoder = DecoderWithAttention(
        attention_dim=attention_dim,
        embed_dim=embed_dim,
        decoder_dim=decoder_dim,
        vocab_size=len(vocab),
        dropout=dropout,
    )
    decoder_optimizer = torch.optim.Adam(
        params=filter(lambda p: p.requires_grad, decoder.parameters()),
        lr=decoder_lr
    )

    encoder.to(device)
    decoder.to(device)

    criterion = nn.CrossEntropyLoss().to(device)
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

    for epoch in range(start_epoch, epochs):
        if epochs_since_improvement == 20: break
        if epochs_since_improvement > 0 and epochs_since_improvement % 8 == 0:
            if fine_tune_encoder: adjust_learning_rate(encoder_optimizer, 0.8)
            adjust_learning_rate(decoder_optimizer, 0.8)

        train(
            data_loader=data_loader,
            encoder=encoder,
            decoder=decoder,
            criterion=criterion,
            encoder_optimizer=encoder_optimizer,
            decoder_optimizer=decoder_optimizer,
            epoch=epoch,
            alpha_c=alpha_c,
            grad_clip=grad_clip,
            print_freq=print_freq,
            device=device,
        )

if __name__ == '__main__':
    main()