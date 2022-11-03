import os
import torch
from torchvision import transforms
from experiment.dataloader.normal import get_loader
import torch.nn as nn
from experiment.utils.vocab import Vocabulary
from experiment.train.config import get_conf, get_model
from experiment.train.normal_loop import loop_normal

def train(model_name):
    conf = get_conf(model_name)
    
    if not os.path.exists(conf['model_path']):
        os.makedirs(conf['model_path'])

    transform = transforms.Compose([ 
        transforms.RandomCrop(conf['crop_size']),
        transforms.RandomHorizontalFlip(), 
        transforms.ToTensor(), 
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    data_loader = get_loader(
        conf['image_dir'],
        conf['wikiart_df'],
        conf['vocab'],
        transform,
        conf['batch_size'],
        shuffle=True,
        num_workers=conf['num_workers'],
    )

    encoder, decoder = get_model(model_name, conf)
    criterion = nn.CrossEntropyLoss()

    encoder_optimizer = torch.optim.Adam(
        params=filter(lambda p: p.requires_grad, encoder.parameters()),
        lr=conf['encoder_lr']
    ) if conf['fine_tune_encoder'] else None
    decoder_optimizer = torch.optim.Adam(
        params=filter(lambda p: p.requires_grad, decoder.parameters()),
        lr=conf['decoder_lr']
    )

    for epoch in range(conf['num_epochs']):
        loop_normal(
            model_name=model_name,
            encoder=encoder,
            decoder=decoder,
            conf=conf,
            data_loader=data_loader,
            criterion=criterion,
            encoder_optimizer=encoder_optimizer,
            decoder_optimizer=decoder_optimizer,
            epoch=epoch,
        )

if __name__ == '__main__':
    train('show_attend_tell')