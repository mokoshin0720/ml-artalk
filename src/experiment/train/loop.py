import torch
import numpy as np
import os
from torch.nn.utils.rnn import pack_padded_sequence
from experiment.models.cnn_lstm.normal import Encoder, Decoder
from experiment.train.config import loging, saving

def normal_cnn_lstm(
    encoder: Encoder, 
    decoder: Decoder, 
    conf: dict,
    data_loader,
    criterion,
    encoder_optimizer,
    decoder_optimizer,
    epoch,
    ):
    total_step = len(data_loader)
    for i, (images, captions, lengths) in enumerate(data_loader):
        images = images.to(conf['device'])
        captions = captions.to(conf['device'])
        targets = pack_padded_sequence(captions, lengths, batch_first=True)[0]

        features = encoder(images)
        outputs = decoder(features, captions, lengths)

        loss = criterion(outputs, targets)

        decoder.zero_grad()
        encoder.zero_grad()
        loss.backward()
        if encoder_optimizer is not None: encoder_optimizer.step()
        decoder_optimizer.step()

        loging(i, conf, epoch, total_step, loss)
        saving(i, conf, epoch, encoder, decoder)
