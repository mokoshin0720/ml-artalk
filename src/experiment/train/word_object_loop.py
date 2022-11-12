from torch.nn.utils.rnn import pack_padded_sequence
import experiment.models.cnn_lstm.with_word_object as word_object_cnn_lstm
from experiment.train.config import loging, saving

def train_loop(
    model_name: str,
    encoder,
    decoder,
    conf: dict,
    data_loader,
    criterion,
    encoder_optimizer,
    decoder_optimizer,
    epoch,
):
    if model_name == 'cnn_lstm_with_word_object':
        cnn_lstm(encoder, decoder, conf, data_loader, criterion, encoder_optimizer, decoder_optimizer, epoch)
    elif model_name == 'show_attend_tell':
        show_attend_tell(encoder, decoder, conf, data_loader, criterion, encoder_optimizer, decoder_optimizer, epoch)

def cnn_lstm(
    encoder: word_object_cnn_lstm.Encoder, 
    decoder: word_object_cnn_lstm.Decoder, 
    conf: dict,
    data_loader,
    criterion,
    encoder_optimizer,
    decoder_optimizer,
    epoch,
    ):
    encoder.train()
    decoder.train()

    total_step = len(data_loader)

    for i, (images, input_objects, captions, lengths) in enumerate(data_loader):
        images = images.to(conf['device'])
        input_objects = input_objects.to(conf['device'])
        captions = captions.to(conf['device'])
        targets = pack_padded_sequence(captions, lengths, batch_first=True, enforce_sorted=False)[0]

        features = encoder(images, input_objects)
        outputs = decoder(features, captions, lengths)

        loss = criterion(outputs, targets)

        decoder.zero_grad()
        encoder.zero_grad()
        loss.backward()
        if encoder_optimizer is not None: encoder_optimizer.step()
        decoder_optimizer.step()

        loging(i, conf, epoch, total_step, loss)
        saving(i, conf, epoch, encoder, decoder)

def show_attend_tell(
    encoder: word_object_cnn_lstm.Encoder, 
    decoder: word_object_cnn_lstm.Decoder, 
    conf: dict,
    data_loader,
    criterion,
    encoder_optimizer,
    decoder_optimizer,
    epoch,
    ):
    encoder.train()
    decoder.train()

    total_step = len(data_loader)

    for i, (imgs, captions, caplens) in enumerate(data_loader):
        imgs = imgs.to(conf['device'])
        captions = captions.to(conf['device'])
        caplens = caplens.to(conf['device'])

        features = encoder(imgs)
        scores, captions_sorted, decode_lengths, alphas, sort_idx = decoder(features, captions, caplens)

        targets = captions_sorted[:, 1:]

        scores = pack_padded_sequence(scores, decode_lengths, batch_first=True).data
        targets = pack_padded_sequence(targets, decode_lengths, batch_first=True).data

        loss = criterion(scores, targets)

        loss += conf['alpha_c'] * ((1. - alphas.sum(dim=1)) ** 2).mean()

        if encoder_optimizer is not None: encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()
        loss.backward()

        if conf['grad_clip'] is not None:
            if encoder_optimizer is not None: conf['clip_gradient'](encoder_optimizer, conf['grad_clip'])
            conf['clip_gradient'](decoder_optimizer, conf['grad_clip'])
        
        if encoder_optimizer is not None: encoder_optimizer.step()
        decoder_optimizer.step()

        loging(i, conf, epoch, total_step, loss)
        saving(i, conf, epoch, encoder, decoder)