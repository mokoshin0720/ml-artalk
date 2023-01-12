from torch.nn.utils.rnn import pack_padded_sequence
import experiment.models.cnn_lstm.normal as normal_cnn_lstm
import experiment.models.cnn_lstm.with_word_object as word_object_cnn_lstm
import experiment.models.show_attend_tell.resnet_encoder as sat_encoder
import experiment.models.show_attend_tell.decoder_with_attention as sat_decoder
import experiment.models.show_attend_tell.with_word_object as word_object_sat_decoder
from experiment.train.utils import loging, saving, plotting
import statistics

def model_loop(
    model_name: str,
    encoder,
    decoder,
    conf: dict,
    data_loader,
    criterion,
    encoder_optimizer,
    decoder_optimizer,
    epoch,
    is_train,
):
    if model_name == 'cnn_lstm':
        cnn_lstm(encoder, decoder, conf, data_loader, criterion, encoder_optimizer, decoder_optimizer, epoch, is_train)
    if model_name == 'cnn_lstm_with_object':
        cnn_lstm_with_object(encoder, decoder, conf, data_loader, criterion, encoder_optimizer, decoder_optimizer, epoch, is_train)
    elif model_name == 'show_attend_tell':
        show_attend_tell(encoder, decoder, conf, data_loader, criterion, encoder_optimizer, decoder_optimizer, epoch, is_train)
    elif model_name == 'show_attend_tell_with_object':
        show_attend_tell_with_object(encoder, decoder, conf, data_loader, criterion, encoder_optimizer, decoder_optimizer, epoch, is_train)

def cnn_lstm(
    encoder: normal_cnn_lstm.Encoder, 
    decoder: normal_cnn_lstm.Decoder, 
    conf: dict,
    data_loader,
    criterion,
    encoder_optimizer,
    decoder_optimizer,
    epoch,
    is_train,
    ):
    if is_train:
        encoder.train()
        decoder.train()
    else:
        encoder.eval()
        decoder.eval()

    losses = []
    for i, (filenames, images, captions, lengths) in enumerate(data_loader):
        images = images.to(conf['device'])
        captions = captions.to(conf['device'])
        targets = pack_padded_sequence(captions, lengths, batch_first=True, enforce_sorted=False)[0]

        features = encoder(images)
        outputs = decoder(features, captions, lengths)
        
        loss = criterion(outputs, targets)
        losses.append(loss.item())

        if is_train:
            decoder_optimizer.zero_grad()
            if encoder_optimizer is not None: encoder_optimizer.zero_grad()

            loss.backward()

            if encoder_optimizer is not None: encoder_optimizer.step()
            decoder_optimizer.step()

        loging(i, conf, epoch, len(data_loader), loss)

    if is_train: saving(conf, epoch, encoder, decoder)
    plotting(statistics.mean(losses), is_train)

def cnn_lstm_with_object(
    encoder: word_object_cnn_lstm.Encoder, 
    decoder: word_object_cnn_lstm.Decoder, 
    conf: dict,
    data_loader,
    criterion,
    encoder_optimizer,
    decoder_optimizer,
    epoch,
    is_train,
    ):
    if is_train:
        encoder.train()
        decoder.train()
    else:
        encoder.eval()
        decoder.eval()

    losses = []
    for i, (filenames, images, input_objects, captions, lengths) in enumerate(data_loader):
        images = images.to(conf['device'])
        input_objects = input_objects.to(conf['device'])
        captions = captions.to(conf['device'])
        targets = pack_padded_sequence(captions, lengths, batch_first=True, enforce_sorted=False)[0]

        features = encoder.forward(images, input_objects)
        outputs = decoder.forward(features, captions, lengths)
        
        loss = criterion(outputs, targets)
        losses.append(loss)

        if is_train:
            decoder_optimizer.zero_grad()
            encoder_optimizer.zero_grad()
            loss.backward()
            if encoder_optimizer is not None: encoder_optimizer.step()
            decoder_optimizer.step()

        loging(i, conf, epoch, len(data_loader), loss)

    if is_train: saving(i, conf, epoch, encoder, decoder)
    plotting(statistics.mean(losses), is_train)

def show_attend_tell(
    encoder: sat_encoder.Encoder, 
    decoder: sat_decoder.DecoderWithAttention, 
    conf: dict,
    data_loader,
    criterion,
    encoder_optimizer,
    decoder_optimizer,
    epoch,
    is_train,
    ):
    if is_train:
        encoder.train()
        decoder.train()
    else:
        encoder.eval()
        decoder.eval()

    losses = []
    for i, (filenames, imgs, captions, caplens) in enumerate(data_loader):
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
        losses.append(loss.item())
        
        if is_train:
            if encoder_optimizer is not None: encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()
            
            loss.backward()

            # if conf['grad_clip'] is not None:
            #     if encoder_optimizer is not None: conf['clip_gradient'](encoder_optimizer, conf['grad_clip'])
            #     conf['clip_gradient'](decoder_optimizer, conf['grad_clip'])
        
            if encoder_optimizer is not None: encoder_optimizer.step()
            decoder_optimizer.step()

        loging(i, conf, epoch, len(data_loader), loss)
        
    if is_train: saving(conf, epoch, encoder, decoder)
    plotting(statistics.mean(losses), is_train)
    
def show_attend_tell_with_object(
    encoder: sat_encoder.Encoder, 
    decoder: word_object_sat_decoder.DecoderWithAttention, 
    conf: dict,
    data_loader,
    criterion,
    encoder_optimizer,
    decoder_optimizer,
    epoch,
    is_train,
    ):
    if is_train:
        encoder.train()
        decoder.train()
    else:
        encoder.eval()
        decoder.eval()

    losses = []
    for i, (filenames, imgs, input_objects, captions, caplens) in enumerate(data_loader):
        imgs = imgs.to(conf['device'])
        input_objects = input_objects.to(conf['device'])
        captions = captions.to(conf['device'])
        caplens = caplens.to(conf['device'])

        features = encoder(imgs)
        scores, captions_sorted, decode_lengths, alphas, sort_idx = decoder(features, captions, caplens, input_objects)

        targets = captions_sorted[:, 1:]
        
        scores = pack_padded_sequence(scores, decode_lengths, batch_first=True).data
        targets = pack_padded_sequence(targets, decode_lengths, batch_first=True).data
        
        loss = criterion(scores, targets)
        loss += conf['alpha_c'] * ((1. - alphas.sum(dim=1)) ** 2).mean()
        losses.append(loss.item())
        
        if is_train:
            if encoder_optimizer is not None: encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()
            
            loss.backward()

            # if conf['grad_clip'] is not None:
            #     if encoder_optimizer is not None: conf['clip_gradient'](encoder_optimizer, conf['grad_clip'])
            #     conf['clip_gradient'](decoder_optimizer, conf['grad_clip'])
        
            if encoder_optimizer is not None: encoder_optimizer.step()
            decoder_optimizer.step()

        loging(i, conf, epoch, len(data_loader), loss)
        
    if is_train: saving(conf, epoch, encoder, decoder)
    plotting(statistics.mean(losses), is_train)
    