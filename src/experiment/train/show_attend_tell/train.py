import time
from torch.nn.utils.rnn import pack_padded_sequence
from experiment.train.utils import AverageMeter, clip_gradient, accuracy

def train(
    data_loader, 
    encoder, 
    decoder, 
    criterion, 
    encoder_optimizer, 
    decoder_optimizer, 
    epoch, 
    alpha_c, 
    grad_clip, 
    print_freq, 
    device):
    encoder.train()
    decoder.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top5accs = AverageMeter()

    start = time.time()

    for i, (imgs, captions, caplens) in enumerate(data_loader):
        data_time.update(time.time() - start)

        imgs = imgs.to(device)
        captions = captions.to(device)
        caplens = caplens.to(device)

        features = encoder(imgs)
        scores, captions_sorted, decode_lengths, alphas, sort_idx = decoder(features, captions, caplens)

        targets = captions_sorted[:, 1:]

        scores = pack_padded_sequence(scores, decode_lengths, batch_first=True).data
        targets = pack_padded_sequence(targets, decode_lengths, batch_first=True).data

        loss = criterion(scores, targets)

        loss += alpha_c * ((1. - alphas.sum(dim=1)) ** 2).mean()

        if encoder_optimizer is not None: encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()
        loss.backward()

        if grad_clip is not None:
            if encoder_optimizer is not None: clip_gradient(encoder_optimizer, grad_clip)
            clip_gradient(decoder_optimizer, grad_clip)
        
        if encoder_optimizer is not None: encoder_optimizer.step()
        decoder_optimizer.step()

        top5 = accuracy(scores, targets, 5)
        top5accs.update(top5, sum(decode_lengths))
        losses.update(loss.item(), sum(decode_lengths))
        batch_time.update(time.time() - start)

        start = time.time()

        if i % print_freq == 0:
            print(
                'Epoch: [{0}][{1}/{2}]\t'
                'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                'Data Load Time {data_time.val:.3f} ({data_time.avg:.3f})\t'
                'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                'Top-5 Accuracy {top5.val:.3f} ({top5.avg:.3f})'.format(epoch, i, len(data_loader),
                                                                        batch_time=batch_time,
                                                                        data_time=data_time, loss=losses,
                                                                        top5=top5accs))