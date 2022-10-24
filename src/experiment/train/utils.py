import torch

def clip_gradient(optimizer, grad_clip):
    for group in optimizer.param_groups:
        for param in group["params"]:
            if param.grad is not None: param.grad.data.clamp_(-grad_clip, grad_clip)

def save_checkpoint(
    data_name,
    epoch,
    epochs_since_improvement,
    encoder,
    decoder,
    encoder_optimizer,
    decoder_optimizer,
    bleu4,
    is_best,
):
    state = {
        'epoch': epoch,
        'epochs_since_improvement': epochs_since_improvement,
        'bleu-4': bleu4,
        'encoder': encoder,
        'decoder': decoder,
        'encoder_optimizer': encoder_optimizer,
        'decoder_optimizer': decoder_optimizer,
    }

    filename = 'checkpoint_' + data_name + '.pth.tar'
    torch.save(state, filename)

    if is_best:
        torch.save(state, 'BEST_' + filename)

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def adjust_learning_rate(optimizer, shrink_factor):
    print('DECAYING learning rate')
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * shrink_factor

    print("The new learning rate is %f" % (optimizer.param_groups[0]['lr']))

def accuracy(scores, targets, k):
    batch_size = targets.size(0)
    _, idx = scores.topk(k, 1, True, True)
    correct = idx.eq(targets.view(-1, 1).expand_as(idx))
    correct_total = correct.view(-1).float().sum()

    return correct_total.item() * (100.0 / batch_size)