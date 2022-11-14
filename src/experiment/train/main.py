import shutil
import os
import torch
from experiment.dataloader.get import get_loader
import torch.nn as nn
from experiment.utils.vocab import Vocabulary
from experiment.train.config import get_conf
from experiment.train.utils import get_model
from experiment.dataset.get import get_dataset
from experiment.train.loop import train_loop
from notify.logger import notify_success, notify_fail, init_logger
import traceback

def train(conf):
    dataset = get_dataset(conf, is_train=True)
    data_loader = get_loader(dataset, conf)
    encoder, decoder = get_model(conf)
    
    if not os.path.exists(conf['model_path']):
        os.makedirs(conf['model_path'])

    criterion = nn.CrossEntropyLoss()

    encoder_optimizer = torch.optim.Adam(
        params=filter(lambda p: p.requires_grad, encoder.parameters()),
        lr=conf['encoder_lr']
    ) if conf['fine_tune_encoder'] else None
    decoder_optimizer = torch.optim.Adam(
        params=filter(lambda p: p.requires_grad, decoder.parameters()),
        lr=conf['decoder_lr']
    )

    for epoch in range(1, conf['num_epochs']):
        train_loop(
            model_name=conf['model_name'],
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
    log_filename = init_logger()
    try:
        conf = get_conf()
        train(conf)
    except Exception as e:
        traceback.print_exc()
        notify_fail(str(e))
    else:
        notify_success(log_filename)

