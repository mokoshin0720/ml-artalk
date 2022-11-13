import os
import torch
from experiment.dataloader.normal import get_loader
import torch.nn as nn
from experiment.utils.vocab import Vocabulary
from experiment.train.config import get_conf, get_model
from experiment.dataset.normal import get_dataset
from experiment.train.loop import train_loop
from notify.logger import notify_success, notify_fail, init_logger

def train(model_name, dataset):
    conf = get_conf(model_name)
    data_loader = get_loader(dataset, conf['batch_size'], conf['shuffle'], conf['num_workers'])
    
    if not os.path.exists(conf['model_path']):
        os.makedirs(conf['model_path'])

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

    for epoch in range(1, conf['num_epochs']):
        train_loop(
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
    log_filename = init_logger()
    try:
        # model_name = 'cnn_lstm'
        model_name = 'cnn_lstm_with_object'
        # model_name = 'show_attend_tell'
        conf = get_conf(model_name)
        dataset = get_dataset(conf, is_train=True)
        train(model_name, dataset)
    except Exception as e:
        notify_fail(str(e))
    else:
        notify_success(log_filename)

