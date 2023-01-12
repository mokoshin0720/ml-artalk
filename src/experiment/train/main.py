import os
import torch
from experiment.dataloader.get import get_loader
import torch.nn as nn
from experiment.utils.vocab import Vocabulary
from experiment.train.config import get_conf
from experiment.train.utils import get_model
from experiment.dataset.get import get_dataset
from experiment.train.loop import model_loop
from experiment.eval.loop import eval_loop
from notify.logger import notify_success, notify_fail, init_logger
import traceback
import wandb
import datetime
import logging

def train(conf):
    # wandb.init(
    #     project="artalk",
    #     config=conf,
    #     name=conf['model_name']+str(datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9)))),
    # )
    
    train_dataset = get_dataset(conf, is_train=True)
    test_dataset = get_dataset(conf, is_train=False)
    train_dataloader = get_loader(train_dataset, conf)
    test_dataloader = get_loader(test_dataset, conf)
    
    encoder, decoder = get_model(conf)
    
    if not os.path.exists(conf['model_path']):
        os.makedirs(conf['model_path'])
        os.chmod(conf['model_path'],0o777)

    criterion = nn.CrossEntropyLoss()

    encoder_optimizer = torch.optim.Adam(
        params=filter(lambda p: p.requires_grad, encoder.parameters()),
        lr=conf['encoder_lr']
    ) if conf['fine_tune_encoder'] else None
    if encoder_optimizer is not None:
        encoder_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            encoder_optimizer,
            gamma=0.95
        )

    decoder_optimizer = torch.optim.Adam(
        params=filter(lambda p: p.requires_grad, decoder.parameters()),
        lr=conf['decoder_lr']
    )
    decoder_scheduler = torch.optim.lr_scheduler.ExponentialLR(
        decoder_optimizer,
        gamma=0.95
    )
    
    for epoch in range(0, conf['num_epochs']):
        # wandb.log({'epoch': epoch})
        logging.info('train loop...')
        print('train loop...', flush=True)
        model_loop(
            model_name=conf['model_name'],
            encoder=encoder,
            decoder=decoder,
            conf=conf,
            data_loader=train_dataloader,
            criterion=criterion,
            encoder_optimizer=encoder_optimizer,
            decoder_optimizer=decoder_optimizer,
            epoch=epoch,
            is_train=True,
        )
        
        logging.info('validation loop...')
        model_loop(
            model_name=conf['model_name'],
            encoder=encoder,
            decoder=decoder,
            conf=conf,
            data_loader=test_dataloader,
            criterion=criterion,
            encoder_optimizer=encoder_optimizer,
            decoder_optimizer=decoder_optimizer,
            epoch=epoch,
            is_train=False,
        )
        
        if encoder_optimizer is not None: encoder_scheduler.step()
        decoder_scheduler.step()

if __name__ == '__main__':
    # log_filename = init_logger()
    # try:
    #     conf = get_conf()
    #     train(conf)
    # except Exception as e:
    #     traceback.print_exc()
    #     notify_fail(str(e))
    # else:
    #     notify_success(log_filename)

    train(get_conf())