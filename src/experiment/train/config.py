import torch
import pandas as pd
import numpy as np
import os
from experiment.models.cnn_lstm.normal import Encoder, Decoder
import pickle
from experiment.utils.vocab import Vocabulary

def get_conf(model_name):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    caption_csv = 'data/artemis_mini.csv'
    idx2obj_csv = 'data/idx2object.csv'

    with open('data/vocab.pkl', 'rb') as f:
        vocab = pickle.load(f)

    return {
        # dataset
        'device': device,
        'model_path': 'models/' + model_name + '/',
        'image_dir': 'data/resized',
        'vocab': vocab,
        'wikiart_df': pd.read_csv(caption_csv),
        'idx2obj_df': pd.read_csv(idx2obj_csv),

        # step
        'log_step': 5,
        'save_step': 50,

        # cnn-lstm
        'embed_size': 256,
        'hidden_size': 512,

        # 

        # train
        'crop_size': 224,
        'num_layers': 1,
        'num_epochs': 10,
        'batch_size': 4,
        'num_workers': 0,
        'fine_tune_encoder': False,
        'encoder_lr': 1e-4,
        'decoder_lr': 1e-4,
    }

def get_model(model_name, conf):
    if model_name == 'cnn_lstm':
        encoder = Encoder(conf['embed_size']).to(conf['device'])
        decoder = Decoder(conf['embed_size'], conf['hidden_size'], len(conf['vocab']), conf['num_layers']).to(conf['device'])
        return encoder, decoder

def loging(i: int, conf: dict, epoch: int, total_step: int, loss):
    if i % conf['log_step'] == 0:
        print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Perplexity: {:5.4f}'
                .format(epoch, conf['num_epochs'], i, total_step, loss.item(), np.exp(loss.item()))) 

def saving(i: int, conf: dict, epoch, encoder, decoder):
    if (i+1) % conf['save_step'] == 0:
            torch.save(encoder.state_dict(), os.path.join(
                conf['model_path'], 'encoder-{}-{}.ckpt'.format(epoch+1, i+1)))
            torch.save(decoder.state_dict(), os.path.join(
                conf['model_path'], 'decoder-{}-{}.ckpt'.format(epoch+1, i+1)))