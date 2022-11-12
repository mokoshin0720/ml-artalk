import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import os
import experiment.models.cnn_lstm.normal as normal_cnn_lstm
import experiment.models.cnn_lstm.with_word_object as object_cnn_lstm
import experiment.models.show_attend_tell.normal as normal_sat
from experiment.train.utils import clip_gradient
import pickle
from experiment.utils.vocab import Vocabulary
import logging

def get_conf(model_name):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_csv = 'data/artemis_train_dataset.csv'
    # train_csv = 'data/artemis_mini.csv'
    test_csv = 'data/artemis_test_dataset.csv'
    
    idx2obj_csv = 'data/idx2object.csv'

    with open('data/vocab.pkl', 'rb') as f:
        vocab = pickle.load(f)
        
    return {
        # dataset
        'device': device,
        'model_path': 'models/' + model_name + '/',
        'image_dir': 'data/resized',
        'vocab': vocab,
        'train_df': pd.read_csv(train_csv),
        'test_df': pd.read_csv(test_csv),
        'idx2obj_df': pd.read_csv(idx2obj_csv),
        'shuffle': True,

        # step
        'log_step': 10,
        'save_step': 100,

        # cnn-lstm
        'embed_size': 512,
        'hidden_size': 512,

        # show-attend-tell
        'alpha_c': 1.,
        'grad_clip': 5.,
        'attention_dim': 512,
        'decoder_dim': 512,
        'dropout': 0.5,
        'encoder_dim': 512,
        'embed_dim': 512,
        'clip_gradient': clip_gradient,

        # train
        'crop_size': 224,
        'num_layers': 1,
        'num_epochs': 10,
        'batch_size': 512,
        'num_workers': 0,
        'fine_tune_encoder': False,
        'encoder_lr': 1e-4,
        'decoder_lr': 4e-4,
    }

def get_model(model_name, conf):
    if model_name == 'cnn_lstm':
        encoder = normal_cnn_lstm.Encoder(conf['embed_size']).to(conf['device'])
        decoder = normal_cnn_lstm.Decoder(conf['embed_size'], conf['hidden_size'], len(conf['vocab']), conf['num_layers']).to(conf['device'])
    elif model_name == 'cnn_lstm_with_word_object':
        encoder = object_cnn_lstm.Encoder(len(conf['vocab']), conf['embed_size']).to(conf['device'])
        decoder = object_cnn_lstm.Decoder(conf['embed_size'], conf['hidden_size'], len(conf['vocab']), conf['num_layers']).to(conf['device'])
    elif model_name == 'show_attend_tell':
        encoder = normal_sat.Encoder().to(conf['device'])
        decoder = normal_sat.DecoderWithAttention(conf['attention_dim'], conf['embed_dim'], conf['decoder_dim'], len(conf['vocab']), conf['encoder_dim'], conf['dropout']).to(conf['device'])
        
    return encoder, decoder

def loging(i: int, conf: dict, epoch: int, total_step: int, loss):
    if i % conf['log_step'] == 0:
        logging.info('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Perplexity: {:5.4f}'
                .format(epoch, conf['num_epochs'], i, total_step, loss.item(), np.exp(loss.item()))) 

def saving(i: int, conf: dict, epoch, encoder, decoder):
    if (i+1) % conf['save_step'] == 0:
            torch.save(encoder.state_dict(), os.path.join(
                conf['model_path'], 'encoder-{}-{}.ckpt'.format(epoch+1, i+1)))
            torch.save(decoder.state_dict(), os.path.join(
                conf['model_path'], 'decoder-{}-{}.ckpt'.format(epoch+1, i+1)))