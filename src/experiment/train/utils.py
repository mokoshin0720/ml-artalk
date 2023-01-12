import torch
import experiment.models.cnn_lstm.normal as normal_cnn_lstm
import experiment.models.cnn_lstm.with_word_object as object_cnn_lstm
import experiment.models.show_attend_tell.normal as normal_sat
import experiment.models.show_attend_tell.normal as normal_sat
import logging
import os
import numpy as np
import wandb

def clip_gradient(optimizer, grad_clip):
    for group in optimizer.param_groups:
        for param in group["params"]:
            if param.grad is not None: param.grad.data.clamp_(-grad_clip, grad_clip)
            
def get_model(conf):
    if conf['model_name'] == 'cnn_lstm':
        encoder = normal_cnn_lstm.Encoder(conf['embed_size']).to(conf['device'])
        decoder = normal_cnn_lstm.Decoder(conf['embed_size'], conf['hidden_size'], len(conf['vocab']), conf['num_layers']).to(conf['device'])
    elif conf['model_name'] == 'cnn_lstm_with_object':
        encoder = object_cnn_lstm.Encoder(len(conf['vocab']), conf['embed_size']).to(conf['device'])
        decoder = object_cnn_lstm.Decoder(conf['embed_size'], conf['hidden_size'], len(conf['vocab']), conf['num_layers']).to(conf['device'])
    elif conf['model_name'] == 'show_attend_tell':
        encoder = normal_sat.Encoder().to(conf['device'])
        decoder = normal_sat.DecoderWithAttention(conf['attention_dim'], conf['embed_dim'], conf['decoder_dim'], len(conf['vocab']), conf['encoder_dim'], conf['dropout']).to(conf['device'])
    else:
        assert 'Invalid model name from get_model'
        
    return encoder, decoder

def loging(i: int, conf: dict, epoch: int, total_step: int, loss):
    if i % conf['log_step'] == 0:
        logging.info('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Perplexity: {:5.4f}'
                .format(epoch, conf['num_epochs'], i, total_step, loss.item(), np.exp(loss.item()))) 
        
def plotting(loss: int, is_train: bool):
    if is_train:
        wandb.log({'train_loss': loss})
    else:
        wandb.log({'val_loss': loss})

def saving(conf: dict, epoch, encoder, decoder):
    torch.save(
        encoder.state_dict(), 
        os.path.join(
        conf['model_path'], 'encoder-{}.ckpt'.format(epoch))
    )
    
    torch.save(
        decoder.state_dict(),
        os.path.join(
        conf['model_path'], 'decoder-{}.ckpt'.format(epoch))
    )