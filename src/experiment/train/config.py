import pandas as pd
from experiment.train.utils import clip_gradient
import pickle
from experiment.utils.vocab import Vocabulary

def get_conf():
    device = 'cuda:0'
    # device = 'cuda:1'
    # device = 'cuda:2'
    # device = 'cuda:3'
    
    # train_csv = 'data/artemis_train_dataset.csv'
    train_csv = 'data/artemis_dataset.csv'
    # train_csv = 'data/artemis_mini.csv'
    test_csv = 'data/artemis_test_dataset.csv'
    idx2obj_csv = 'data/idx2object.csv'
    
    cnn_lstm = 'cnn_lstm'
    cnn_lstm_with_object = 'cnn_lstm_with_object'
    show_attend_tell = 'show_attend_tell'
    show_attend_tell_with_object = 'show_attend_tell_with_object'
    
    normal_models = [
        cnn_lstm,
        show_attend_tell
    ]
    word_object_models = [
        cnn_lstm_with_object,
        show_attend_tell_with_object,
    ]
    
    use_model = cnn_lstm_with_object

    with open('data/vocab.pkl', 'rb') as f:
        vocab = pickle.load(f)
        
    return {
        # models
        'model_name': use_model,
        'normal_models': normal_models,
        'word_object_models': word_object_models,
        
        # dataset
        'device': device,
        'model_path': 'models/' + use_model + '/',
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