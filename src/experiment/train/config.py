import pandas as pd
from experiment.train.utils import clip_gradient
import pickle
from experiment.utils.vocab import Vocabulary

def get_conf():
    # device = 'cuda:0'
    device = 'cuda:1'
    # device = 'cuda:2'
    # device = 'cuda:3'
    
    train_csv = 'data/train.csv'
    test_csv = 'data/origin_test.csv'
    train_object_txt_dir = 'data/image_info/train/object'
    test_object_txt_dir = 'data/image_info/test/object'
    train_mask_txt_dir = 'data/image_info/train/mask'
    test_mask_txt_dir = 'data/image_info/test/mask'
    
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
    
    use_model = show_attend_tell_with_object

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
        'train_object_txt_dir': train_object_txt_dir,
        'test_object_txt_dir': test_object_txt_dir,
        'train_mask_txt_dir': train_mask_txt_dir,
        'test_mask_txt_dir': test_mask_txt_dir,
        'shuffle': True,

        # step
        'log_step': 10,

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
        'num_epochs': 1, # 30?
        'batch_size': 128,
        'num_workers': 0,
        'fine_tune_encoder': False,
        'encoder_lr': 1e-4,
        'decoder_lr': 1e-4,
    }