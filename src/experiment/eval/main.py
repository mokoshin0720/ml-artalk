import torch
from experiment.train.config import get_conf
from experiment.dataloader.get import get_loader
from experiment.dataset.get import get_dataset
from experiment.train.utils import get_model
from experiment.utils.vocab import Vocabulary
from notify.logger import notify_success, notify_fail, init_logger
from experiment.eval.loop import eval_loop
import traceback

def evaluate(conf, encoder_path, decoder_path):
    dataset = get_dataset(conf, is_train=False)
    data_loader = get_loader(dataset, conf)

    with torch.no_grad():
        accs = []

        encoder, decoder = get_model(conf)
        
        encoder.load_state_dict(torch.load(encoder_path))
        decoder.load_state_dict(torch.load(decoder_path))
        
        eval_loop(
            model_name=conf['model_name'],
            encoder=encoder,
            decoder=decoder,
            conf=conf,
            dataloader=data_loader
        )

if __name__ == '__main__':
    log_filename = init_logger()
    try:
        conf = get_conf()
        encoder_path = 'models/cnn_lstm_with_object/encoder-9-800.ckpt'
        decoder_path = 'models/cnn_lstm_with_object/decoder-9-800.ckpt'
        
        evaluate(conf, encoder_path, decoder_path)
    except Exception as e:
        traceback.print_exc()
        notify_fail(str(e))
    else:
        notify_success(log_filename)