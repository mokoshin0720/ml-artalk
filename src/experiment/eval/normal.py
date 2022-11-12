import torch
from experiment.train.config import get_conf, get_model
from experiment.dataloader.normal import get_loader
from experiment.dataset.normal import get_dataset
from experiment.utils.vocab import Vocabulary
from notify.logger import notify_success, notify_fail, init_logger

def evaluate(dataset, model_name, encoder_path, decoder_path):
    conf = get_conf(model_name)
    data_loader = get_loader(dataset, conf['batch_size'], conf['shuffle'], conf['num_workers'])

    with torch.no_grad():
        accs = []

        encoder, decoder = get_model(model_name, conf)
        encoder.load_state_dict(torch.load(encoder_path))
        decoder.load_state_dict(torch.load(decoder_path))

        data_loader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=conf['batch_size'],
            shuffle=conf['shuffle'],
        )

        for images, captions, lengths in data_loader:
            images, captions, lengths = images.to(conf['device']), captions.to(conf['device']), lengths.to(conf['device'])

            features = encoder(images)
            predicted_ids = decoder.predict(features)
            predicted_ids = predicted_ids[0].numpy()

            predicted_caption = []
            for word_id in predicted_ids:
                word = conf['vocab'].idx2word[word_id]
                predicted_caption.append(word)
                if word == '<end>': break
            sentence = ' '.join(predicted_caption[1:-2])

            print(sentence)

if __name__ == '__main__':
    log_filename = init_logger()
    try:
        # model_name = 'cnn_lstm'
        model_name = 'show_attend_tell'

        encoder_path = 'models/cnn_lstm/encoder-4-150.ckpt'
        decoder_path = 'models/cnn_lstm/decoder-4-150.ckpt'

        conf = get_conf(model_name)
        dataset = get_dataset(conf, is_train=False)
        evaluate(dataset, model_name, encoder_path, decoder_path)
    except Exception as e:
        notify_fail(str(e))
    else:
        notify_success(log_filename)