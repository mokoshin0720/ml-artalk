import torch
from torchvision import transforms
from experiment.train.config import get_conf, get_model
from experiment.dataloader.normal import get_loader

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

