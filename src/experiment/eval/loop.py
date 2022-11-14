import experiment.models.cnn_lstm.normal as normal_cnn_lstm
import experiment.models.cnn_lstm.with_word_object as word_object_cnn_lstm

def eval_loop(
    model_name: str,
    encoder,
    decoder,
    conf: dict,
    dataloader,
):
    if model_name == 'cnn_lstm':
        cnn_lstm(encoder, decoder, conf, dataloader)
    elif model_name == 'cnn_lstm_with_object':
        cnn_lstm_with_object(encoder, decoder, conf, dataloader)
    
def cnn_lstm(
    encoder: normal_cnn_lstm.Encoder,
    decoder: normal_cnn_lstm.Decoder,
    conf: dict,
    data_loader,
):
    encoder.eval()
    decoder.eval()
    
    for images, captions, lengths in data_loader:
            images, captions, lengths = images.to(conf['device']), captions.to(conf['device']), lengths.to(conf['device'])
            
            features = encoder(images)
            predicted_ids = decoder.predict(features)
            predicted_ids = predicted_ids[0].cpu().numpy()

            predicted_caption = []
            for word_id in predicted_ids:
                word = conf['vocab'].idx2word[word_id]
                predicted_caption.append(word)
                if word == '<eos>': break
            sentence = ' '.join(predicted_caption[1:-2])

            print(sentence)
            
def cnn_lstm_with_object(
    encoder: word_object_cnn_lstm.Encoder,
    decoder: word_object_cnn_lstm.Decoder,
    conf: dict,
    data_loader,
):
    encoder.eval()
    decoder.eval()
    
    for images, input_objects, captions, lengths in data_loader:
            images, input_objects, captions = images.to(conf['device']), input_objects.to(conf['device']), captions.to(conf['device'])
            
            features = encoder(images, input_objects)
            predicted_ids = decoder.predict(features)
            predicted_ids = predicted_ids[0].cpu().numpy()

            predicted_caption = []
            for word_id in predicted_ids:
                word = conf['vocab'].idx2word[word_id]
                predicted_caption.append(word)
                if word == '<eos>': break
            sentence = ' '.join(predicted_caption[1:-2])

            print(sentence)