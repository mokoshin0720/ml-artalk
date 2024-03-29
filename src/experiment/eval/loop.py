import experiment.models.cnn_lstm.normal as normal_cnn_lstm
import experiment.models.cnn_lstm.with_word_object as word_object_cnn_lstm
from experiment.eval.method import batch_eval_scores
from torch.nn.utils.rnn import pack_padded_sequence
from experiment.train.utils import plotting
import statistics

def eval_loop(
    model_name: str,
    encoder,
    decoder,
    conf: dict,
    dataloader,
    criterion,
    epoch,
):
    if model_name == 'cnn_lstm':
        cnn_lstm(encoder, decoder, conf, dataloader, criterion, epoch)
    elif model_name == 'cnn_lstm_with_object':
        cnn_lstm_with_object(encoder, decoder, conf, dataloader, criterion, epoch)
    
def cnn_lstm(
    encoder: normal_cnn_lstm.Encoder,
    decoder: normal_cnn_lstm.Decoder,
    conf: dict,
    data_loader,
    criterion,
    epoch
):
    encoder.eval()
    decoder.eval()
    
    losses = []
    for filenames, images, captions, lengths in data_loader:
        images = images.to(conf['device'])
        captions = captions.to(conf['device'])
        targets = pack_padded_sequence(captions, lengths, batch_first=True, enforce_sorted=False)[0]
            
        features = encoder(images)
        outputs = decoder(features, captions, lengths)
            
        loss = criterion(outputs, targets)
        losses.append(loss)
            
        # predicted_ids = decoder.predict(features)
            
        # predicted_ids = predicted_ids.cpu().numpy()
        # target_ids = captions.cpu().numpy()

        # for i, (caption_id_list, predicted_id_list) in enumerate(zip(target_ids, predicted_ids)):
        #     predicted_sentence = ''
        #     target_sentence = ''

        #     for i, predicted_id in enumerate(predicted_id_list):
        #         word = conf['vocab'].idx2word[predicted_id]
        #         if word == '<eos>': break

        #         if i == 0:
        #             predicted_sentence = word
        #         else:
        #             predicted_sentence = predicted_sentence + ' ' + word

        #     for i, caption_id in enumerate(caption_id_list):
        #         word = conf['vocab'].idx2word[caption_id]
        #         if word == '<eos>': break

        #         if i == 0:
        #             target_sentence = word
        #         else:
        #             target_sentence = target_sentence + ' ' + word
        #         target_sentence += ' '
        
            # print(target_sentence)
            # print(predicted_sentence)
            # TODO: 評価の追加（バッチじゃなくて良さそう & 配列を渡さなくて良いように変更した方が良さそう）
        
    
        plotting(epoch, plotting(epoch, statistics.mean(losses)))

def cnn_lstm_with_object(
    encoder: word_object_cnn_lstm.Encoder,
    decoder: word_object_cnn_lstm.Decoder,
    conf: dict,
    data_loader,
    criterion,
    epoch
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