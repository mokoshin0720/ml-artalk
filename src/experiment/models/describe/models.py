from torch import dropout
from torchinfo import summary
import experiment.models.cnn_lstm.normal as normal_cnn_lstm
import experiment.models.cnn_lstm.with_word_object as cnn_lstm_with_object
import experiment.models.show_attend_tell.normal as normal_sat
from experiment.models.describe.utils import get_vocab_size, get_normal_input_data, get_data_with_object
from experiment.utils.vocab import Vocabulary
from experiment.train.config import get_conf

def describe_normal_cnn_lstm():
    embed_size = 256
    hidden_size=512
    num_layers=1
    vocab_size = get_vocab_size()

    model = normal_cnn_lstm.CNNLSTM(
        embed_size=embed_size,
        hidden_size=hidden_size,
        vocab_size=vocab_size,
        num_layers=num_layers,
    )

    conf = get_conf('cnn_lstm')

    images, captions, lengths = get_normal_input_data(conf)

    summary(
        model,
        input_data=[images, captions, lengths]
    )

def describe_cnn_lstm_with_object():
    embed_size = 256
    hidden_size=512
    num_layers=1
    vocab_size = get_vocab_size()
    max_seq_length = 20

    model = cnn_lstm_with_object.Net(
        embed_size=embed_size,
        hidden_size=hidden_size,
        vocab_size=vocab_size,
        num_layers=num_layers,
        max_seq_length=max_seq_length,
    )

    images, input_objects, captions, lengths = get_data_with_object()

    summary(
        model,
        input_data=[images, input_objects, captions, lengths]
    )

def describe_normal_sat():
    encoder_embed_size = 14
    decoder_embed_size = 512
    attention_dim = 512
    encoder_dim = 512
    decoder_dim = 512
    vocab_size = get_vocab_size()
    dropout=0.5

    model = normal_sat.ShowAttendTell(
        encoder_embed_size=encoder_embed_size,
        decoder_embed_size=decoder_embed_size,
        attention_dim=attention_dim,
        decoder_dim=decoder_dim,
        vocab_size=vocab_size,
        encoder_dim=encoder_dim,
        dropout=dropout
    )

    conf = get_conf('show_attend_tell')

    images, captions, lengths = get_normal_input_data(conf)

    summary(
        model,
        input_data=[images, captions, lengths]
    )

if __name__ == '__main__':
    # describe_normal_cnn_lstm()
    # describe_cnn_lstm_with_object()
    describe_normal_sat()
