from torchinfo import summary
import experiment.models.cnn_lstm.normal as normal_cnn_lstm
from experiment.models.describe.utils import get_vocab_size, get_normal_input_data
from experiment.utils.vocab import Vocabulary

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

    images, captions, lengths = get_normal_input_data()

    summary(
        model,
        input_data=[images, captions, lengths]
    )

if __name__ == '__main__':
    describe_normal_cnn_lstm()
