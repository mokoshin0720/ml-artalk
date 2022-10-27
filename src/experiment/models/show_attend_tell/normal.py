import torch.nn as nn
from experiment.models.show_attend_tell.resnet_encoder import Encoder
from experiment.models.show_attend_tell.decoder_with_attention import DecoderWithAttention

class ShowAttendTell(nn.Module):
    def __init__(self, encoder_embed_size, decoder_embed_size, attention_dim, decoder_dim, vocab_size, encoder_dim, dropout):
        super(ShowAttendTell, self).__init__()
        self.encoder = Encoder(encoder_embed_size)
        self.decoder = DecoderWithAttention(attention_dim, decoder_embed_size, decoder_dim, vocab_size, encoder_dim, dropout)

    def forward(self, images, encoded_captions, caption_lengths):
        features = self.encoder.forward(images)
        outputs = self.decoder.forward(features, encoded_captions, caption_lengths)

        return outputs