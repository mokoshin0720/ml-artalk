import torch
import torch.nn as nn
from experiment.models.show_attend_tell.attention import Attention
import pprint

class DecoderWithAttention(nn.Module):
    def __init__(self, attention_dim, embed_dim, decoder_dim, vocab_size, encoder_dim, dropout, max_seq_length=50):
        super(DecoderWithAttention, self).__init__()
        self.encoder_dim = encoder_dim
        self.attention_dim = attention_dim
        self.embed_dim = embed_dim
        self.decoder_dim = decoder_dim
        self.vocab_size = vocab_size
        self.dropout = dropout
        self.max_seq_length = max_seq_length

        self.attention = Attention(encoder_dim, decoder_dim, attention_dim)

        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.dropout = nn.Dropout(p=self.dropout)
        self.decode_step = nn.LSTMCell(embed_dim + encoder_dim, decoder_dim, bias=True)
        self.init_h = nn.Linear(encoder_dim, decoder_dim)
        self.init_c = nn.Linear(encoder_dim, decoder_dim)
        self.f_beta = nn.Linear(decoder_dim, encoder_dim)
        self.sigmoid = nn.Sigmoid()
        self.fc_to_next_word = nn.Linear(decoder_dim, vocab_size)
        self.init_weights()

    def init_weights(self):
        self.embedding.weight.data.uniform_(-0.1, 0.1)
        self.fc_to_next_word.bias.data.fill_(0)
        self.fc_to_next_word.weight.data.uniform_(-0.1, 0.1)

    def init_hidden_state(self, encoder_out):
        mean_encoder_out = encoder_out.mean(dim=1)
        h = self.init_h(mean_encoder_out)
        c = self.init_c(mean_encoder_out)

        return h, c

    def forward(self, encoder_out, encoded_captions, caption_lengths):
        batch_size = encoder_out.size(0)
        encoder_dim = encoder_out.size(-1)
        vocab_size = self.vocab_size

        encoder_out = encoder_out.view(batch_size, -1, encoder_dim) # (batch_size, num_pixels, encoder_dim)
        num_pixels = encoder_out.size(1)
        
        caption_lengths, sort_idx = caption_lengths.sort(dim=0, descending=True)
        encoder_out = encoder_out[sort_idx]
        encoded_captions = encoded_captions[sort_idx]

        embeddings = self.embedding(encoded_captions)
        
        h, c = self.init_hidden_state(encoder_out)
        decode_lengths = (caption_lengths - 1).tolist()
        device = embeddings.device

        predictions = torch.zeros(batch_size, int(max(decode_lengths)), vocab_size).to(device)
        alphas = torch.zeros(batch_size, int(max(decode_lengths)), num_pixels).to(device)
        
        for t in range(int(max(decode_lengths))):
            batch_size_t = sum([l > t for l in decode_lengths])
            attention_weighted_encoding, alpha = self.attention(encoder_out[:batch_size_t], h[:batch_size_t])
            
            gate = self.sigmoid(self.f_beta(h[:batch_size_t]))
            attention_weighted_encoding = gate * attention_weighted_encoding
            
            h, c = self.decode_step(
                torch.cat([embeddings[:batch_size_t, t, :], attention_weighted_encoding], dim=1),
                (h[:batch_size_t], c[:batch_size_t])
            )
            
            preds = self.fc_to_next_word(self.dropout(h))
            predictions[:batch_size_t, t, :] = preds
            alphas[:batch_size_t, t, :] = alpha

        return predictions, encoded_captions, decode_lengths, alphas, sort_idx
    
    def predict(self, encoder_out):
        batch_size = encoder_out.size(0)
        encoder_dim = encoder_out.size(-1)
        encoder_out = encoder_out.view(batch_size, -1, encoder_dim) # (batch_size, num_pixels, encoder_dim)
        
        prev_words = torch.LongTensor([1] * batch_size).to('cuda:1') # TODO: vocab.sosで初期化するように修正
        
        batch_size = encoder_out.size(0)
        encoder_dim = encoder_out.size(-1)
        
        h, c = self.init_hidden_state(encoder_out)
        
        predicted_ids = []
        for _ in range(self.max_seq_length):
            h, c, pred_t, alpha = self.attend_and_predict_next_word(encoder_out, h, c, prev_words)
            prev_words = torch.argmax(pred_t, 1)
            
            predicted_ids.append(prev_words)
            
        predicted_ids = torch.stack(predicted_ids, 1)
        
        return predicted_ids
    
    def attend_and_predict_next_word(self, encoder_out, h, c, tokens):
        attention_weighted_encoding, alpha = self.attention(encoder_out, h)
        gate = self.sigmoid(self.f_beta(h))
        attention_weighted_encoding = gate * attention_weighted_encoding
        embeddings = self.embedding(tokens)
        
        decoder_input = torch.cat([embeddings, attention_weighted_encoding], dim=1)
        
        h, c = self.decode_step(decoder_input, (h, c))
        logits = self.fc_to_next_word(h)
        
        return h, c, logits, alpha

