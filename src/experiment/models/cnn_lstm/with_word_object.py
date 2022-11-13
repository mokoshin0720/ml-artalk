from matplotlib import image
import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence

class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_size):
        super(Encoder, self).__init__()

        resnet = models.resnet34(pretrained=True)
        modules = list(resnet.children())[:-1]

        self.resnet = nn.Sequential(*modules)
        self.linear = nn.Linear(resnet.fc.in_features, embed_size)
        self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)

        self.embed = nn.Embedding(vocab_size, embed_size)

    def forward(self, images, input_objects):
        with torch.no_grad():
            features = self.resnet(images)
        
        features = features.reshape(features.size(0), -1)
        features = self.bn(self.linear(features))
        
        object_features = self.embed(input_objects).view(features.size(0), -1)
        features = torch.cat([features, object_features], 1)

        return features

class Decoder(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers, max_seq_length=20):
        super(Decoder, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size * 6)
        self.lstm = nn.LSTM(embed_size * 6, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.max_seq_length = max_seq_length

    def forward(self, features, captions, lengths):
        embeddings = self.embed(captions)
        embeddings = torch.cat((features.unsqueeze(1), embeddings), 1)
        packed = pack_padded_sequence(embeddings, lengths, batch_first=True, enforce_sorted=False)
        hiddens, _ = self.lstm(packed)
        outputs = self.linear(hiddens[0])

        return outputs

    def predict(self, features, states=None):
        predicted_ids = []
        inputs = features.unsqueeze(1)

        for _ in range(self.max_seq_length):
            hiddens, states = self.lstm(inputs, states)
            outputs = self.linear(hiddens.squeeze(1))
            _, predicted = outputs.max(1)
            predicted_ids.append(predicted)
            inputs = self.embed(predicted)
            inputs = inputs.unsqueeze(1)

        predicted_ids = torch.stack(predicted_ids, 1)

        return predicted_ids

class Net(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers, max_seq_length):
        super(Net, self).__init__()
        self.encoder = Encoder(vocab_size, embed_size)
        self.decoder = Decoder(embed_size, hidden_size, vocab_size, num_layers, max_seq_length)

    def forward(self, images, input_objects, captions, lengths):
        features = self.encoder.forward(images, input_objects)
        outputs = self.decoder.forward(features, captions, lengths)

        return outputs