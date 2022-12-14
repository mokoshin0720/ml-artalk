import pandas as pd
import pickle
import nltk
nltk.download('punkt')
from collections import Counter

class Vocabulary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0

    def add_word(self, word):
        if not word in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def __call__(self, word):
        if not word in self.word2idx:
            return self.word2idx['<unk>']
        return self.word2idx[word]

    def __len__(self):
        return len(self.word2idx)

def make_vocab(df, threshold):
    counter = Counter()

    for i, row in df.iterrows():
        caption = str(row['utterance'])
        tokens = nltk.tokenize.word_tokenize(caption.lower())
        counter.update(tokens)

        if (i+1) % 10 == 0:
            print("[{}/{}] Tokenized captions.".format(i+1, len(df)))

        words = [word for word, cnt in counter.items() if cnt >= threshold]

    vocab = Vocabulary()
    vocab.add_word('<pad>')
    vocab.add_word('<sos>')
    vocab.add_word('<eos>')
    vocab.add_word('<unk>')

    for i, word in enumerate(words):
        vocab.add_word(word)

    return vocab

if __name__ == '__main__':
    filename = 'data/artemis_dataset.csv'
    df = pd.read_csv(filename)

    vocab = make_vocab(df, threshold=4)

    vocab_path = 'data/vocab.pkl'
    with open(vocab_path, 'wb') as f:
        pickle.dump(vocab, f)

    print("Total vocabulary size: {}".format(len(vocab)))
    print("Saved vocabulary wrapper to '{}'".format(vocab_path))