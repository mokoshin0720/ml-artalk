import spacy
from sklearn.linear_model import LogisticRegression
import numpy as np

def train(nlp):
    train_set = [
        ['apple', 'owl', 'house'],
        ['agony', 'kowledge', 'process'],
    ]

    x = np.stack([list(nlp(w))[0].vector for part in train_set for w in part])
    y = [label for label, part in enumerate(train_set) for _ in part]
    classifier = LogisticRegression(C=0.1, class_weight='balanced').fit(x, y)
    return classifier

def classify(classifier, token):
    classes = ['concrete', 'abstract']

    return classes[classifier.predict([token.vector])[0]]

if __name__ == "__main__":
    nlp = spacy.load("en_core_web_sm")
    classifier = train(nlp)

    for token in nlp("Have a seat in that chair with comfort and drink some juice to soothe your thirst."):
        if token.pos_ == 'NOUN':
            result = classify(classifier, token)
            print(token, result)