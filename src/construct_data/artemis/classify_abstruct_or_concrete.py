import spacy
from sklearn.linear_model import LogisticRegression
import numpy as np
import pandas as pd
import nltk
nltk.download('averaged_perceptron_tagger')

def train(nlp):
    filename = 'data/brysbaer.xlsx'
    df = pd.read_excel(filename)

    concrete_word_list = df[df['Conc.M'] >= 3]['Word'].tolist()
    abstract_word_list = df[df['Conc.M'] < 3]['Word'].tolist()
    
    concrete_word_list = extract_noun(concrete_word_list)
    abstract_word_list = extract_noun(abstract_word_list)
    
    train_set = [concrete_word_list, abstract_word_list]

    x = np.stack([list(nlp(w))[0].vector for part in train_set for w in part])
    y = [label for label, part in enumerate(train_set) for _ in part]
    classifier = LogisticRegression(C=0.1, class_weight='balanced').fit(x, y)
    return classifier

def extract_noun(word_list):
    result = []
    
    for word_pos in nltk.pos_tag(word_list[:10]):
        if word_pos[1] in ['NN', 'NNS', 'NNP', 'NNPS', 'PP', 'PRP$']: result.append(word_pos[0])
    
    return result

def classify(classifier, token):
    classes = ['concrete', 'abstract']

    return classes[classifier.predict([token.vector])[0]]

if __name__ == "__main__":
    nlp = spacy.load("en_core_web_lg")
    classifier = train(nlp)

    for token in nlp("Have a seat in that chair with comfort and drink some juice to soothe your thirst."):
        if token.pos_ == 'NOUN':
            result = classify(classifier, token)
            print(token, result)