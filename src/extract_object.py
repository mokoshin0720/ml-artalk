from select import select
import pandas as pd
import nltk
import numpy as np

def extract_noun(text):
    morph = nltk.word_tokenize(text)
    pos = nltk.pos_tag(morph)

    select_speeches = ['NN', 'NNS', 'NNPS']
    noun_list = []

    for word in pos:
        if word[1] in select_speeches: noun_list.append(word[0])
    return noun_list

if __name__ == '__main__':
    nltk.download('punkt')
    nltk.download('averaged_perceptron_tagger')
    nltk.download('brown')

    filename = 'artemis_dataset.csv'
    df = pd.read_csv(filename)

    object_list = []
    cnt = 0
    for index, row in df.iterrows():
        obj = extract_noun(row['utterance'])
        if len(obj) == 0:
            cnt += 1
            # print(row['utterance'])
            # print(obj)
            # print('=====================================')
    print(cnt)
    #     object_list.append(obj)
    
    # df['noun'] = object_list

    # df.to_csv('artemis_mini_noun.csv')