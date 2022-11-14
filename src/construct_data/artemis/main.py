import csv
import pandas as pd
import spacy
from validate import is_valid, get_sconj_like_range
from classify_abstruct_or_concrete import train
from tqdm import tqdm

def extract_noun_chunks(sentence, nlp, classifier):
    doc = nlp(sentence)
    token_dic = {}
    result = []

    for i, token in enumerate(doc):
        token_dic[i] = [token.i, token.text, token.pos_, token.dep_, token.head.i]

    sconj_like_range = get_sconj_like_range(doc)

    for noun_chunk in doc.noun_chunks:
        if is_valid(noun_chunk, sconj_like_range, token_dic, classifier):
            result.append(noun_chunk.root.lemma_)
        else:
            continue
    
    return result

if __name__ == '__main__':
    nlp = spacy.load('en_core_web_lg')
    classifier = train(nlp)

    filename = 'data/artemis_dataset.csv'
    df = pd.read_csv(filename)

    idx2object_list = []

    bar = tqdm(total=len(df))
    for index, row in df.iterrows():
        noun_chunks = extract_noun_chunks(row['utterance'], nlp, classifier)
        for noun_chunk in noun_chunks:
            idx2object = [index+1, noun_chunk]
            idx2object_list.append(idx2object)
            bar.update(1)

    with open('data/idx2object.csv', 'w') as file:
        writer = csv.writer(file, lineterminator='\n')
        writer.writerows(idx2object_list)