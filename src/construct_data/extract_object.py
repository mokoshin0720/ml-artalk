import pandas as pd
import spacy
from validate import is_valid, get_sconj_like_range
from classify_abstruct_or_concrete import train

def extract_noun_chunks(sentence, nlp, classifier):
    doc = nlp(sentence)
    token_dic = {}
    all_chunk_nouns = []
    all_root_nouns = []
    result = []
    root_result = []

    for i, token in enumerate(doc):
        token_dic[i] = [token.i, token.text, token.pos_, token.dep_, token.head.i]

    sconj_like_range = get_sconj_like_range(doc)

    for noun_chunk in doc.noun_chunks:
        all_chunk_nouns.append(noun_chunk)
        all_root_nouns.append(noun_chunk.root.lemma_)
        if is_valid(noun_chunk, sconj_like_range, token_dic, classifier):
            result.append(noun_chunk.text)
            root_result.append(noun_chunk.root.text)
        else:
            continue
    
    return result, root_result, all_chunk_nouns, all_root_nouns

if __name__ == '__main__':
    nlp = spacy.load('en_core_web_lg')
    classifier = train(nlp)

    filename = 'data/artemis_mini_translated.csv'
    df = pd.read_csv(filename)

    object_list = []
    cnt = 0
    for index, row in df.iterrows():
        noun_chunks, root_nouns, all_chunk_nouns, all_root_nouns = extract_noun_chunks(row['utterance'], nlp, classifier)
        print(row['utterance'])
        print(row['ja_utterance'])
        print(all_chunk_nouns)
        print(all_root_nouns)
        print(noun_chunks)
        print(root_nouns)
        print('=====================================')
        if len(noun_chunks) == 0:
            object_list.append(row['utterance'])
            cnt += 1
    print('**************************************')
    print('1つも名詞が検出されていない例文')
    print('**************************************')
    print(object_list)
    print(cnt)
    #     object_list.append(obj)
    
    # df['noun'] = object_list

    # df.to_csv('artemis_mini_noun.csv')