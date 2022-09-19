import pandas as pd
import spacy
from pprint import pprint

def extract_noun_chunks(sentence, nlp):
    doc = nlp(sentence)
    token_dic = {}
    result = []

    for i, token in enumerate(doc):
        token_dic[i] = [token.i, token.text, token.pos_, token.dep_, token.head.i]

    pprint(token_dic)

    sconj_like_range = get_sconj_like_range(doc)

    for noun_chunk in doc.noun_chunks:
        if is_valid(noun_chunk, sconj_like_range):
            print(noun_chunk.root, noun_chunk.root.head.i, noun_chunk.root.head.text, noun_chunk.root.head.pos_)
            result.append(noun_chunk.text)
        else:
            continue
    
    return result

def get_sconj_like_range(doc):
    sconj_like_range = []
    is_started = False

    for token in doc:
        if is_started == False and str(token.text).lower() == "like" and token.pos_ == "SCONJ": 
            sconj_like_range.append(token.i)
            is_started = True
        elif is_started == True and token.pos_ == "PUNCT": 
            sconj_like_range.append(token.i)
            is_started = False

    if len(sconj_like_range) % 2 == 1: sconj_like_range.append(len(doc))

    return sconj_like_range

def is_valid(noun_chunk, sconj_like_range):
    PRP_LIST = ['i', 'my', 'me', 'mine', 'you', 'your', 'yours', 'she', 'her', 'hers', 'he', 'his', 'him', 'they', 'their', 'them', 'theirs', 'we', 'our', 'us', 'ours', 'this', 'that', 'it']
    QUESTION_LIST = ['which', 'what', 'why', 'how', 'where', 'when', 'who', 'whatever', 'whenever', 'wherever', 'whoever']
    OTHER_LIST = ['all', 'some', 'any', 'something', 'anything', 'everything']
    SPECIAL_LIST = ['evokes']
    invalid_list = PRP_LIST + QUESTION_LIST + OTHER_LIST + SPECIAL_LIST

    if str(noun_chunk).lower() in invalid_list:
        return False

    if str(noun_chunk.root.head.text).lower() == "like" and noun_chunk.root.head.pos_ == "ADP":
        return False

    if sconj_like_range == []: return True
    if sconj_like_range[0] < noun_chunk.root.i < sconj_like_range[1]:
        return False

    return True

if __name__ == '__main__':
    nlp = spacy.load('en_core_web_sm')

    filename = 'data/artemis_mini_translated.csv'
    df = pd.read_csv(filename)

    object_list = []
    cnt = 0
    for index, row in df.iterrows():
        noun_chunks = extract_noun_chunks(row['utterance'], nlp)
        print(row['utterance'])
        print(row['ja_utterance'])
        print(noun_chunks)
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