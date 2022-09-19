import pandas as pd
import spacy

def extract_noun_chunks(sentence, nlp):
    doc = nlp(sentence)
    result = []

    for noun_chunk in doc.noun_chunks:
        if is_valid(str(noun_chunk)):
            result.append(noun_chunk.text)
        else:
            continue
    
    return result

def is_valid(noun, token_dep_lemmma_dic):
    PRP_LIST = ['i', 'my', 'me', 'mine', 'you', 'your', 'yours', 'she', 'her', 'hers', 'he', 'his', 'him', 'they', 'their', 'them', 'theirs', 'we', 'our', 'us', 'ours', 'this', 'that', 'it']
    QUESTION_LIST = ['which', 'what', 'why', 'how', 'where', 'when', 'who', 'whatever', 'whenever', 'wherever', 'whoever']
    OTHER_LIST = ['all', 'some', 'any', 'something', 'anything', 'everything']

    invalid_list = PRP_LIST + QUESTION_LIST + OTHER_LIST
    if noun.lower() in invalid_list:
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