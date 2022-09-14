import pandas as pd
import spacy

def extract_noun(sentence, nlp):
    PRP_LIST = ['i', 'my', 'me', 'mine', 'you', 'your', 'yours', 'she', 'her', 'hers', 'he', 'his', 'him', 'they', 'their', 'them', 'theirs', 'we', 'our', 'us', 'ours', 'this', 'that', 'it']
    doc = nlp(sentence)
    result = []
    for noun_chunk in doc.noun_chunks:
        result.append(noun_chunk.text)
    for k, noun in enumerate(result):
        if noun.lower() in PRP_LIST: result.pop(k)
    return result

if __name__ == '__main__':
    nlp = spacy.load('en_core_web_sm')

    filename = 'artemis_mini_translated.csv'
    df = pd.read_csv(filename)

    object_list = []
    cnt = 0
    for index, row in df.iterrows():
        noun_chunks = extract_noun(row['utterance'], nlp)
        print(row['utterance'])
        print(row['ja_utterance'])
        print(noun_chunks)
        print('=====================================')
        if len(noun_chunks) == 0:
            object_list.append(row['utterance'])
            cnt += 1
    print('**************************************')
    print(object_list)
    print(cnt)
    #     object_list.append(obj)
    
    # df['noun'] = object_list

    # df.to_csv('artemis_mini_noun.csv')