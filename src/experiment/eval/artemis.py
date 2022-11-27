import pandas as pd
import itertools
import pprint
import statistics
from construct_data.artemis.img2comments import list_utterances_of_img
from sentence_transformers import SentenceTransformer, util

def get_similarity(model, list_sentences):
    sentence_embeddings = model.encode(list_sentences)    
    combinations = list(itertools.combinations(sentence_embeddings, 2))
    
    similarities = []
    for comb in combinations:
        similarities.append(float(util.pytorch_cos_sim(comb[0], comb[1])))
        
    return statistics.mean(similarities)

if __name__ == '__main__':
    dataset = 'data/artemis_dataset.csv'
    df = pd.read_csv(dataset)
    
    model = SentenceTransformer('distilbert-base-nli-mean-tokens')
    
    paintings = []
    avg_similarities = []
    for idx, row in df.iterrows():
        painting = row['painting']
        
        if painting in paintings: continue
        
        paintings.append(painting)
        list_utterances = list_utterances_of_img(df,painting)
        
        print(get_similarity(model, list_utterances))
        avg_similarities.append(get_similarity(model, list_utterances))
    
    print(statistics.mean(avg_similarities))