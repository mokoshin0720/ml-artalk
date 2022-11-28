import pandas as pd
import itertools
from tqdm import tqdm
import json
import pandas as pd
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

def artemis_similarities():
    print('analyze artemis...')
    
    dataset = 'data/artemis_dataset.csv'
    df = pd.read_csv(dataset)
    
    model = SentenceTransformer('distilbert-base-nli-mean-tokens')
    
    paintings = []
    avg_similarities = []
    bar = tqdm(total=len(df))
    for idx, row in tqdm(df.iterrows()):
        bar.update(1)
        painting = row['painting']
        
        if painting in paintings: continue
        paintings.append(painting)
        
        list_utterances = list_utterances_of_img(df,painting)
        
        sim = get_similarity(model, list_utterances)
        avg_similarities.append(sim)
    
    print('artemis similarities: ', statistics.mean(avg_similarities))

def coco_similarities():
    print('analyze ccoo...')
    model = SentenceTransformer('distilbert-base-nli-mean-tokens')
    
    dataset = json.load(open('data/coco/captions_train2014.json', encoding='utf-8'))

    annotations = dataset['annotations']
    
    image_ids = []
    avg_similarities = []
    for annotation in tqdm(annotations):
        image_id = annotation['image_id']
        
        if image_id in image_ids: continue
        image_ids.append(image_id)
        
        list_captions = [caption['caption'] for caption in list(filter(lambda x: x['image_id'] == image_id, annotations))]
        
        sim = get_similarity(model, list_captions)
        avg_similarities.append(sim)
        
    print('coco similarities: ', statistics.mean(avg_similarities))

if __name__ == '__main__':
    # artemis_similarities()
    coco_similarities()