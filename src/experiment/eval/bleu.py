from typing import List
from nltk import bleu_score
import pandas as pd
from tqdm import tqdm
from construct_data.artemis.img2comments import list_utterances_of_img
from nltk import word_tokenize
import itertools
import statistics
import json
import pprint

class Bleu():
    def __init__(self, n_gram: int):
        weights_dict = {
            1: (1, 0, 0, 0),
            2: (0.5, 0.5, 0, 0),
            3: (0.33, 0.33, 0.33, 0),
            4: (0.25, 0.25, 0.25, 0.25),
        }
        self.weights = weights_dict[n_gram]

    def compute_score(
        self,
        hypothesis: List[str], 
        references: List[List[str]],
        ):
        fn = bleu_score.SmoothingFunction().method3
        normal_score = bleu_score.sentence_bleu(references, hypothesis, smoothing_function=fn)

        total_scores = 0
        for reference in references:
            total_scores += bleu_score.sentence_bleu(reference, hypothesis, smoothing_function=fn, weights=self.weights)
        
        return normal_score, total_scores/len(references) # 多分normal_scoreを使えば良い
    

def get_similarity(scorer: Bleu, list_sentences):
    divided_sentences = []
    for sentence in list_sentences:
        divided_sentences.append(sentence.split())
        
    similarities = []
    for i in range(len(divided_sentences)):
        src = divided_sentences[i]
        tgt = [sentence for idx, sentence in enumerate(divided_sentences) if idx != i]
        
        normal_score, _ = scorer.compute_score(src, tgt)
        similarities.append(normal_score)
        
    return statistics.mean(similarities)

def artemis_bleu(n_gram: int):
    print('analyze artemis...')
    
    dataset = 'data/artemis_dataset.csv'
    df = pd.read_csv(dataset)
    
    scorer = Bleu(n_gram=n_gram)
    
    paintings = []
    avg_similarities = []
    bar = tqdm(total=len(df))
    for idx, row in tqdm(df.iterrows()):
        bar.update(1)
        painting = row['painting']
        
        if painting in paintings: continue
        paintings.append(painting)
        
        list_utterances = list_utterances_of_img(df,painting)
        
        sim = get_similarity(scorer, list_utterances)
        avg_similarities.append(sim)
    
    print('artemis similarities {}-gram: '.format(n_gram), statistics.mean(avg_similarities))
    
def coco_bleu(n_gram: int):
    print('analyze ccoo...')
    scorer = Bleu(n_gram=n_gram)
    
    dataset = json.load(open('data/coco/captions_train2014.json', encoding='utf-8'))

    annotations = dataset['annotations']
    
    image_ids = []
    avg_similarities = []
    for annotation in tqdm(annotations):
        image_id = annotation['image_id']
        
        if image_id in image_ids: continue
        image_ids.append(image_id)
        
        list_captions = [caption['caption'] for caption in list(filter(lambda x: x['image_id'] == image_id, annotations))]
        
        sim = get_similarity(scorer, list_captions)
        avg_similarities.append(sim)
        
    print('coco similarities {}-gram: '.format(n_gram), statistics.mean(avg_similarities))
    
if __name__ == '__main__':
    artemis_bleu(n_gram=2)
    # coco_bleu(n_gram=2)