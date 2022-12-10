import csv
import pandas as pd
from construct_data.artemis.detect_object import get_object_info, object_detection_rate

if __name__ == '__main__':
    filename = 'data/idx2object.csv'
    origin_df = pd.read_csv('data/artemis_dataset.csv')
    idx2object_df = pd.read_csv(filename)
    
    checked_sentences = []
    
    for index, row in idx2object_df.iterrows():
        if index > 100: break
        
        sentence_id = row['sentence_id']
        if sentence_id in checked_sentences: continue
        
        checked_sentences.append(sentence_id)
        object_list = idx2object_df[idx2object_df['sentence_id'] == sentence_id]['object'].tolist()
        object_list = ','.join(object_list)
        
        art_style = origin_df.at[sentence_id-1, "art_style"]
        painting = origin_df.at[sentence_id-1, "painting"]        
        image_filename = 'data/wikiart/{}/{}.jpg'.format(art_style, painting)
        
        print('--------------------------')
        _, predict_labels = get_object_info(
            input_image=image_filename,
            search_word=object_list,
            confidence_threshold=0.1,
        )
        
        rate = object_detection_rate(search_words=object_list, predict_object_list=predict_labels)
        
        print('search word: {}'.format(object_list))
        print('predict word: {}'.format(predict_labels))
        print(rate)
        print('DONE!')
        print('--------------------------')