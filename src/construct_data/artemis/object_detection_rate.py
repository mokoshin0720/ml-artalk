import csv
import pandas as pd
from construct_data.artemis.detect_object import get_object_info, object_detection_rate

if __name__ == '__main__':
    filename = 'data/idx2object.csv'
    origin_df = pd.read_csv('data/artemis_dataset.csv')
    idx2object_df = pd.read_csv(filename)
    
    checked_filenames = []
    
    for index, row in idx2object_df.iterrows():
        if index > 100: break
        
        sentence_id = row['sentence_id']
        art_style = origin_df.at[sentence_id, "art_style"]
        painting = origin_df.at[sentence_id, "painting"]
        image_filename = 'data/wikiart/{}/{}.jpg'.format(art_style, painting)
        
        if image_filename in checked_filenames: continue
        checked_filenames.append(image_filename)
        
        sentence_ids = origin_df[(origin_df['art_style'] == art_style) & (origin_df['painting'] == painting)].index
        print(sentence_ids)
        
        object_list = idx2object_df[idx2object_df['sentence_id'].isin(sentence_ids)]['object'].tolist()
        object_list = ','.join(object_list)
        
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