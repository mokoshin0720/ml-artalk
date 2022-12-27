import pandas as pd
from construct_data.artemis.detect_object import get_object_info, object_detection_rate
import matplotlib.pyplot as plt
import statistics

if __name__ == '__main__':
    ratio_dict_by_style = {
        'Abstract_Expressionism': [],
        'Action_painting': [],
        'Analytical_Cubism': [],
        'Art_Nouveau_Modern': [],
        'Baroque': [],
        'Color_Field_Painting': [],
        'Contemporary_Realism': [],
        'Cubism': [],
        'Early_Renaissance': [],
        'Expressionism': [],
        'Fauvism': [],
        'High_Renaissance': [],
        'Impressionism': [],
        'Mannerism_Late_Renaissance': [],
        'Minimalism': [],
        'Naive_Art_Primitivism': [],
        'New_Realism': [],
        'Northern_Renaissance': [],
        'Pointillism': [],
        'Pop_Art': [],
        'Post_Impressionism': [],
        'Realism': [],
        'Rococo': [],
        'Romanticism': [],
        'Symbolism': [],
        'Synthetic_Cubism': [],
        'Ukiyo_e': [],
    }
    
    origin_df = pd.read_csv('data/artemis_dataset.csv')
    idx2object_df = pd.read_csv('data/idx2object_test.csv', header=0)
    
    checked_filenames = []
    
    for index, row in idx2object_df.iterrows():
        if index > 10: break
        
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
            search_method='custom',
            search_word=object_list,
            confidence_threshold=0.1,
        )
        
        rate = object_detection_rate(search_words=object_list, predict_object_list=predict_labels)
        
        ratio_dict_by_style[art_style].append(rate)
        
        print('search word: {}'.format(object_list))
        print('predict word: {}'.format(predict_labels))
        print(rate)
        print(ratio_dict_by_style)
        print('DONE!')
        print('--------------------------')
    
    x_pos = [i for i in range(len(ratio_dict_by_style))]
    labels = [k for k, _ in ratio_dict_by_style.items()]
    ratios = [statistics.mean(v) for _, v in ratio_dict_by_style.items()]
    plt.rcParams['font.family'] = 'Noto Sans JP'
    plt.figure(figsize = (10, 10))
    plt.bar(x_pos, ratios, tick_label=labels, align='center')
    plt.xticks(rotation=90)
    plt.title('絵画スタイルごとの検出率')
    plt.tight_layout()
    plt.savefig('detection_rate.png')