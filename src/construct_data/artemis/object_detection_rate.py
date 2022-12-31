import pandas as pd
from construct_data.artemis.detect_object import get_object_info, object_detection_rate, get_panoptic_info
import matplotlib.pyplot as plt

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
        
        print("start {}...".format(image_filename))
        
        sentence_ids = origin_df[(origin_df['art_style'] == art_style) & (origin_df['painting'] == painting)].index
        
        object_list = idx2object_df[idx2object_df['sentence_id'].isin(sentence_ids)]['object'].tolist()
        
        try:
            panoptic_masks, panoptic_labels = get_panoptic_info(
                input_image=image_filename
            )
        except Exception as e:
            panoptic_masks, panoptic_labels = [], []
        
        search_words = ""
        for obj in object_list:
            if obj not in panoptic_labels: search_words = search_words + obj + ','
        print('------------------------')
        print(object_list)
        print(panoptic_labels)
        print(search_words)
        print('------------------------')
        
        detic_masks, detic_labels = get_object_info(
            input_image=image_filename,
            search_method='custom',
            search_word=search_words,
            confidence_threshold=0.1,
        )
        
        rate = object_detection_rate(
            search_words=','.join(object_list), 
            detic_object_words=detic_labels,
            detic_object_masks=detic_masks,
            panoptic_object_words=panoptic_labels,
            panoptic_object_masks=panoptic_masks
        )
        
        ratio_dict_by_style[art_style].append(rate)
        
        print('object list: {}'.format(object_list))
        print('search words: {}'.format(search_words))
        print('panoptic predict word: {}'.format(panoptic_labels))
        print('detic predict word: {}'.format(detic_labels))
        print(rate)
        print(ratio_dict_by_style)
        print('DONE!')
        print('--------------------------')
    
    x_pos = [i for i in range(len(ratio_dict_by_style))]
    labels = [k for k, _ in ratio_dict_by_style.items()]
    ratios = []
    for _, v in ratio_dict_by_style.items():
        try:
            ratio = sum(v)/len(v)
            ratios.append(ratio)
        except ZeroDivisionError:
            ratios.append(0)
    # plt.rcParams['font.family'] = 'Noto Sans JP'
    # plt.figure(figsize = (10, 10))
    # plt.bar(x_pos, ratios, tick_label=labels, align='center')
    # plt.xticks(rotation=90)
    # plt.title('絵画スタイルごとの検出率')
    # plt.tight_layout()
    # plt.savefig('detection_rate.png')