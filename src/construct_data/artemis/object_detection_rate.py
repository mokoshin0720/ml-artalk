import pandas as pd
from construct_data.artemis.detect_object import get_object_info, object_detection_rate, get_panoptic_info
import matplotlib.pyplot as plt
from notify.logger import notify_success, notify_fail, init_logger
import traceback

def get_panoptic_dict(
    object_list,
    panoptic_labels,
    panoptic_masks,
):
    result = {}
    for obj in object_list:
        if obj in panoptic_labels:
            idx = panoptic_labels.index(obj)
            result[obj] = panoptic_masks[idx]
            
    return result

def get_detic_dict(
    object_list,
    detic_labels,
    detic_masks,
):
    result = {}
    for obj in object_list:
        if obj in detic_labels:
            idx = detic_labels.index(obj)
            result[obj] = detic_masks[idx]
            
    return result

def launch():
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
    
    origin_df = pd.read_csv('data/artemis_test_dataset.csv')
    idx2object_df = pd.read_csv('data/idx2object_test.csv', header=0)
    
    checked_filenames = []
    new_csv_list = [] # 最終的に欲しいもの [[sentence_id, object, mask]]
    
    for _, row in idx2object_df.iterrows():
        sentence_id = row['sentence_id']
        art_style = origin_df.at[sentence_id, "art_style"]
        painting = origin_df.at[sentence_id, "painting"]
        image_filename = 'data/wikiart/{}/{}.jpg'.format(art_style, painting)
        
        if image_filename in checked_filenames: continue
        checked_filenames.append(image_filename)
        
        sentence_ids = origin_df[(origin_df['art_style'] == art_style) & (origin_df['painting'] == painting)].index
        object_list = idx2object_df[idx2object_df['sentence_id'].isin(sentence_ids)]['object'].tolist()
        
        object_dict = {} # {'word1': idx1, 'word2': idx2}
        for id in sentence_ids:
            for obj in idx2object_df.query('sentence_id == {}'.format(id))['object'].tolist():
                object_dict[obj] = id
                
        try:
            panoptic_masks, panoptic_labels = get_panoptic_info(
                input_image=image_filename
            )
        except Exception as e:
            panoptic_masks, panoptic_labels = [], []
        
        search_words = ""
        for obj in object_list:
            if obj not in panoptic_labels: search_words = search_words + obj + ','
        
        detic_masks, detic_labels = get_object_info(
            input_image=image_filename,
            search_method='custom',
            search_word=search_words,
            confidence_threshold=0.1,
        )
        
        rate = object_detection_rate(
            search_words=','.join(object_list), 
            detic_object_words=detic_labels,
            panoptic_object_words=panoptic_labels,
        )
        ratio_dict_by_style[art_style].append(rate)
        
        panoptic_dict = get_panoptic_dict(
            object_list=object_list,
            panoptic_labels=panoptic_labels,
            panoptic_masks=panoptic_masks
        ) # {'word1': mask1, 'word2': mask2}
        
        detic_dict = get_detic_dict(
            object_list=object_list,
            detic_labels=detic_labels,
            detic_masks=detic_masks
        ) # {'word1': mask1, 'word2': mask2}
        
        for obj, mask in panoptic_dict.items():
            if obj in object_dict:
                sentence_id = object_dict[obj]
                mask = str(mask.tolist())
                row = [sentence_id, obj, mask]
                new_csv_list.append(row)
            else:
                continue
            
        for obj, mask in detic_dict.items():
            if obj in object_dict:
                mask = str(mask.tolist())
                sentence_id = object_dict[obj]
                row = [sentence_id, obj, mask]
                new_csv_list.append(row)
            else:
                continue
        
    mask_df = pd.DataFrame(new_csv_list,columns =['sentence_id', 'object', 'mask'])
    mask_df.to_csv('data/object2mask_test.csv', index=False)
    
    x_pos = [i for i in range(len(ratio_dict_by_style))]
    labels = [k for k, _ in ratio_dict_by_style.items()]
    ratios = []
    for _, v in ratio_dict_by_style.items():
        try:
            ratio = sum(v)/len(v)
            ratios.append(ratio)
        except ZeroDivisionError:
            ratios.append(0)
    plt.rcParams['font.family'] = 'Noto Sans JP'
    plt.figure(figsize = (10, 10))
    plt.bar(x_pos, ratios, tick_label=labels, align='center')
    plt.xticks(rotation=90)
    plt.title('絵画スタイルごとの検出率')
    plt.tight_layout()
    plt.savefig('detection_rate.png')

if __name__ == '__main__':
    log_filename = init_logger()
    try:
        launch()
    except Exception as e:
        traceback.print_exc()
        notify_fail(str(e))
    else:
        notify_success(log_filename)