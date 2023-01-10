import pandas as pd
from construct_data.artemis.detect_object import get_object_info, object_detection_rate, get_panoptic_info
import matplotlib.pyplot as plt
from notify.logger import notify_success, notify_fail, init_logger
import traceback
import torch
import os
import pathlib

def is_exist_csv(sentence_ids):
    CSV_DATA = 'data/object2mask/train/'
    for id in sentence_ids:
        if os.path.isfile(CSV_DATA + str(id) + '.csv'): return True
    return False

def touch_filename(filename):
    target_dir = 'data/object2mask/checked/' + str(filename)
    if os.path.isfile(target_dir): return
    
    print('create!', flush=True)
    make_path = pathlib.Path(target_dir)
    make_path.touch()
    
    return

def launch(train_or_test):
    origin_df = pd.read_csv('data/artemis_{}_dataset.csv'.format(train_or_test))
    idx2object_df = pd.read_csv('data/idx2object_{}.csv'.format(train_or_test), header=0)
    
    checked_filenames = []
    
    for idx, row in idx2object_df.iterrows():
        print("{}/{}[{}%] start".format(idx, len(idx2object_df), idx/len(idx2object_df)*100), flush=True)
        sentence_id = row['sentence_id']
        art_style = origin_df.at[sentence_id, "art_style"]
        painting = origin_df.at[sentence_id, "painting"]
        image_filename = 'data/wikiart/{}/{}.jpg'.format(art_style, painting)
        
        if image_filename in checked_filenames: continue
        
        sentence_ids = origin_df[(origin_df['art_style'] == art_style) & (origin_df['painting'] == painting)].index
        if is_exist_csv(sentence_ids):
            checked_filenames.append(image_filename)
        else:
            continue
        print('checked_file_nums: {}'.format(len(checked_filenames)))
        print('===================================', flush=True)
        
    print(len(checked_filenames))
    
def generate_checked_file(train_or_test):
    origin_df = pd.read_csv('data/artemis_{}_dataset.csv'.format(train_or_test))
    idx2object_df = pd.read_csv('data/idx2object_{}.csv'.format(train_or_test), header=0)
    
    checked_filenames = []
    
    for idx, row in idx2object_df.iterrows():
        print("{}/{}[{}%] start".format(idx, len(idx2object_df), idx/len(idx2object_df)*100), flush=True)
        sentence_id = row['sentence_id']
        art_style = origin_df.at[sentence_id, "art_style"]
        painting = origin_df.at[sentence_id, "painting"]
        image_filename = 'data/wikiart/{}/{}.jpg'.format(art_style, painting)
        
        if image_filename in checked_filenames: continue
        
        sentence_ids = origin_df[(origin_df['art_style'] == art_style) & (origin_df['painting'] == painting)].index
        if is_exist_csv(sentence_ids):
            print('exist!', flush=True)
            touch_filename(art_style+painting)
        else:
            print('not found!', flush=True)
        checked_filenames.append(image_filename)
        
        print('checked_file_nums: {}'.format(len(checked_filenames)), flush=True)
        print('===================================', flush=True)
        
    print(len(checked_filenames))

if __name__ == '__main__':
    # launch('train')
    generate_checked_file('train')
    # len(all_checked_filenames) = 80022
    # 1/7(土)の夜実行 -> 26933