import numpy as np
import os
import pandas as pd
from notify.logger import notify_success, notify_fail, init_logger
import traceback
import random

def calc_disjunction(target_list):
    if len(target_list) == 0 :
        return eval(target_list[0])
    else:
        result = eval(target_list[0])
        
        for target in target_list[1:]:
            result = np.logical_or(result, eval(target))
        return result
    
def launch(train_or_test):
    csv_dir = 'data/object2mask/{}'.format(train_or_test)
    save_dir = 'data/image_info/{}'.format(train_or_test)
    list_files = os.listdir(csv_dir)
    
    random.shuffle(list_files)
    
    for i, filename in enumerate(list_files):
        print('start {}/{}[{}]'.format(i, len(list_files), i/len(list_files)*100))
        save_mask_path = save_dir + '/' + 'mask/' + filename.replace('csv', 'txt')
        save_object_path = save_dir + '/' + 'object/' + filename.replace('csv', 'txt')
        
        if os.path.isfile(save_mask_path):
            print('{} already exist!'.format(save_mask_path), flush=True)
            continue
        
        obj2mask_df = pd.read_csv(csv_dir + '/' + filename)
        
        mask_list = list(obj2mask_df['mask'])
        mask_list = calc_disjunction(mask_list)
        with open(save_mask_path, 'w') as f:
            f.write(str(mask_list))
            f.close()
            
        noun_list = list(obj2mask_df['object'])
        with open(save_object_path, 'w') as f:
            f.write(str(noun_list))
            f.close()
        
        print('-------------------------------', flush=True)
        
def print_fail_files():
    target_dir = 'data/image_info/train/mask/'
    list_files = os.listdir(target_dir)
    for i, filename in enumerate(list_files):
        if sum([1 for _ in open(target_dir + filename)]) == 7:
            os.remove(target_dir + filename)
    return
    
if __name__ == '__main__':
    log_filename = init_logger()
    try:
        # launch('train')
        print_fail_files()
    except Exception as e:
        traceback.print_exc()
        notify_fail(str(e))
    else:
        notify_success(log_filename)