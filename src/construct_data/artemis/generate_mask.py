import numpy as np
import os
import pandas as pd
from notify.logger import notify_success, notify_fail, init_logger
import traceback
import random
from PIL import Image

def calc_disjunction(target_list):
    if len(target_list) == 0 :
        return eval(target_list[0])
    else:
        result = eval(target_list[0])
        
        for target in target_list[1:]:
            result = np.logical_or(result, eval(target))
        return result

def resize(array, size):
    i = Image.fromarray(np.uint8(array))
    return np.asarray(i.resize((size, size)))

def launch(train_or_test):
    csv_dir = 'data/object2mask/{}'.format(train_or_test)
    save_dir = 'data/image_info/{}'.format(train_or_test)
    list_files = os.listdir(csv_dir)
    list_files = sorted(list_files)
    
    random.shuffle(list_files)
    
    for i, filename in enumerate(list_files):
        try:
            print('start {}/{}[{}%]'.format(i, len(list_files), i/len(list_files)*100), flush=True)
            save_mask_path = save_dir + '/' + 'mask/' + filename.replace('csv', 'txt')
            save_object_path = save_dir + '/' + 'object/' + filename.replace('csv', 'txt')

            if os.path.isfile(save_mask_path):
                print('{} already exist!'.format(save_mask_path), flush=True)
                continue
            
            obj2mask_df = pd.read_csv(csv_dir + '/' + filename)

            mask_list = list(obj2mask_df['mask'])
            mask_list = calc_disjunction(mask_list)
            mask_list = resize(mask_list, 256)
            np.savetxt(save_mask_path, np.array(mask_list))

            noun_list = list(obj2mask_df['object'])
            np.savetxt(save_object_path, np.array(noun_list), fmt='%s')

            os.remove(csv_dir+'/'+filename)

            print('-------------------------------', flush=True)
        except Exception as e:
            print('error {}'.format(e))
            continue
        
if __name__ == '__main__':
    log_filename = init_logger()
    try:
        launch('train')
    except Exception as e:
        traceback.print_exc()
        notify_fail(str(e))
    else:
        notify_success(log_filename)
    # launch('train')