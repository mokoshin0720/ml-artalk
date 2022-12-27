import os
from PIL import Image
import traceback
from notify.logger import notify_success, notify_fail, init_logger
import logging

def check_coco():
    raw_dir = "data/coco/train2017-raw/"
    stylized_dir = "data/coco/train2017/"

    image_files = os.listdir(raw_dir)    
    for i, file_name in enumerate(image_files):
        print("{}/{}".format(i, len(image_files)))
        source_file = raw_dir + file_name
        target_file = stylized_dir + file_name
        
        if not os.path.isfile(target_file):
            print('not found {}...'.format(target_file))
        else:
            source_img = Image.open(source_file)
            target_img = Image.open(target_file)
            
            source_w, source_h = source_img.size
            target_w, target_h = target_img.size
            
            if source_w != target_w or source_h != target_h:
                print("************************************")
                print('different size {}...'.format(target_img))
                print("************************************")
            else:
                pass
                
            source_img.close()
            target_img.close()
            
if __name__ == '__main__':
    check_coco()