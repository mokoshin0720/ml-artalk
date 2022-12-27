import os
from PIL import Image
import traceback
from notify.logger import notify_success, notify_fail, init_logger
import logging

def resize_coco():
    raw_dir = "data/coco/train2017-raw/"
    stylized_dir = "data/coco/train2017/"
    resized_dir = "data/coco/train2017-resized/"

    image_files = os.listdir(raw_dir)    
    for file_name in image_files:
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
                print('different size {}...'.format(target_img))
                
                target_resize = target_img.resize((source_w, source_h), Image.LANCZOS)
                target_resize.save(resized_dir + file_name)
            else:
                target_img.save(resized_dir + file_name)
            source_img.close()
            target_img.close()
            
def resize_imagenet():
    raw_dir = "data/imagenet/ImageNet-LVIS-raw/"
    stylized_dir = "data/imagenet/ImageNet-LVIS/"
    resized_dir = "data/imagenet/ImageNet-LVIS-resized/"

    n_dirs = os.listdir(raw_dir)

    for i, d in enumerate(n_dirs):
        logging.info('start dir {}/{}'.format(i, len(n_dirs)))
        image_files = os.listdir(raw_dir+d)
        
        if not os.path.exists(resized_dir + d):
            os.makedirs(resized_dir + d)
            os.chmod(resized_dir+d, 0o777)
            
        for j, file_name in enumerate(image_files):
            logging.info('start image_files {}/{}'.format(j, len(image_files)))
            source_file = raw_dir + d + '/' + file_name
            target_file = stylized_dir + d + "/" + file_name
            
            if not os.path.exists(target_file):
                logging.info('not found {}...'.format(target_file))
                continue
            
            source_img = Image.open(source_file)
            target_img = Image.open(target_file)
            
            source_w, source_h = source_img.size
            target_w, target_h = target_img.size
            
            if source_w != target_w or source_h != target_h:
                target_resize = target_img.resize((source_w, source_h), Image.LANCZOS)
                target_resize.save(resized_dir + d + "/" + file_name)
            else:
                target_img.save(resized_dir + d + "/" + file_name)
            
            source_img.close()
            target_img.close()
            
def resize_objects365():
    raw_dir = "data/objects365/val/images-raw/"
    stylized_dir = "data/objects365/val/images/"
    resized_dir = "data/objects365/val/images-resized/"

    n_dirs = os.listdir(raw_dir)

    for i, d in enumerate(n_dirs):
        logging.info('start dir {}/{}'.format(i, len(n_dirs)))
        image_files = os.listdir(raw_dir+d)
        
        if not os.path.exists(resized_dir + d):
            os.makedirs(resized_dir + d)
            os.chmod(resized_dir+d, 0o777)
            
        for j, file_name in enumerate(image_files):
            logging.info('start image_files {}/{}'.format(j, len(image_files)))
            source_file = raw_dir + d + '/' + file_name
            target_file = stylized_dir + d + "/" + file_name
            
            if not os.path.exists(target_file):
                logging.info('not found {}...'.format(target_file))
                continue
            
            source_img = Image.open(source_file)
            target_img = Image.open(target_file)
            
            source_w, source_h = source_img.size
            target_w, target_h = target_img.size
            
            if source_w != target_w or source_h != target_h:
                target_resize = target_img.resize((source_w, source_h), Image.LANCZOS)
                target_resize.save(resized_dir + d + "/" + file_name)
            else:
                target_img.save(resized_dir + d + "/" + file_name)
            
            source_img.close()
            target_img.close()
        
def resize_oid():
    raw_dir = "data/oid/images-raw/"
    stylized_dir = "data/oid/images/"
    resized_dir = "data/oid/images-resized/"

    n_dirs = os.listdir(raw_dir)

    for i, d in enumerate(n_dirs):
        logging.info('start dir {}/{}'.format(i, len(n_dirs)))
        image_files = os.listdir(raw_dir+d)
        
        if not os.path.exists(resized_dir + d):
            os.makedirs(resized_dir + d)
            os.chmod(resized_dir+d, 0o777)
            
        for j, file_name in enumerate(image_files):
            logging.info('start image_files {}/{}'.format(j, len(image_files)))
            source_file = raw_dir + d + '/' + file_name
            target_file = stylized_dir + d + "/" + file_name
            
            if not os.path.exists(target_file):
                logging.info('not found {}...'.format(target_file))
                continue
            
            source_img = Image.open(source_file)
            target_img = Image.open(target_file)
            
            source_w, source_h = source_img.size
            target_w, target_h = target_img.size
            
            if source_w != target_w or source_h != target_h:
                target_resize = target_img.resize((source_w, source_h), Image.LANCZOS)
                target_resize.save(resized_dir + d + "/" + file_name)
            else:
                target_img.save(resized_dir + d + "/" + file_name)
            
            source_img.close()
            target_img.close()

if __name__ == '__main__':
    log_filename = init_logger()
    try:
        # resize_imagenet()
        # resize_objects365()
        resize_oid()
    except Exception as e:
        traceback.print_exc()
        notify_fail(str(e))
    else:
        notify_success(log_filename)