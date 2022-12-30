import os
from PIL import Image
import traceback
from notify.logger import notify_success, notify_fail, init_logger
import logging
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

def resize_coco():
    raw_dir = "data/coco/train2017-raw/"
    stylized_dir = "data/coco/train2017-stylized/"
    resized_dir = "data/coco/train2017-resized/"
    
    if not os.path.exists(resized_dir):
        os.makedirs(resized_dir)
        os.chmod(resized_dir, 0o777)

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
                print('-------------------')
                print('same')
                print('-------------------')
                target_img.save(resized_dir + file_name)
            source_img.close()
            target_img.close()
            
def resize_imagenet():
    raw_dir = "data/imagenet/ImageNet-LVIS-raw/"
    stylized_dir = "data/imagenet/ImageNet-LVIS-stylized/"
    resized_dir = "data/imagenet/ImageNet-LVIS-resized/"
    
    if not os.path.exists(resized_dir):
        os.makedirs(resized_dir)
        os.chmod(resized_dir, 0o777)
    
    convert_image = {
        1: lambda img: img,
        2: lambda img: img.transpose(Image.FLIP_LEFT_RIGHT),                              # 左右反転
        3: lambda img: img.transpose(Image.ROTATE_180),                                   # 180度回転
        4: lambda img: img.transpose(Image.FLIP_TOP_BOTTOM),                              # 上下反転
        5: lambda img: img.transpose(Image.FLIP_LEFT_RIGHT).transpose(Pillow.ROTATE_90),  # 左右反転＆反時計回りに90度回転
        6: lambda img: img.transpose(Image.ROTATE_270),                                   # 反時計回りに270度回転
        7: lambda img: img.transpose(Image.FLIP_LEFT_RIGHT).transpose(Pillow.ROTATE_270), # 左右反転＆反時計回りに270度回転
        8: lambda img: img.transpose(Image.ROTATE_90),                                    # 反時計回りに90度回転
    }

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
            
            exif = source_img._getexif()
            if exif is not None:
                orientation = exif.get(0x112, 1)
                if orientation == 6:
                    target_img = Image.open(target_file).rotate(270)
                    source_h, source_w = source_img.size
                elif orientation == 8:
                    target_img = Image.open(target_file).rotate(90)
                    source_h, source_w = source_img.size
                else:
                    target_img = Image.open(target_file)
                    source_w, source_h = source_img.size
            else:
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
        resize_imagenet()
    except Exception as e:
        traceback.print_exc()
        notify_fail(str(e))
    else:
        notify_success(log_filename)