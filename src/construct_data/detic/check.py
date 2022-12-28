import os
from PIL import Image

def check_exif():
    source_file = 'data/imagenet/ImageNet-LVIS-raw/n02780815/n02780815_4317.JPEG'
    stylize_file = 'data/imagenet/ImageNet-LVIS/n02780815/n02780815_4317.JPEG'
    
    source_img = Image.open(source_file)
    exif = source_img._getexif()
    orientation = exif.get(0x112, 1)
    print(orientation)
    
    source_img = source_img.rotate(90)
    
    print(source_img.size)
    

def check_imagenet():
    raw_dir = "data/imagenet/ImageNet-LVIS-raw/"
    stylized_dir = "data/imagenet/ImageNet-LVIS/"
    
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

    print('start checking...')
    n_dirs = os.listdir(raw_dir)

    for i, d in enumerate(n_dirs):
        print('start dir {}'.format(d))
        image_files = os.listdir(raw_dir+d)
        
        for j, file_name in enumerate(image_files):
            source_file = raw_dir + d + '/' + file_name
            target_file = stylized_dir + d + "/" + file_name
            
            if not os.path.exists(target_file):
                print('not found {}...'.format(target_file))
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
            
            if source_file == 'data/imagenet/ImageNet-LVIS-raw/n02780815/n02780815_4317.JPEG':
                print(source_w, source_h)
                print(target_w, target_h)
            
            if source_w != target_w or source_h != target_h:
                print("different size {}...".format(file_name))
                pass
            
            source_img.close()
            target_img.close()
            
if __name__ == '__main__':
    check_imagenet()
    # check_exif()