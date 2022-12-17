import shutil
import os

if __name__ == '__main__':
    dir_name = 'data/imagenet/ImageNet-LVIS'
    os.chmod(dir_name, 0o777)