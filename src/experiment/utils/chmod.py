import shutil
import os

if __name__ == '__main__':
    dir_name = 'data/coco/'
    shutil.rmtree(dir_name)
    # os.chmod(dir_name, 0o777)