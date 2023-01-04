import shutil
import os

if __name__ == '__main__':
    dir_name = 'data/idx2object_train.csv'
    # shutil.rmtree(dir_name)
    os.chmod(dir_name, 0o777)