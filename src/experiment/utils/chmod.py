import shutil
import os

if __name__ == '__main__':
    dir_name = 'output/Detic/Detic_LI_CLIP_SwinB_896b32_4x_ft4x_max-size'
    shutil.rmtree(dir_name)
    # os.chmod(dir_name, 0o777)