import shutil
import os

if __name__ == '__main__':
    dir_name = 'output/Detic/BoxSup-C2_Lbase_CLIP_R5021k_640b64_4x/inference_lvis_v1_val/lvis_instances_results.json'
    os.chmod(dir_name, 0o777)