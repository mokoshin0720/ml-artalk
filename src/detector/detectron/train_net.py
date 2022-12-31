# Dectron2 Tutorial.ipynb https://colab.research.google.com/drive/16jcaJoc6bCFAQ96jDe2HwtXj7BMD_-m5
# データセットがCOCOフォーマットなら以下のコードは次の3行で置き換えることができる。
# from detector.detectron.detectron2..data.datasets import register_coco_instances
# register_coco_instances("my_dataset_train", {}, "json_annotation_train.json", "path/to/image/dir")
# register_coco_instances("my_dataset_val", {}, "json_annotation_val.json", "path/to/image/dir")

import os
from detector.detectron.detectron2 import model_zoo
from detector.detectron.detectron2.engine import DefaultTrainer
from detector.detectron.detectron2.config import get_cfg

def train_dataset(dataset_name):
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("src/detector/detectron/configs/COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml"))
    cfg.DATASETS.TRAIN = (dataset_name,)
    cfg.DATASETS.TEST = ()
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml")
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
    cfg.SOLVER.MAX_ITER = 1
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512
    
    print(cfg)
    
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = DefaultTrainer(cfg) 
    trainer.resume_or_load(resume=False)
    
    print('start train')
    trainer.train()

if __name__ == '__main__':
    train_dataset_name = 'coco_2017_train_panoptic_separated'
    train_dataset(train_dataset_name)