from construct_data.detic.detectron2.utils.logger import setup_logger
setup_logger()

import numpy as np
import cv2
import pprint
import os

from construct_data.detic.detectron2 import model_zoo
from construct_data.detic.detectron2.engine import DefaultPredictor
from construct_data.detic.detectron2.config import get_cfg
from construct_data.detic.detectron2.utils.visualizer import Visualizer
from construct_data.detic.detectron2.data import MetadataCatalog

def do_segmentation():
    cfg = get_cfg()
    # add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model

    # Find a model from construct_data.detic.detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    predictor = DefaultPredictor(cfg)

    im = cv2.imread("./data/resized/zinaida-serebriakova_woman-in-blue-1934.jpg")
    outputs = predictor(im)
    v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)

    # out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    cv2.imwrite('test.png', out.get_image()[:, :, ::-1])
    
def get_segmentation_pixel(object: str):
    cfg = get_cfg()
    # add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model

    # Find a model from construct_data.detic.detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    predictor = DefaultPredictor(cfg)

    im = cv2.imread("data/resized/zinaida-serebriakova_woman-in-blue-1934.jpg")
    outputs = predictor(im)
    
    target = MetadataCatalog.get(cfg.DATASETS.TRAIN[0]).thing_classes.index(object)
    classes = np.asarray(outputs['instances'].to('cpu').pred_classes)
    masks = np.asarray(outputs['instances'].to('cpu').pred_masks)[classes==target].astype('uint8')
    contours = [cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0] for m in masks]
    con = np.asarray(contours)[0]
    for i in range(1, len(masks)):
        con = con + np.asarray(contours)[i]
        
    pprint.pprint(con)
    im_con = im.copy()
    output = cv2.drawContours(im_con, con, -1, (0, 255, 0), 2)
    cv2.imwrite('person.png', output)
    
if __name__ == '__main__':
    # do_segmentation()
    get_segmentation_pixel('person')