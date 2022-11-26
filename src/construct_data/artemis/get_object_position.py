from construct_data.detic.detectron2.utils.logger import setup_logger
setup_logger()

import numpy as np
import cv2
import pprint
import os
import traceback

from construct_data.detic.detectron2 import model_zoo
from construct_data.detic.detectron2.engine import DefaultPredictor
from construct_data.detic.detectron2.config import get_cfg
from construct_data.detic.detectron2.utils.visualizer import Visualizer
from construct_data.detic.detectron2.data import MetadataCatalog

def panoptic_segmentation(filename: str):
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml"))
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = 0.7
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml")
    predictor = DefaultPredictor(cfg)
    
    im = cv2.imread(filename)
    panoptic_seg, segments_info = predictor(im)["panoptic_seg"]
    
    v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
    v = v.draw_panoptic_seg_predictions(panoptic_seg.to("cpu"), segments_info)
    cv2.imwrite('person2.png', v.get_image()[:, :, ::-1])
    
def get_segmentation_pixel(filename: str, object: str):
    result = {}
    result['object'] = object
    
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml"))
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = 0.7
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml")
    predictor = DefaultPredictor(cfg)
    
    im = cv2.imread(filename)
    panoptic_seg, segments_info = predictor(im)["panoptic_seg"]
    
    v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
    
    try:
        mask = v.get_panoptic_pixel(panoptic_seg.to('cpu'), segments_info, object)
        print(mask)
    except:
        traceback.print_exc()

# def get_segmentation_pixel(filename: str ,object: str):
#     cfg = get_cfg()
#     cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
#     cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model

#     cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
#     predictor = DefaultPredictor(cfg)

#     im = cv2.imread(filename)
#     outputs = predictor(im)
    
#     target = MetadataCatalog.get(cfg.DATASETS.TRAIN[0]).thing_classes.index(object)
#     classes = np.asarray(outputs['instances'].to('cpu').pred_classes)
#     masks = np.asarray(outputs['instances'].to('cpu').pred_masks)[classes==target].astype('uint8')
#     contours = [cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0] for m in masks]
#     con = np.asarray(contours)[0]
#     for i in range(1, len(masks)):
#         con = con + np.asarray(contours)[i]
        
#     im_con = im.copy()
#     output = cv2.drawContours(im_con, con, -1, (0, 255, 0), 2)
#     cv2.imwrite('person.png', output)

if __name__ == '__main__':
    np.set_printoptions(threshold=np.inf)
    
    img_file = 'data/wikiart/Baroque/adriaen-brouwer_drinkers-in-the-yard.jpg'
    object = 'person'
    # get_segmentation_pixel(img_file, object)
    # panoptic_segmentation(img_file)
    get_segmentation_pixel(img_file, object)