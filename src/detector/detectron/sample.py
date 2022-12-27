from detector.detectron.detectron2.config import get_cfg
from detector.detectron.detectron2 import model_zoo
from detector.detectron.detectron2.engine import DefaultPredictor
from detector.detectron.detectron2.data import MetadataCatalog
from detector.detectron.detectron2.utils.visualizer import Visualizer
import cv2

if __name__ == '__main__':
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml"))
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = 0.7
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml")
    predictor = DefaultPredictor(cfg)
    
    im = cv2.imread('data/coco/val2017/000000051314.jpg')
    panoptic_seg, segments_info = predictor(im)["panoptic_seg"]
    v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
    v = v.draw_panoptic_seg_predictions(panoptic_seg.to("cpu"), segments_info)
    # cv2_imshow(v.get_image()[:, :, ::-1])
    cv2.imwrite('data/detic/out-of-panoptic.jpg', v.get_image()[:, :, ::-1])