import glob
import os
import tqdm
import multiprocessing as mp
from detector.detectron.detectron2.config.config import get_cfg
from detector.detic.third_party.CenterNet2.centernet.config import add_centernet_config
from detector.detic.detic.config import add_detic_config
from detector.detic.detic.predictor import VisualizationDemo
from detector.detectron.detectron2.data.detection_utils import read_image
import argparse
from typing import List

def setup_args(input_image, output_image, search_method, search_words, confidence_threshold):
    parser = argparse.ArgumentParser(description="Detectron2 demo for builtin configs")
    parser.add_argument(
        "--config-file",
        default="src/detector/detic/configs/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.yaml",
    )
    parser.add_argument(
        "--input",
        nargs="+",
        default=[input_image],
    )
    parser.add_argument(
        "--output",
        default=output_image
    )
    parser.add_argument("--pred_all_class", action='store_true')
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=confidence_threshold,
    )
    parser.add_argument(
        "--opts",
        default=[],
    )
    
    if search_method == 'custom':
        parser.add_argument(
            "--vocabulary",
            default="custom",
            choices=['lvis', 'openimages', 'objects365', 'coco', 'custom'],
        )
        parser.add_argument(
            "--custom_vocabulary",
            default=search_words,
        )
    elif search_method == 'lvis':
        parser.add_argument(
            "--vocabulary",
            default="lvis",
            choices=['lvis', 'openimages', 'objects365', 'coco', 'custom'],
        )
        parser.add_argument(
            "--custom_vocabulary",
            default='',
        )
        
    return parser

def setup_cfg(args):

    cfg = get_cfg()
    
    cfg.MODEL.DEVICE="cpu"
    
    add_centernet_config(cfg)
    add_detic_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    
    # Set score_threshold for builtin models
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = args.confidence_threshold
    cfg.MODEL.ROI_BOX_HEAD.ZEROSHOT_WEIGHT_PATH = 'rand' # load later
    
    model_weights = 'models/detic/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth'
    cfg.MODEL.WEIGHTS = model_weights
    
    if not args.pred_all_class:
        cfg.MODEL.ROI_HEADS.ONE_CLASS_PER_PROPOSAL = True

    cfg.freeze()
    
    return cfg

def save_output_image(args, path):
    if os.path.isdir(args.output):
        assert os.path.isdir(args.output), args.output
        out_filename = os.path.join(args.output, os.path.basename(path))
    else:
        assert len(args.input) == 1, "Please specify a directory with args.output"
        out_filename = args.output

def get_object_info(input_image, search_word, confidence_threshold):
    mp.set_start_method("spawn", force=True)
    
    args = setup_args(
        input_image=input_image,
        output_image='data/detic/out.jpg',
        search_method='custom',
        search_words=search_word,
        confidence_threshold=confidence_threshold,
    ).parse_args()
    cfg = setup_cfg(args)
    
    demo = VisualizationDemo(cfg, args)
    args.input = glob.glob(os.path.expanduser(args.input[0]))
    assert args.input, "The input path(s) was not found"
    
    for path in args.input:
        img = read_image(path, format="BGR")
        predict_info, visualized_output, labels = demo.run_on_image(img)
        
    out_filename = args.output
    visualized_output.save(out_filename)
    
    return predict_info, labels

def object_detection_rate(search_words: str, predict_object_list: List[str]):
    search_word_list = search_words.split(',')
    
    cnt = 0
    for w in search_word_list:
        if w in predict_object_list: cnt += 1
        
    return cnt / len(search_word_list)

if __name__ == '__main__':
    search_words = 'person, glasses, tie, ear'
    
    predict_info, labels = get_object_info(
        input_image='data/detic/person.jpeg',
        search_word=search_words,
        confidence_threshold=0.01
    )
    
    rate = object_detection_rate(search_words, labels)
    print(rate)