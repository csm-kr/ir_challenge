import os
import glob
from sam_ir import YOLOSamBox
from segment_anything.utils.amg import mask_to_rle_pytorch, coco_encode_rle
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import json 
import torch
from sam_ir import visualize_detections, class_dict
import cv2
from tqdm import tqdm

def eval_ir(path_list, model):
    return 

if __name__ == "__main__":
    
    device = torch.device('cuda:0')
    # load model 
    model = YOLOSamBox(device=device)

    # load val dataset
    eval_path_list = sorted(glob.glob(os.path.join("./data/yolo/images/val", '*.png')))

    # iterate through all images in the val dataset
    seg_results = []
    for path in tqdm(eval_path_list):

        # get image_id 
        image_id = int(path.split('_')[-1].split('.png')[0]) + 1

        # get results 
        boxes, masks, scores, classes = model(image_dir=path)
        if not isinstance(boxes, torch.Tensor):
            continue
        
        image_cv = cv2.imread(path)

        vis_image = visualize_detections(image_cv2_bgr=image_cv, boxes=boxes, masks=masks, scores=scores, classes=classes, class_dict=class_dict)
        cv2.imwrite(f'./outputs/{os.path.basename(path)}', vis_image)

        for i, (box, mask, score, label) in enumerate(zip(boxes, masks, scores, classes)):

            mask = mask_to_rle_pytorch(mask.type(torch.uint8))
            mask = coco_encode_rle(mask[0])

            seg_result = {
                        'image_id': image_id,
                        'category_id': int(label + 1),
                        'segmentation': mask,
                        'score': float(score)
                    }
            seg_results.append(seg_result)        

    with open(os.path.join('./pred_instances_val2025.json'), 'w') as f:
        json.dump(seg_results, f)

    coco_gt = COCO(f'./data/instances_val2025.json')

    # COCO eval for segmentations
    coco_seg_dt = coco_gt.loadRes('pred_instances_val2025.json')
    coco_seg_eval = COCOeval(coco_gt, coco_seg_dt, 'segm')
    coco_seg_eval.evaluate()
    coco_seg_eval.accumulate()
    coco_seg_eval.summarize()
