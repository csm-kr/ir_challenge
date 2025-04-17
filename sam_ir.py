import os
import sys
import torch
import numpy as np
import torch.nn as nn
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..' ))) # models_를 path 에 추가 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..' ))) # models_를 path 에 추가 
from segment_anything import sam_model_registry, SamPredictor
from segment_anything.utils.amg import remove_small_regions
from ultralytics import YOLO


from PIL import Image


class SamBox(nn.Module):
    def __init__(self, predictor, args=None) -> None:
        super(SamBox, self).__init__()
        self.args = args
        self.predictor = predictor

    def preprocess(self, image_pil):
        image = np.array(image_pil)

        return image
    
    def postprocess(self, masks):
        masks = masks[:, 0:1, ...].type(torch.float32)
        return masks

    def forward(self, image_pil, boxes):

        image = self.preprocess(image_pil)
        boxes = boxes.to(self.predictor.model.device)
        self.predictor.set_image(image)
        transformed_boxes = self.predictor.transform.apply_boxes_torch(boxes, image.shape[:2])

        masks, _, _ = self.predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed_boxes,
            multimask_output=False,
        )

        masks = self.postprocess(masks)
        return masks
    

class YOLOSamBox(nn.Module):
    def __init__(self, args=None, device=None) -> None:
        super().__init__()
        self.args = args

        # load yolo
        yolo = YOLO("./runs/detect/train/weights/best.pt").to(device)  # pretrained YOLO11n model

        sam = sam_model_registry["vit_h"](checkpoint='./models/checkpoints/sam/sam_vit_h_4b8939.pth').to(device)
        # sam_model_type = "vit_b"   
        # sam = sam_model_registry[sam_model_type](checkpoint='./models/checkpoints/sam/vit_b.pth').to(device)
        predictor = SamPredictor(sam)
        sambox = SamBox(predictor)

        self.detector = yolo
        self.sambox = sambox

    def forward(self, image_dir):
        """
        Args:
            image_pil (PIL.Image): input image
            caption (str): caption
            box_threshold (float): threshold for filtering boxes
            top_k (int): number of boxes to keep
        Returns:
            boxes (torch.Tensor): boxes (B, 4) x1, y1, x2, y2 \in [0~W, 0~H], torch.int32
            masks (torch.Tensor): masks (B, 1, H, W), torch.float32
            scores (torch.Tensor): scores, (B), torch.float32
        """
        results = self.detector(image_dir)
        image_pil = Image.open(image_dir).convert('RGB')
        result = results[0]

        if len(result.boxes.xyxy) == 0:
            return 0, 0, 0, 0

        boxes = result.boxes.xyxy
        scores = result.boxes.conf
        cls = result.boxes.cls

        masks = self.sambox(image_pil, boxes.clone())

        removed_masks_np = []
        masks_np = masks.cpu().numpy()[:, 0, ...].astype(np.uint8)
        for mask_np in masks_np:
            removed_mask_np, _ = remove_small_regions(mask_np, area_thresh=10000, mode='holes')
            removed_mask_np, _ = remove_small_regions(removed_mask_np, area_thresh=10000, mode="islands")
            removed_masks_np.append(removed_mask_np)
        
        masks = torch.from_numpy(np.array(removed_masks_np)).to(self.sambox.predictor.model.device).unsqueeze(1).float()
        
        return boxes, masks, scores, cls

import cv2
import numpy as np

class_dict = {
    0: "person",
    1: "car",
    2: "truck",
    3: "bus",
    4: "bicycle",
    5: "bike",
    6: "extra_vehicle",
    7: "dog"
}

# def visualize_detections(image_cv2_bgr, boxes, masks, scores, classes, threshold=0.5, color_array=None, class_dict=None):
#     """
#     주어진 bounding boxes, masks, scores, classes를 기반으로 이미지에 객체 검출 결과를 시각화합니다.
    
#     Args:
#         image_cv2_bgr (np.array): BGR 형식의 원본 이미지.
#         boxes (iterable): 각 객체의 bounding box 정보를 담은 리스트 또는 배열. 포맷: [x1, y1, x2, y2].
#         masks (iterable): 각 객체의 이진 마스크 정보. 각 마스크는 원본 이미지와 같은 해상도여야 하며, 픽셀 값은 0 또는 1이어야 함.
#         scores (iterable): 각 객체의 검출 신뢰도 점수.
#         classes (iterable): 각 객체의 클래스 인덱스.
#         threshold (float): 시각화할 최소 신뢰도 임계값. 이 값 이상인 객체만 시각화됩니다.
#         color_array (list or np.array): 각 클래스 인덱스에 대응하는 색상 정보 배열. 예: model.color_array.
#         class_dict (dict): 클래스 인덱스와 클래스 이름을 매핑하는 딕셔너리.
    
#     Returns:
#         np.array: 검출 결과가 시각화된 이미지.
#     """
    
#     # 원본 이미지 복사 (원본 데이터 변경 방지)
#     vis_image = image_cv2_bgr.copy()
#     white_color = (255, 255, 255)  # 경계선 표시용 흰색
    
#     # 검출 결과들을 순회합니다.
#     for score, cls_idx, box, mask in zip(scores, classes, boxes, masks):
#         # score 임계치보다 작으면 스킵
#         if score < threshold:
#             continue
        
#         # 해당 클래스의 색상 결정 (color_array가 제공될 경우)
#         if color_array is not None:
#             # 만일 cls_idx가 텐서나 다른 형식이면 int로 변환 (예: int(cls_idx))
#             color = color_array[int(cls_idx)]
#             color = [int(c) for c in color]  # 색상 값 정수형으로 변환
#         else:
#             color = (0, 255, 0)  # 기본으로 초록색 사용
        
#         # 바운딩 박스 그리기
#         x1, y1, x2, y2 = map(int, box)
#         cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, 2)
        
#         mask = mask[0]  # torch.Size([1, 480, 640]) -->  torch.Size([480, 640])
        
#         # torch 이진 마스크를 uint8 형식(0 또는 255)으로 변환
#         mask_uint8 = (mask.numpy().astype(np.uint8) * 255)
#         # 마스크의 경계선을 추출 (Canny Edge Detection)
#         edge = cv2.Canny(mask_uint8, 0, 1)
#         # 팽창 커널 적용: 경계선 두께를 조정
#         kernel = np.ones((2, 2), np.uint8)
#         edge_mask = cv2.dilate(edge, kernel, iterations=1) == 255      
#         # 마스크 영역에 반투명 오버레이 적용 (가중합산을 통해 색상 블렌딩)
#         vis_image[mask == 1] = cv2.addWeighted(
#             vis_image[mask == 1], 0.4,
#             np.full_like(vis_image[mask == 1], color, dtype=np.uint8), 0.6, 0)
#         # 경계선 영역은 흰색으로 표시
#         vis_image[edge_mask] = white_color
        
#         # 클래스 이름 텍스트 생성
#         if class_dict is not None:
#             # {클래스 인덱스: 클래스 이름} 형식인 경우
#             class_name = class_dict.get(int(cls_idx), f'Class {cls_idx}')
#         else:
#             class_name = f'Class {cls_idx}'
#         label = f'{class_name} : {score:.2f}'
        
#         # 텍스트 출력 (박스의 좌측 상단)
#         cv2.putText(vis_image, text=label, org=(x1, y1),
#                     fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=1,
#                     color=(0, 255, 0), thickness=2)
    
#     return vis_image

def visualize_detections(image_cv2_bgr, boxes, masks, scores, classes, threshold=0.5, color_array=None, class_dict=None):
    """
    주어진 bounding boxes, masks, scores, classes를 기반으로 이미지에 객체 검출 결과를 시각화합니다.
    클래스별로 다른 색상을 사용합니다.

    Args:
        image_cv2_bgr (np.array): BGR 형식의 원본 이미지.
        boxes (iterable): 각 객체의 bounding box 정보. 포맷: [x1, y1, x2, y2].
        masks (iterable): 각 객체의 이진 마스크 정보.
        scores (iterable): 각 객체의 검출 신뢰도 점수.
        classes (iterable): 각 객체의 클래스 인덱스.
        threshold (float): 시각화할 최소 신뢰도 임계값.
        color_array (list or np.array, optional): 각 클래스 인덱스에 대응하는 색상 정보 배열.
                                                  제공되지 않으면 클래스별로 동적 색상 할당.
        class_dict (dict, optional): 클래스 인덱스와 클래스 이름을 매핑하는 딕셔너리.

    Returns:
        np.array: 검출 결과가 시각화된 이미지.
    """

    # 원본 이미지 복사
    vis_image = image_cv2_bgr.copy()
    white_color = (255, 255, 255)  # 경계선용 흰색

    # 클래스별 색상을 저장할 딕셔너리 (color_array가 없을 경우 사용)
    assigned_colors = {}

    # 각 클래스에 대한 고유 색상을 생성하는 함수 (필요시 사용)
    def get_class_color(cls_idx):
        if cls_idx not in assigned_colors:
            # 간단한 해싱 기반 색상 생성 (더 정교한 방법도 가능)
            # random.seed(cls_idx) # 시드 고정으로 매번 같은 색상 생성 가능
            # color = (random.randint(64, 255), random.randint(64, 255), random.randint(64, 255))
            # 또는 modulo 연산을 사용한 방법
            B = (cls_idx * 47) % 256 # 소수 사용
            G = (cls_idx * 97) % 256 # 다른 소수 사용
            R = (cls_idx * 137) % 256 # 또 다른 소수 사용
            # 너무 어두운 색 방지 (선택 사항)
            B = max(B, 64)
            G = max(G, 64)
            R = max(R, 64)
            assigned_colors[cls_idx] = (B, G, R)
        return assigned_colors[cls_idx]

    # 검출 결과 순회
    for score, cls_idx, box, mask in zip(scores, classes, boxes, masks):
        if score < threshold:
            continue

        # 클래스 인덱스를 정수형으로 변환
        try:
            # PyTorch 텐서 등 다른 타입일 수 있으므로 변환 시도
            int_cls_idx = int(cls_idx)
        except Exception as e:
            print(f"Warning: Could not convert class index {cls_idx} to int. Skipping. Error: {e}")
            continue

        # 색상 결정
        if color_array is not None:
            # 제공된 color_array 사용
            if int_cls_idx < len(color_array):
                color = color_array[int_cls_idx]
                # 색상 값이 float일 수 있으므로 int로 변환
                color = [int(c) for c in color]
            else:
                # color_array 범위를 벗어나는 경우 동적 색상 할당
                print(f"Warning: Class index {int_cls_idx} out of range for color_array. Assigning dynamic color.")
                color = get_class_color(int_cls_idx)
        else:
            # color_array가 없으면 동적으로 색상 할당/조회
            color = get_class_color(int_cls_idx)

        # --- 이하 시각화 로직은 거의 동일 ---

        # 바운딩 박스 그리기
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, 2)

        # 마스크 처리 (마스크 데이터 타입에 따라 조정 필요)
        try:
            # 입력 마스크 형태 확인 및 numpy 변환 (예: PyTorch Tensor)
            if hasattr(mask, 'cpu') and hasattr(mask, 'numpy'): # PyTorch Tensor 확인
                 # GPU -> CPU -> NumPy
                processed_mask = mask.cpu().numpy()
            elif isinstance(mask, np.ndarray):
                processed_mask = mask
            else:
                # 지원하지 않는 타입 처리 (필요시 에러 발생 또는 로깅)
                print(f"Warning: Unsupported mask type {type(mask)}. Skipping mask visualization for this object.")
                continue # 마스크 시각화 건너뛰기

            # 마스크 차원 확인 및 조정 (예: [1, H, W] -> [H, W])
            if processed_mask.ndim == 3 and processed_mask.shape[0] == 1:
                processed_mask = processed_mask[0]

            # 마스크가 이진(0 또는 1) 형태라고 가정하고 uint8 (0 또는 255)로 변환
            # 만약 마스크가 이미 0~1 사이 float 값이라면 (processed_mask > 0.5) 등으로 이진화 필요
            mask_uint8 = (processed_mask.astype(np.float32) * 255).astype(np.uint8) # float 거쳐서 변환 안정성 높임

            # 마스크 영역 유효성 검사 (형태가 이미지와 맞는지)
            if mask_uint8.shape != vis_image.shape[:2]:
                 print(f"Warning: Mask shape {mask_uint8.shape} does not match image shape {vis_image.shape[:2]}. Resizing mask.")
                 # 크기 조정 시도 (OpenCV 사용)
                 mask_uint8 = cv2.resize(mask_uint8, (vis_image.shape[1], vis_image.shape[0]), interpolation=cv2.INTER_NEAREST)


            # 마스크 경계선 추출
            edge = cv2.Canny(mask_uint8, 100, 200) # 임계값 조정 가능
            kernel = np.ones((2, 2), np.uint8)
            edge_mask = cv2.dilate(edge, kernel, iterations=1) == 255

            # 마스크 영역에 반투명 오버레이 적용 (색상 적용)
            # mask_uint8 > 0 (또는 다른 임계값) 를 사용하여 마스크 영역 선택
            mask_area = mask_uint8 > 128 # 이진 마스크 영역 선택 (임계값 조정 가능)
            vis_image[mask_area] = cv2.addWeighted(
                vis_image[mask_area], 0.4,
                np.full_like(vis_image[mask_area], color, dtype=np.uint8), 0.6, 0
            )
            # 경계선은 흰색으로 표시
            vis_image[edge_mask] = white_color

        except Exception as e:
            print(f"Error processing mask for class {int_cls_idx}: {e}")
            # 마스크 처리 중 오류 발생 시 다음 객체로 넘어감

        # 클래스 이름 및 점수 텍스트 생성
        class_name = f'Class {int_cls_idx}' # 기본값
        if class_dict is not None:
            class_name = class_dict.get(int_cls_idx, class_name) # 딕셔너리에서 이름 조회

        label = f'{class_name}: {score:.2f}'

        # 텍스트 표시 위치 계산 (박스 상단 또는 내부에 적절히)
        text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1, 2)
        text_w, text_h = text_size
        # 텍스트 배경 추가 (가독성 향상)
        cv2.rectangle(vis_image, (x1, y1 - text_h - 4), (x1 + text_w, y1), color, -1) # 배경 채우기
        cv2.putText(vis_image, label, (x1, y1 - 3),
                    cv2.FONT_HERSHEY_PLAIN, 1, white_color, 1, cv2.LINE_AA) # 흰색 텍스트

    return vis_image


if __name__ == "__main__":
    model = YOLOSamBox()

    # image_pil = Image.open("./data/yolo/images/val/val_0.png")
    # model(image_pil)
    file_path = "./data/yolo/images/val/val_0.png"
    boxes, masks, scores, cls = model(image_dir="./data/yolo/images/val/val_0.png")

    image_cv = cv2.imread(filename=file_path)
    vis_image = visualize_detections(image_cv2_bgr=image_cv, boxes=boxes, masks=masks, scores=scores, classes=cls, class_dict=class_dict)

    cv2.imshow('results', vis_image)
    cv2.waitKey(0)