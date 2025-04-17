import torch 
from ultralytics import YOLO

if __name__ == "__main__":
    device = torch.device('cuda:0')
    yolo = YOLO("./runs/detect/train/weights/best.pt").to(device)  # pretrained YOLO11n model

    results = yolo(["./data/yolo/images/val/val_0.png"])

    for result in results:
        boxes = result.boxes  # Boxes object for bounding box outputs
        masks = result.masks  # Masks object for segmentation masks outputs
        probs = result.probs  # Probs object for classification outputs

        print("boxes : ", boxes)
        print("probs : ", probs)


        keypoints = result.keypoints  # Keypoints object for pose outputs
        obb = result.obb  # Oriented boxes object for OBB outputs
        result.show()  # display to screen
        result.save(filename="result.jpg")  # save to disk
