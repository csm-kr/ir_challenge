from ultralytics import YOLO

# go to sitepackages/ultralytics/datasets
# # Ultralytics YOLO 🚀, AGPL-3.0 license
# # COCO8 dataset (first 8 images from COCO train2017) by Ultralytics
# # Documentation: https://docs.ultralytics.com/datasets/detect/coco8/
# # Example usage: yolo train data=coco8.yaml
# # parent
# # ├── ultralytics
# # └── datasets
# #     └── coco8  ← downloads here (1 MB)

# # Train/val/test sets as 1) dir: path/to/imgs, 2) file: path/to/imgs.txt, or 3) list: [path/to/imgs1, path/to/imgs2, ..]
# path: C:\Users\cho_sm\Desktop\hanhwa_ir_seg\data\yolo # dataset root dir
# train: C:\Users\cho_sm\Desktop\hanhwa_ir_seg\data\yolo\images\train # train images (relative to 'path') 4 images
# val: C:\Users\cho_sm\Desktop\hanhwa_ir_seg\data\yolo\images\val # val images (relative to 'path') 4 images
# test: # test images (optional)

# # Classes
# names:
#   0: person
#   1: car
#   2: truck
#   3: bus
#   4: bicycle
#   5: bike
#   6: extra_vehicle
#   7: dog

def main_train():
    # Load a COCO-pretrained YOLO11n model
    model = YOLO("yolo11x.pt")

    # Train the model on the COCO8 example dataset for 100 epochs
    results = model.train(data="hanhwa_ir.yaml", epochs=100, imgsz=640)

    # Run inference with the YOLO11n model on the 'bus.jpg' image
    # results = model("path/to/bus.jpg")

# # Load a COCO-pretrained YOLO11n model
# model = YOLO("yolo11n.pt")

# # Train the model on the COCO8 example dataset for 100 epochs
# results = model.train(data="hanhwa_ir.yaml", epochs=100, imgsz=640)

# # Run inference with the YOLO11n model on the 'bus.jpg' image
# # results = model("path/to/bus.jpg")


if __name__ == '__main__':
    main_train()