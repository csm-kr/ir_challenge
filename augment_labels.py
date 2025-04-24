import os
import random
from PIL import Image
from collections import defaultdict


def collect_all_label_entries(label_dirs, image_dirs):
    """
    전체 라벨 파일을 스캔하여 라벨별로 (image_path, label_path) 리스트를 반환

    Returns:
        Dict[str, List[Tuple[str, str]]]
    """
    label_to_entries = defaultdict(list)

    for label_dir, image_dir in zip(label_dirs, image_dirs):
        for filename in os.listdir(label_dir):
            if not filename.endswith(".txt"):
                continue

            label_path = os.path.join(label_dir, filename)
            image_path = os.path.join(image_dir, filename.replace(".txt", ".png"))

            if not os.path.exists(image_path):
                continue

            with open(label_path, "r") as file:
                lines = file.readlines()

            for line in lines:
                label = line.strip().split()[0]
                label_to_entries[label].append((image_path, label_path))

    return label_to_entries


def augment_single_label_random_crop(
    label, labeled_image_label_dirs, iter_num, aug_image_dir, aug_label_dir
):
    os.makedirs(aug_image_dir, exist_ok=True)
    os.makedirs(aug_label_dir, exist_ok=True)

    label = str(label)
    augmented_count = 0
    attempt_limit = 20

    while augmented_count < iter_num:
        if not labeled_image_label_dirs:
            print(f"⚠️ 라벨 {label} 관련 이미지가 없습니다.")
            break

        image_path, label_path = random.choice(labeled_image_label_dirs)
        image = Image.open(image_path)
        width, height = image.size

        with open(label_path, "r") as f:
            lines = f.readlines()

        success = False

        for _ in range(attempt_limit):
            crop_scale = random.uniform(0.5, 0.95)
            crop_width = int(width * crop_scale)
            crop_height = int(height * crop_scale)
            crop_x_min = random.randint(0, width - crop_width)
            crop_y_min = random.randint(0, height - crop_height)
            crop_x_max = crop_x_min + crop_width
            crop_y_max = crop_y_min + crop_height

            cropped_image = image.crop((crop_x_min, crop_y_min, crop_x_max, crop_y_max))
            flipped_image = cropped_image.transpose(Image.FLIP_LEFT_RIGHT)
            cropped_width, cropped_height = cropped_image.size

            included = False
            new_labels = []
            flipped_labels = []

            for line in lines:
                parts = line.strip().split()
                cls, xc, yc, bw, bh = parts[0], *map(float, parts[1:])
                if cls != label:
                    continue

                x_min = (xc - bw / 2) * width
                y_min = (yc - bh / 2) * height
                x_max = (xc + bw / 2) * width
                y_max = (yc + bh / 2) * height
                box_cx = (x_min + x_max) / 2
                box_cy = (y_min + y_max) / 2

                if crop_x_min < box_cx < crop_x_max and crop_y_min < box_cy < crop_y_max:
                    included = True
                    new_xc = (box_cx - crop_x_min) / cropped_width
                    new_yc = (box_cy - crop_y_min) / cropped_height
                    new_bw = (x_max - x_min) / cropped_width
                    new_bh = (y_max - y_min) / cropped_height
                    new_labels.append(f"{label} {new_xc:.6f} {new_yc:.6f} {new_bw:.6f} {new_bh:.6f}\n")
                    flip_xc = 1.0 - new_xc
                    flipped_labels.append(f"{label} {flip_xc:.6f} {new_yc:.6f} {new_bw:.6f} {new_bh:.6f}\n")

            if included:
                base_name = f"aug_label{label}_{augmented_count}"
                cropped_image.save(os.path.join(aug_image_dir, base_name + ".png"))
                with open(os.path.join(aug_label_dir, base_name + ".txt"), "w") as f:
                    f.writelines(new_labels)

                flipped_image.save(os.path.join(aug_image_dir, base_name + "_flip.png"))
                with open(os.path.join(aug_label_dir, base_name + "_flip.txt"), "w") as f:
                    f.writelines(flipped_labels)

                augmented_count += 1
                success = True
                break

        if not success:
            print(f"❌ 라벨 {label} 증강 실패 (20회 시도 내 포함 불가)")


def compute_iter_num_by_label(label_dirs, top_k=3):
    label_counts = {}

    for label_dir in label_dirs:
        for filename in os.listdir(label_dir):
            if filename.endswith(".txt"):
                with open(os.path.join(label_dir, filename), "r") as file:
                    for line in file:
                        label = line.strip().split()[0]
                        label_counts[label] = label_counts.get(label, 0) + 1

    sorted_counts = sorted(label_counts.values(), reverse=True)
    if len(sorted_counts) < top_k:
        raise ValueError(f"라벨 개수가 {top_k}개 미만입니다.")
    third_largest_count = sorted_counts[top_k - 1]

    label_to_iter = {}
    for label, count in label_counts.items():
        iter_num = max(0, third_largest_count - count)
        label_to_iter[label] = iter_num

    return label_to_iter



import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random

def visualize_augmented_samples(aug_image_dir, aug_label_dir, num_samples=4):
    """
    증강된 이미지와 라벨을 시각화합니다.

    Parameters:
        aug_image_dir (str): 증강된 이미지 디렉토리
        aug_label_dir (str): 증강된 라벨 디렉토리
        num_samples (int): 시각화할 이미지 개수
    """
    label_files = [f for f in os.listdir(aug_label_dir) if f.endswith(".txt")]
    selected_files = random.sample(label_files, min(len(label_files), num_samples))

    for label_file in selected_files:
        image_file = label_file.replace(".txt", ".png")
        image_path = os.path.join(aug_image_dir, image_file)
        label_path = os.path.join(aug_label_dir, label_file)

        if not os.path.exists(image_path):
            continue

        image = Image.open(image_path)
        width, height = image.size

        fig, ax = plt.subplots(1)
        ax.imshow(image)

        with open(label_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                cls, xc, yc, bw, bh = parts[0], *map(float, parts[1:])
                x = (xc - bw / 2) * width
                y = (yc - bh / 2) * height
                w = bw * width
                h = bh * height

                rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor='r', facecolor='none')
                ax.add_patch(rect)
                ax.text(x, y - 5, f"Label {cls}", color='red', fontsize=10)

        plt.title(f"Augmented: {image_file}")
        plt.axis('off')
        plt.tight_layout()
        plt.show()


from collections import defaultdict

def print_augmentation_summary(aug_label_dir):
    """
    증강된 라벨 디렉토리를 스캔하여, 라벨별 생성된 샘플 수를 출력합니다.

    Parameters:
        aug_label_dir (str): 증강된 YOLO 라벨(.txt) 파일이 저장된 디렉토리
    """
    label_counts = defaultdict(int)

    for filename in os.listdir(aug_label_dir):
        if not filename.endswith(".txt"):
            continue
        with open(os.path.join(aug_label_dir, filename), "r") as f:
            for line in f:
                label = line.strip().split()[0]
                label_counts[label] += 1

    print("\n📊 증강 결과 요약:")
    print("-" * 30)
    for label in sorted(label_counts, key=int):
        print(f"Label {label}: {label_counts[label]}개 생성")
    print("-" * 30)



if __name__ == "__main__":
    # 대상 디렉토리
    label_dirs = ["data/yolo/labels/train", "data/yolo/labels/val"]
    image_dirs = ["data/yolo/images/train", "data/yolo/images/val"]
    aug_image_dir = "data/yolo/augmented/images"
    aug_label_dir = "data/yolo/augmented/labels"

    # 라벨별 증강 횟수 계산
    label_to_iter = compute_iter_num_by_label(label_dirs)

    # 전체 라벨에 대한 이미지/라벨 경로 캐시
    label_to_entries = collect_all_label_entries(label_dirs, image_dirs)

    for label, iter_num in label_to_iter.items():
        if iter_num <= 0:
            continue  # 증강 불필요
        if label not in label_to_entries:
            print(f"⚠️ 라벨 {label}에 대한 데이터 없음.")
            continue
        augment_single_label_random_crop(
            label, label_to_entries[label], iter_num, aug_image_dir, aug_label_dir
        )


    # 증강 완료 후 요약 출력
    print_augmentation_summary(aug_label_dir)

    # 증강된 이미지 몇 개 시각화
    visualize_augmented_samples(aug_image_dir, aug_label_dir, num_samples=4)