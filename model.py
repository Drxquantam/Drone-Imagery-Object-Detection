
import os
os.environ["WANDB_DISABLED"] = "true"

# ---------- install ----------
!pip uninstall -y wandb >/dev/null 2>&1

# ---------- imports ----------
import zipfile
import random
import shutil
import yaml
from pathlib import Path
from collections import Counter, defaultdict

import cv2
import numpy as np
from PIL import Image
import torch
from google.colab import files
from ultralytics import YOLO

# ---------- settings ----------
MODEL_NAME = "yolov8m.pt"
EPOCHS = 50
IMG_SIZE = 896
BATCH = 4              
VAL_RATIO = 0.2
SEED = 42

MAX_OVERSAMPLE_PER_IMAGE = 5
RARE_CLASS_FREQ_THRESHOLD = 0.12

MIN_NORM_W = 0.006
MIN_NORM_H = 0.006

PRED_CONF = 0.30
PRED_IOU = 0.60

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


uploaded = files.upload()
zip_path = list(uploaded.keys())[0]
print("Uploaded:", zip_path)


extract_dir = Path("/content/uploaded_dataset")
if extract_dir.exists():
    shutil.rmtree(extract_dir)
extract_dir.mkdir(parents=True, exist_ok=True)

with zipfile.ZipFile(zip_path, "r") as zip_ref:
    zip_ref.extractall(extract_dir)

print("ZIP extracted to:", extract_dir)


def find_root(base_dir):
    base_dir = Path(base_dir)
    if (base_dir / "images").exists() and (base_dir / "annotations").exists():
        return base_dir
    for p in base_dir.rglob("*"):
        if p.is_dir() and (p / "images").exists() and (p / "annotations").exists():
            return p
    return None

ROOT = find_root(extract_dir)
assert ROOT is not None, "Could not find dataset root containing images/ and annotations/"

img_dir = ROOT / "images"
ann_dir = ROOT / "annotations"

print("Detected ROOT:", ROOT)
print("Images folder:", img_dir)
print("Annotations folder:", ann_dir)


image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
images = sorted([p for p in img_dir.iterdir() if p.suffix.lower() in image_exts])
assert len(images) > 0, f"No images found in {img_dir}"
print("Total images found:", len(images))


BASE_DATASET_DIR = Path("/content/yolo_dataset_baseline")
NOVEL_DATASET_DIR = Path("/content/yolo_dataset_novelty")

for d in [BASE_DATASET_DIR, NOVEL_DATASET_DIR]:
    if d.exists():
        shutil.rmtree(d)
    for split in ["train", "val"]:
        (d / "images" / split).mkdir(parents=True, exist_ok=True)
        (d / "labels" / split).mkdir(parents=True, exist_ok=True)


random.shuffle(images)
val_count = max(1, int(len(images) * VAL_RATIO))
val_set = set(images[:val_count])

def clamp(v, low, high):
    return max(low, min(v, high))


def apply_atem_bgr(img_bgr):
   
    gamma = 1.45
    inv_gamma = 1.0 / gamma
    table = np.array([(i / 255.0) ** inv_gamma * 255 for i in np.arange(256)]).astype("uint8")
    gamma_img = cv2.LUT(img_bgr, table)

    
    lab = cv2.cvtColor(gamma_img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.5, tileGridSize=(8, 8))
    l2 = clahe.apply(l)
    lab2 = cv2.merge((l2, a, b))
    clahe_img = cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)

    # unsharp masking
    blur = cv2.GaussianBlur(clahe_img, (0, 0), sigmaX=1.1, sigmaY=1.1)
    sharp = cv2.addWeighted(clahe_img, 1.75, blur, -0.75, 0)

    # slight denoising to reduce enhancement noise
    out = cv2.bilateralFilter(sharp, d=5, sigmaColor=35, sigmaSpace=35)

    return out


def convert_visdrone_annotation(src_txt, dst_txt, img_path):
    img = Image.open(img_path)
    img_w, img_h = img.size

    converted = []
    classes_found = set()
    bad_lines = []

    with open(src_txt, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    for idx, line in enumerate(lines, start=1):
        line = line.strip()
        if not line:
            continue

        parts = [x.strip() for x in line.split(",")]
        if len(parts) != 8:
            bad_lines.append((idx, line))
            continue

        try:
            x, y, w, h, score, category, truncation, occlusion = map(float, parts)
        except:
            bad_lines.append((idx, line))
            continue

        category = int(category)

        # ignore category 0
        if category == 0:
            continue

        x = clamp(x, 0, img_w - 1)
        y = clamp(y, 0, img_h - 1)
        w = clamp(w, 1, img_w)
        h = clamp(h, 1, img_h)

        x2 = min(x + w, img_w)
        y2 = min(y + h, img_h)
        bw = x2 - x
        bh = y2 - y

        if bw <= 1 or bh <= 1:
            bad_lines.append((idx, line))
            continue

        xc = (x + bw / 2.0) / img_w
        yc = (y + bh / 2.0) / img_h
        bw = bw / img_w
        bh = bh / img_h

        # stricter filtering of harmful tiny boxes
        if bw < MIN_NORM_W or bh < MIN_NORM_H:
            continue

        converted.append(f"{category} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")
        classes_found.add(category)

    with open(dst_txt, "w") as f:
        f.write("\n".join(converted))

    return converted, classes_found, bad_lines

all_classes = set()
missing_labels = []
bad_files = []
empty_after_conversion = []
train_image_classes = defaultdict(set)


for img_path in images:
    split = "val" if img_path in val_set else "train"
    src_txt = ann_dir / f"{img_path.stem}.txt"

    if not src_txt.exists():
        missing_labels.append(img_path.name)
        continue

    
    dst_img_base = BASE_DATASET_DIR / "images" / split / img_path.name
    shutil.copy2(img_path, dst_img_base)

   
    dst_img_novel = NOVEL_DATASET_DIR / "images" / split / img_path.name
    img_bgr = cv2.imread(str(img_path))
    if img_bgr is None:
        missing_labels.append(img_path.name)
        continue
    novel_bgr = apply_atem_bgr(img_bgr)
    cv2.imwrite(str(dst_img_novel), novel_bgr)

    # labels
    dst_txt_base = BASE_DATASET_DIR / "labels" / split / f"{img_path.stem}.txt"
    dst_txt_novel = NOVEL_DATASET_DIR / "labels" / split / f"{img_path.stem}.txt"

    converted, classes_found, bad_lines = convert_visdrone_annotation(src_txt, dst_txt_base, img_path)
    shutil.copy2(dst_txt_base, dst_txt_novel)

    if len(converted) == 0:
        empty_after_conversion.append(src_txt.name)

    all_classes.update(classes_found)

    if split == "train":
        train_image_classes[img_path.name] = set(classes_found)

    if bad_lines:
        bad_files.append((src_txt.name, bad_lines[:5]))

print("Missing labels:", len(missing_labels))
print("Empty after conversion:", len(empty_after_conversion))
print("Files with bad lines:", len(bad_files))
assert len(all_classes) > 0, "No valid classes found after conversion."

# ---------- remap classes ----------
sorted_classes = sorted(all_classes)
class_map = {old: new for new, old in enumerate(sorted_classes)}

def remap_labels(dataset_dir):
    for split in ["train", "val"]:
        label_folder = dataset_dir / "labels" / split
        for txt_path in label_folder.glob("*.txt"):
            new_lines = []
            with open(txt_path, "r") as f:
                for line in f:
                    if not line.strip():
                        continue
                    parts = line.strip().split()
                    old_cls = int(parts[0])
                    if old_cls not in class_map:
                        continue
                    parts[0] = str(class_map[old_cls])
                    new_lines.append(" ".join(parts))
            with open(txt_path, "w") as f:
                f.write("\n".join(new_lines))

remap_labels(BASE_DATASET_DIR)
remap_labels(NOVEL_DATASET_DIR)

nc = len(sorted_classes)
names = [f"class_{i}" for i in range(nc)]

print("Original class ids:", sorted_classes)
print("Class remap:", class_map)
print("Number of classes:", nc)


train_image_classes_remap = {}
for img_name, cls_set in train_image_classes.items():
    remapped = set()
    for c in cls_set:
        if c in class_map:
            remapped.add(class_map[c])
    train_image_classes_remap[img_name] = remapped


class_counts = Counter()
for cls_set in train_image_classes_remap.values():
    for c in cls_set:
        class_counts[c] += 1

total_train_images = max(len(train_image_classes_remap), 1)
class_freq = {c: class_counts[c] / total_train_images for c in range(nc)}
rare_classes = {c for c, f in class_freq.items() if f < RARE_CLASS_FREQ_THRESHOLD}

print("Class image-frequency:", class_freq)
print("Rare classes selected:", rare_classes)

oversampled = 0
for img_name, cls_set in train_image_classes_remap.items():
    rare_in_image = list(cls_set.intersection(rare_classes))
    if len(rare_in_image) == 0:
        continue

    repeat_n = min(MAX_OVERSAMPLE_PER_IMAGE, len(rare_in_image) + 1)

    src_img = NOVEL_DATASET_DIR / "images" / "train" / img_name
    src_lbl = NOVEL_DATASET_DIR / "labels" / "train" / f"{Path(img_name).stem}.txt"

    if not src_img.exists() or not src_lbl.exists():
        continue

    for k in range(1, repeat_n + 1):
        new_img = NOVEL_DATASET_DIR / "images" / "train" / f"{Path(img_name).stem}_dup{k}{Path(img_name).suffix}"
        new_lbl = NOVEL_DATASET_DIR / "labels" / "train" / f"{Path(img_name).stem}_dup{k}.txt"
        shutil.copy2(src_img, new_img)
        shutil.copy2(src_lbl, new_lbl)
        oversampled += 1

print("Oversampled train images added:", oversampled)

# ---------- write YAML ----------
def write_data_yaml(dataset_dir, yaml_path):
    data_yaml = {
        "path": str(dataset_dir),
        "train": "images/train",
        "val": "images/val",
        "nc": nc,
        "names": names
    }
    with open(yaml_path, "w") as f:
        yaml.safe_dump(data_yaml, f, sort_keys=False)

base_yaml = BASE_DATASET_DIR / "data.yaml"
novel_yaml = NOVEL_DATASET_DIR / "data.yaml"

write_data_yaml(BASE_DATASET_DIR, base_yaml)
write_data_yaml(NOVEL_DATASET_DIR, novel_yaml)


device_value = 0 if torch.cuda.is_available() else "cpu"
print("Using device:", device_value)


train_kwargs = dict(
    epochs=EPOCHS,
    imgsz=IMG_SIZE,
    batch=BATCH,
    device=device_value,
    optimizer="AdamW",
    lr0=0.0004,
    lrf=0.01,
    weight_decay=0.0005,
    warmup_epochs=5.0,
    workers=2,
    cache=True,
    amp=True,
    pretrained=True,
    close_mosaic=20,
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    degrees=0.0,
    translate=0.08,
    scale=0.70,
    shear=0.0,
    perspective=0.0,
    flipud=0.0,
    fliplr=0.5,
    mosaic=1.0,
    mixup=0.20,
    copy_paste=0.0,
    cls=1.0,
    box=7.5,
    dfl=1.5,
    patience=40,
    plots=True,
    verbose=True
)

# ---------- train baseline ----------
print("\n========== TRAINING BASELINE ==========\n")
baseline_model = YOLO(MODEL_NAME)
baseline_model.train(
    data=str(base_yaml),
    project="/content/runs",
    name="baseline_yolov8m_maxperf",
    **train_kwargs
)

baseline_best = "/content/runs/baseline_yolov8m_maxperf/weights/best.pt"
baseline_model = YOLO(baseline_best)

print("\n========== VALIDATING BASELINE ==========\n")
baseline_metrics = baseline_model.val(
    data=str(base_yaml),
    imgsz=IMG_SIZE,
    conf=0.001,
    iou=0.60,
    split="val"
)
print("Baseline metrics:", baseline_metrics)

# ---------- train novelty ----------
print("\n========== TRAINING NOVELTY ==========\n")
novel_model = YOLO(MODEL_NAME)
novel_model.train(
    data=str(novel_yaml),
    project="/content/runs",
    name="novelty_atem_oversample_yolov8m_maxperf",
    **train_kwargs
)

novel_best = "/content/runs/novelty_atem_oversample_yolov8m_maxperf/weights/best.pt"
novel_model = YOLO(novel_best)

print("\n========== VALIDATING NOVELTY ==========\n")
novel_metrics = novel_model.val(
    data=str(novel_yaml),
    imgsz=IMG_SIZE,
    conf=0.001,
    iou=0.60,
    split="val"
)
print("Novelty metrics:", novel_metrics)

# ---------- clean prediction images ----------
base_val_images = sorted(list((BASE_DATASET_DIR / "images" / "val").glob("*")))
novel_val_images = sorted(list((NOVEL_DATASET_DIR / "images" / "val").glob("*")))

if len(base_val_images) > 0:
    baseline_model.predict(
        source=str(base_val_images[0]),
        imgsz=IMG_SIZE,
        conf=PRED_CONF,
        iou=PRED_IOU,
        save=True,
        project="/content/runs",
        name="baseline_pred_clean_maxperf",
        line_width=2
    )

if len(novel_val_images) > 0:
    novel_model.predict(
        source=str(novel_val_images[0]),
        imgsz=IMG_SIZE,
        conf=PRED_CONF,
        iou=PRED_IOU,
        save=True,
        project="/content/runs",
        name="novelty_pred_clean_maxperf",
        line_width=2
    )

print("\n========== DONE ==========")
print("Baseline train folder: /content/runs/baseline_yolov8m_maxperf")
print("Novelty train folder: /content/runs/novelty_atem_oversample_yolov8m_maxperf")
print("Baseline clean prediction: /content/runs/baseline_pred_clean_maxperf")
print("Novelty clean prediction: /content/runs/novelty_pred_clean_maxperf")