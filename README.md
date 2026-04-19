# ATEM-Enhanced Small Object Detection in Drone Imagery

ATEM-Enhanced Small Object Detection is a reproducible pipeline for improving small object detection in aerial images using YOLOv8m, Adaptive Tone Enhancement (ATEM), and rare-class guided oversampling.

## Overview

Detecting small objects in drone imagery is difficult due to low resolution, poor contrast, and class imbalance. This project focuses on improving detection performance through data-centric enhancements rather than modifying the model architecture.

The pipeline improves:

- Visibility of small objects
- Training distribution across classes
- Label quality by removing noisy annotations

The approach is practical, lightweight, and easy to reproduce.

## Method

### Pipeline

```
Raw Dataset (VisDrone)
        ↓
Annotation Conversion (to YOLO format)
        ↓
Tiny Box Filtering
        ↓
ATEM Image Enhancement
        ↓
Rare-Class Guided Oversampling
        ↓
YOLOv8m Training (AdamW + Augmentations)
        ↓
Evaluation (mAP, Precision, Recall)
```

### Adaptive Tone Enhancement Module (ATEM)

Enhances visibility of weak and distant objects using:

- Gamma correction (brightens dark regions)
- CLAHE on luminance channel (improves local contrast)
- Unsharp masking (sharpens edges)
- Bilateral filtering (reduces noise while preserving structure)

### Tiny Box Filtering

Very small bounding boxes are removed during preprocessing:

```
width < 0.006 or height < 0.006 → discard
```

This reduces noisy supervision and stabilizes training.

### Rare-Class Guided Oversampling

Instead of naive duplication, the method:

- Computes image-level class frequency
- Marks rare classes (frequency < 0.12)
- Duplicates images containing rare classes (up to 5 times)

This increases exposure to underrepresented classes without altering validation data.

## Results

| Model                             | Precision | Recall | mAP@0.5 |
|----------------------------------|----------|--------|---------|
| YOLOv8m (Baseline)               | 0.68     | 0.34   | 0.324   |
| ATEM + Oversampling + YOLOv8m    | 0.75     | 0.45   | 0.426   |

The proposed approach improves mAP@0.5 by 0.102, demonstrating the effectiveness of improving input quality and training distribution.

## Dataset

Dataset used: VisDrone

Characteristics:

- High object density
- Large scale variation
- Significant class imbalance

## Training Configuration

```
Model: YOLOv8m
Epochs: 50
Image size: 896
Batch size: 4
Optimizer: AdamW
Learning rate: 0.0004
Weight decay: 0.0005
Validation split: 20%
```

Augmentations include mosaic, mixup, flipping, scaling, and HSV adjustments.

## Implementation Details

The code implements a complete pipeline:

- Automatic dataset extraction from ZIP
- Conversion from VisDrone annotations to YOLO format
- Class ID remapping
- ATEM-based image preprocessing
- Rare-class oversampling on training split
- Training and validation using Ultralytics YOLOv8
- Generation of evaluation plots and prediction outputs

## Installation

```bash
git clone https://github.com/Drxquantam/Drone-Imagery-Object-Detection
cd Drone-Imagery-Object-Detection

pip install -r requirements.txt
```

## Usage

Run training pipeline:

```bash
python model.py
```

## Outputs

```
runs/
 ├── baseline_yolov8m_maxperf/
 ├── novelty_atem_oversample_yolov8m_maxperf/
 ├── baseline_pred_clean_maxperf/
 ├── novelty_pred_clean_maxperf/
```

## Inference

```python
model.predict(
    source="image.jpg",
    conf=0.3,
    iou=0.6
)
```

## Key Takeaways

- Performance can be improved significantly without changing model architecture
- Enhancing input quality is critical for small object detection
- Addressing class imbalance improves generalization
- The pipeline is modular and easy to extend

## Contributing

Contributions are welcome!
If you'd like to improve this project, feel free to:

* Fork the repository
* Create a new branch
* Submit a pull request

---

## Authors

* **Dhruv Rai**
* **Krish Bansal**

Department of Computer Science and Engineering,
Netaji Subhas University of Technology, New Delhi
