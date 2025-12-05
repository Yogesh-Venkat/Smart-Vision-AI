---
license: mit
title: SmartVision AI
sdk: streamlit
emoji: 🚀
colorFrom: red
colorTo: red
short_description: Multi-domain smart object detection and classification syste
---

# SmartVision AI – Complete Vision Pipeline (YOLOv8 + CNN Classifiers + Streamlit Dashboard)

SmartVision AI is a fully integrated **Computer Vision system** that combines:

- **Object Detection** using YOLOv8  
- **Image Classification** using 4 deep-learning models:  
  **VGG16**, **ResNet50**, **MobileNetV2**, **EfficientNetB0**  
- A complete **Streamlit-based Dashboard** for inference, comparison, metrics visualization, and webcam snapshots  
- A modified dataset built on a **25‑class COCO subset**

This README explains setup, architecture, training, deployment, and usage.

---

## 🚀 Features

### ✅ 1. Image Classification (4 Models)
Each model is fine‑tuned on your custom 25‑class dataset:
- **VGG16**
- **ResNet50**
- **MobileNetV2**
- **EfficientNetB0**

Outputs:
- Top‑1 class prediction  
- Top‑5 predictions  
- Class probabilities  

---

### 🎯 2. Object Detection – YOLOv8s
YOLO detects multiple objects in images or webcam snapshots.

Features:
- Bounding boxes  
- Confidence scores  
- Optional classification verification using ResNet50  
- Annotated images saved automatically  

---

### 🔗 3. Integrated Classification + Detection Pipeline
For each YOLO‑detected box:
1. Crop region  
2. Classify using chosen CNN model  
3. Display YOLO label + classifier label  
4. Draw combined annotated results  

---

### 📊 4. Metrics Dashboard
Displays:
- Accuracy  
- Weighted F1 score  
- Top‑5 accuracy  
- Images per second  
- Model size  
- YOLOv8 mAP scores  
- Confusion matrices  
- Comparison bar charts  

---

### 📷 5. Webcam Snapshot Detection
Take a photo via webcam → YOLO detection → annotated results.

---

## 📁 Project Structure

```
SmartVision_AI/
│
├── app.py                     # Main Streamlit App
├── saved_models/              # Trained weights (VGG16, ResNet, MobileNetV2, EfficientNet)
├── yolo_runs/                 # YOLOv8 training folder
├── smartvision_dataset/       # 25-class dataset
│   ├── classification/
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   └── detection/             # Labels + images for YOLOv8
│
├── smartvision_metrics/       # Accuracy, F1, confusion matrices
├── scripts/                   # Weight converters, training scripts
├── inference_outputs/         # Annotated results
├── requirements.txt           
└── README.md                  
```

---

## ⚙️ Installation

### 1️⃣ Clone Repository

```
git clone https://github.com/<your-username>/SmartVision_AI.git
cd SmartVision_AI
```

### 2️⃣ Install Dependencies

```
pip install -r requirements.txt
```

### 3️⃣ Install YOLOv8 (Ultralytics)

```
pip install ultralytics
```

---

## ▶️ Run Streamlit App

```
streamlit run app.py
```

App will open at:

```
http://localhost:8501
```

---

## 🏋️ Training Workflow

### 1️⃣ Classification Models
Each model has:
- Stage 1 → Train head with frozen backbone  
- Stage 2 → Unfreeze top layers + fine‑tune  

Scripts:
```
scripts/train_mobilenetv2.py
scripts/train_efficientnetb0.py
scripts/train_resnet50.py
scripts/train_vgg16.py
```

### 2️⃣ YOLO Training

```
yolo task=detect mode=train model=yolov8s.pt data=data.yaml epochs=50 imgsz=640
```

Outputs saved to:
```
yolo_runs/smartvision_yolov8s/
```

---

## 🧪 Supported Classes (25 COCO Classes)

```
airplane, bed, bench, bicycle, bird, bottle, bowl,
bus, cake, car, cat, chair, couch, cow, cup, dog,
elephant, horse, motorcycle, person, pizza, potted plant,
stop sign, traffic light, truck
```

---

## 🧰 Deployment on Hugging Face Spaces

You can deploy using **Streamlit SDK**.

### Steps:
1. Create public repository on GitHub  
2. Push project files  
3. Create new Hugging Face Space → select **Streamlit**  
4. Connect GitHub repo  
5. Add `requirements.txt`  
6. Enable **GPU** for YOLO (optional)  
7. Deploy 🚀  

---

## 🧾 requirements.txt Example

```
streamlit
tensorflow==2.13.0
ultralytics
numpy
pandas
Pillow
matplotlib
scikit-learn
opencv-python-headless
```

---

## 📄 .gitignore Example

```
saved_models/
*.h5
*.pt
*.weights.h5
yolo_runs/
smartvision_metrics/
inference_outputs/
__pycache__/
*.pyc
.DS_Store
env/
```

---

## 🙋 Developer

**SmartVision AI Project**  
Yogesh Kumar V
M.Sc. Seed Science & Technology (TNAU)  
Passion: AI, Computer Vision, Agribusiness Technology  

---

## 🏁 Conclusion

SmartVision AI integrates:
- Multi‑model classification  
- YOLO detection  
- Streamlit visualization  
- Full evaluation suite  

Perfect for:
- Research  
- Demonstrations  
- CV/AI portfolio  
- Real‑world image understanding  

---

Enjoy using SmartVision AI! 🚀🧠
