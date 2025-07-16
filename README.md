# ⚽ YOLOv11-Based Football Player Detection & Re-Identification

This repository presents a real-time system for detecting and re-identifying football players in broadcast match footage using a custom-trained YOLOv11 model. It integrates object detection, player re-identification, and tracking into a unified pipeline to enhance game analysis and visualization.

---

## 📽️ Demo Showcase

### 🔹 Pretrained YOLOv11 Inference

> **Demo:** Football player detection using default YOLOv11 weights.
>  [Watch Inference Demo Video](https://drive.google.com/file/d/1gEpO-yrgygrApMtOI-agtseWdIj6A1cs/view?usp=sharing)
> ✅ Detects players and objects with basic labels and bounding boxes.

### 🔹 Custom YOLOv11 Inference & Tracking

> **Demo:** Inference using a fine-tuned YOLOv11 model on annotated football datasets.
> 🎥 [Watch Re-ID & Tracking Demo Video](https://drive.google.com/file/d/1gAK4_7pYo_oJUTzn3kzyZkh_4w-3rnQe/view?usp=sharing)
> ✅ Outputs consistent player IDs and frame-to-frame tracking.

---

## 📁 Project Overview

This project enables:

* ⚡ **Real-Time Detection** of players, referees, footballs, and other objects
* 🔄 **Player Re-Identification** across video frames using tracking logic
* 🎥 **Video Generation** with annotated bounding boxes and consistent IDs
* 💾 **Result Export** as `.pkl` files for downstream analytics

---

## 📂 Directory Structure

```
.
├── input_video/               # Source input video(s)
├── model/                     # Model weights (e.g., best.pt or pretrained)
│   └── model_link             # Text file with model download URL
├── notebook_for_training_the_model/
│   └── Football_Analysis_System (1).ipynb  # Training notebook
├── output_videos/sf/          # Output annotated videos
├── tracker_stubs/             # Pickle files of tracking metadata
├── trackers/                  # Custom tracking logic
├── utils/                     # Utility scripts
├── yolo_inference.py          # Inference with pretrained YOLOv11
├── main.py                    # Main pipeline using trained model
├── requirements.txt           # Required Python packages
└── README.md                  # Project documentation
```

---

## ⚙️ Setup Instructions

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/Sanjuu2004/YOLOv11-Model-for-Football-Player-Reidentification
cd YOLOv11-Model-for-Football-Player-Reidentification
```

### 2️⃣ Create & Activate a Virtual Environment (Recommended)

#### On Windows

```bash
python -m venv venv
venv\Scripts\activate
```

#### On Unix/macOS

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

> ⚠️ **Known Issue:**
> If you encounter an error related to `torch` installation:
>
> ```
> Could not find a version that satisfies the requirement torch>=1.7.0
> ```
>
> ✅ **Fix:**
>
> * Make sure you're using **Python 3.8–3.11 (64-bit)**
> * Upgrade pip:
>
>   ```bash
>   python -m pip install --upgrade pip
>   ```
> * Or install dependencies manually:
>
>   ```bash
>   pip install opencv-python torch torchvision numpy pillow scikit-learn ultralytics
>   ```

---

## 🧠 Model Setup

### 🔹 Option 1: Run with Pretrained YOLOv11

```bash
python yolo_inference.py
```

**Output:**

* Players and objects are detected with basic bounding boxes.
* No player identity consistency across frames.

---

### 🟢 Option 2: Run with Custom Trained YOLOv11

1. Train a model on your custom annotated dataset
2. Download your `best.pt` weights and place them in the `model/` folder

   > 📥 **Download Link:** Provided in `model/model_link`
3. Then run:

```bash
python main.py
```

**Output:**

* Players, referees, and footballs are accurately detected.
* Players are re-identified consistently across frames.
* Tracking metadata is saved as `tracker_stubs/player_detection.pkl`.

---

## ▶️ Running the Pipeline

### 1. Detection Using YOLOv11

```python
from ultralytics import YOLO

model = YOLO("yolo11l.pt")  # Or use "model/best.pt" for custom model
results = model.predict(source="input_video/15sec_input_720p.mp4", save=True)
```

* 🔄 Output saved to: `runs/detect/predict/`

---

### 2. Detection + Tracking + Re-ID

```python
from ultralytics import YOLO

model = YOLO("model/best.pt")
results = model.track(source="input_video/15sec_input_720p.mp4", save=True, persist=True)
```

* 🎯 Annotated output saved to: `runs/track/predict/`
* 🧾 Re-ID metadata stored in: `tracker_stubs/player_detection.pkl`


## 🧾 Dependencies

* Python 3.8+
* [Ultralytics](https://github.com/ultralytics/ultralytics) (YOLOv8 base)
* OpenCV
* Torch + TorchVision
* Torchreid (for re-identification)
* NumPy, Pillow, scikit-learn


## 🧠 Future Enhancements

* ⏱️ Multi-camera cross-view player tracking
* 📊 Heatmap and tactical analytics using tracking data
* 🧠 Integration with pose estimation (e.g., OpenPose, Mediapipe)
