# 🔥 Multimodal Boiling Regime Classification (Audio + Video + Scalar Features)

> **Novel Research Project** conducted at the **Indian Institute of Science (IISc), Bangalore**  
> **Under the guidance of Prof. [Professor's Full Name]**  
> Multimodal deep learning combining **audio**, **video**, and **scalar process features** to classify **boiling regimes**. Work prepared as a **research journal** and experimental study.

---

## 📝 One-Line Summary
We develop and evaluate two state-of-the-art models for boiling regime classification:
1. **Audio → Spectrogram → EfficientNet + Tabular Fusion** (CNN-based audio embedding fused with scalar features).  
2. **Video → Frame Encoder (ResNet18) + Transformer Encoder** (temporal modeling of frame sequences).

Both models were developed as part of a novel research study at IISc and are designed for robust industrial and lab-scale monitoring of boiling dynamics.

---

## 🚀 Highlights / Contributions
- Novel multimodal integration of acoustic, visual, and scalar features for boiling regime detection.
- Data augmentation strategies for spectrograms (time/freq masks, noise) and for video frames (rotation, color jitter, blur, flips).
- Transformer-based temporal modeling for video; EfficientNet transfer-learning for spectrograms.
- Class-imbalance handling via inverse-class-weighting (Keras) and weighted CrossEntropy (PyTorch).
- Prepared as a research manuscript with experimental evaluation and comparative results.

---

## 📂 Project structure (suggested)
```
multimodal-boiling-classification/
│
├── audio_model/
│   ├── data/                       # spectrogram images, raw audio (if available), metadata.xlsx
│   ├── preprocess_audio.py         # audio -> spectrogram, augmentation
│   ├── train_audio_tabular.py      # training script (Keras/TensorFlow)
│   ├── models.py                   # model definition (EfficientNet + tabular fusion)
│   ├── evaluate_audio.py           # evaluation & plotting
│   └── checkpoints/
│
├── video_model/
│   ├── frames_dataset/             # extracted frames per video (folder per video)
│   ├── preprocess_video.py         # frame extraction & augmentation
│   ├── train_video_transformer.py  # training script (PyTorch)
│   ├── models.py                   # ResNet frame encoder + Transformer classifier
│   ├── evaluate_video.py           # evaluation & plotting
│   └── checkpoints/
│
├── docs/
│   ├── manuscript_draft.pdf        # research journal draft (optional)
│   └── figures/
│
├── requirements.txt
├── README.md                       # <-- you are editing this file
└── LICENSE
```

---

## 🛠️ Installation
```bash
# Clone
git clone https://github.com/yourusername/multimodal-boiling-classification.git
cd multimodal-boiling-classification

# Create env and install
python -m venv venv
source venv/bin/activate   # macOS/Linux
# venv\Scripts\activate    # Windows

pip install -r requirements.txt
```

### Example `requirements.txt` (suggested)
```
numpy
pandas
scikit-learn
matplotlib
seaborn
tensorflow>=2.9
torch
torchvision
tqdm
Pillow
openpyxl
librosa
soundfile
transformers
```

---

## ▶️ Usage

### 1) Audio + Tabular Model (Keras / TensorFlow)
- Prepare metadata Excel/CSV with columns: `image_file_name` (path to spectrogram PNG/JPG), `Regime` (label), `T` (temperature), `We` (Weber number) or other scalar features.
- Put spectrogram images in a folder and update `metadata_path`.

Train:
```bash
python audio_model/train_audio_tabular.py   --metadata /path/to/metadata.xlsx   --spec-dir /path/to/spectrograms   --epochs 100   --batch-size 16
```

Evaluate / Predict:
```bash
python audio_model/evaluate_audio.py --checkpoint audio_model/checkpoints/best_model.h5
```

---

### 2) Video Transformer Model (PyTorch)
- Prepare frames as `frames_dataset/<label>/<video_id>/<frame_0001.jpg>` etc. Each `video_id` folder contains extracted frames for that video.
- Adjust `ROOT_DIR` in `video_model/train_video_transformer.py`.

Train:
```bash
python video_model/train_video_transformer.py   --frames_dir /path/to/frames_dataset   --epochs 16   --batch-size 4
```

Evaluate / Predict:
```bash
python video_model/evaluate_video.py --checkpoint video_model/checkpoints/video_transformer_final.pth
```

---

## 📈 Model Architectures (Mermaid diagrams)

### 1) Audio → Spectrogram → EfficientNet + Tabular Fusion
```mermaid
flowchart TD
  A[Raw Audio (.wav / .flac)] --> B[Spectrogram Generation]
  B --> C[Spec Augmentation]
  C --> D[Spec (64x128x3) Input]
  D --> E[EfficientNetB0 (pretrained, include_top=False)]
  E --> F[GlobalAveragePooling2D]
  F --> G[Dropout + Dense Embedding]

  subgraph TAB [Tabular (Scalar) Path]
    H[Scalar Features (T, We, ...)] --> I[StandardScaler]
    I --> J[Dense(32) → BatchNorm → Dense(16)]
  end

  G --> K[Fusion: Concatenate (Spec Embedding + Tabular Embedding)]
  J --> K
  K --> L[Dense(64) → Dropout]
  L --> M[Output Dense(num_classes, activation='softmax')]
  M --> N[Prediction: Boiling Regime]
```

---

### 2) Video → Frame Encoder (ResNet18) + Transformer Encoder
```mermaid
flowchart TD
  V[Input Video (.mp4)] --> FE[Frame Extraction / Sampling (max 200 frames)]
  FE --> Pre[Per-frame Preprocessing & Augmentation]
  Pre --> R[ResNet18 Frame Encoder (pretrained)]
  R --> Flatten[Project to 512-d per-frame embeddings]
  Flatten --> Pos[Add Positional Encoding]
  Pos --> T[Transformer Encoder (num_layers, num_heads)]
  T --> Q[Query token (learned) + MultiheadAttention Pooling]
  Q --> Class[LayerNorm -> Linear(num_classes)]
  Class --> Out[Softmax -> Boiling Regime Prediction]
```

---

## 🧪 Training & Experimental Details
- **Audio model**
  - Input size: spectrograms resized to (64, 128, 3).
  - Transfer learning: EfficientNetB0 pretrained on ImageNet (frozen then optionally unfrozen).
  - Optimizer: Adam, LR schedule with ReduceLROnPlateau, EarlyStopping with restore_best_weights.
  - Class imbalance: inverse-class-weighting.

- **Video model**
  - Frame size: 224×224 input to ResNet18.
  - Max frames per video: 200 (downsample/upsample to fixed length).
  - Transformer: embed_dim=512, num_heads=8, num_layers=2.
  - Loss: CrossEntropyLoss with class weights.
  - Augmentations: rotation, color jitter, blur, flips.

---

## ✅ Evaluation & Metrics
- Reported metrics: Accuracy, Precision, Recall, F1-score, Confusion Matrix.
- Use `classification_report` and `confusion_matrix` (scikit-learn) for per-class breakdown.
- Visualize training with loss/accuracy plots and confusion matrix heatmaps.

---

## 🧾 Results (fill in with your numbers)
- Audio + Tabular Fusion — **Accuracy:** `82%`  
- Video Transformer — **Accuracy:** `90%`  


---

## ♻️ Reproducibility Checklist
- [ ] Provide raw audio files and spectrogram generation script.
- [ ] Provide video files or extracted frame folders.
- [ ] Provide metadata spreadsheet with labels and scalar features.
- [ ] Provide random seed, training hyperparameters, and environment details (Python + package versions).
- [ ] Upload best model checkpoints and inference script.

---

## 🤝 Acknowledgements
This research was carried out at the **Indian Institute of Science (IISc), Bangalore**, under the guidance of **Prof. [Professor's Full Name]**. The project benefited from experimental infrastructure, lab recordings, and insightful domain expertise.

---

## 📜 License
Include your preferred license (e.g., MIT, Apache-2.0). Example:
```
MIT License
```

---

## 🔁 Contact & Citation
If you reuse part of this work or want collaboration, please contact:  
**[Your Name]** — email: `your.email@domain.edu`  
When citing this work, please reference the project repo and the IISc guidance.

---

## Appendix: Helpful Commands

- Convert audio to mel-spectrograms (example using `librosa`):
```python
import librosa, librosa.display, numpy as np
y, sr = librosa.load('audio.wav', sr=22050)
S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
S_db = librosa.power_to_db(S, ref=np.max)
# save S_db as image
```

- Extract frames from video:
```bash
ffmpeg -i input_video.mp4 -vf "fps=10" frames_dir/frame_%05d.jpg
```
#   M u l t i m o d a l - B o i l i n g - R e g i m e - C l a s s i f i c a t i o n 
 
 