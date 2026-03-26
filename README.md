# 🌿 Plant Identification & Benefit Prediction System

> **Academic Internship Project** — Machine Learning with Transfer Learning (MobileNetV2) + Streamlit Web UI

---

## 📁 Project Structure

```
PlantML/
├── app.py                  # Streamlit web application (main UI)
├── train.py                # Model training script (MobileNetV2 transfer learning)
├── utils.py                # Preprocessing, plant database, prediction helpers
├── download_dataset.py     # Sample dataset downloader (Wikimedia Commons)
├── requirements.txt        # Python dependencies
├── dataset/                # (Created by download_dataset.py or manually)
│   ├── train/
│   │   ├── Aloe_Vera/
│   │   ├── Banana/
│   │   ├── Basil/
│   │   ├── Mango/
│   │   ├── Neem/
│   │   ├── Rose/
│   │   ├── Tulsi/
│   │   ├── Turmeric/
│   │   ├── Almond/
│   │   └── Papaya/
│   └── val/               # (same folder structure as train/)
├── model/                  # (Created by train.py)
│   ├── plant_model.h5      # Saved trained model
│   ├── class_names.json    # Class order used during training
│   └── training_curves.png # Accuracy/loss plots
└── sample_images/          # Put test images here to try predictions
```

---

## ⚡ Quick Start

### 1. Clone / Navigate to the Project

```bash
cd PlantML
```

### 2. Create and Activate a Virtual Environment (Recommended)

```bash
python -m venv venv
source venv/bin/activate        # macOS / Linux
# venv\Scripts\activate         # Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 📥 Dataset Setup

### Option A — Sample Dataset (Quick Demo, ~2 minutes)

Downloads small freely-licensed images from Wikimedia Commons:

```bash
python download_dataset.py
```

This creates `dataset/train/` and `dataset/val/` with ~10 images per class (duplicated for demo purposes).

> ⚠️ Sample dataset accuracy will be LOW. Use Option B for a real model.

### Option B — PlantVillage Dataset (Recommended, ~2 GB)

```bash
pip install kaggle
# Place your kaggle.json API key in ~/.kaggle/
kaggle datasets download -d emmarex/plantdisease
unzip plantdisease.zip -d dataset/
```

Then rename/reorganise folders to match the 10 class names in `utils.py`:
`Aloe_Vera`, `Banana`, `Basil`, `Mango`, `Neem`, `Rose`, `Tulsi`, `Turmeric`, `Almond`, `Papaya`

---

## 🏋️ Model Training

```bash
python train.py
```

**Optional arguments:**

| Argument        | Default | Description                          |
|----------------|---------|--------------------------------------|
| `--data_dir`   | dataset | Path to the dataset root folder      |
| `--epochs_head`| 10      | Epochs for Phase 1 (frozen base)     |
| `--epochs_fine`| 15      | Epochs for Phase 2 (fine-tuning)     |
| `--batch_size` | 32      | Mini-batch size                      |

**Example (custom settings):**

```bash
python train.py --data_dir dataset --epochs_head 15 --epochs_fine 20 --batch_size 16
```

**Training outputs:**
- `model/plant_model.h5` — Saved trained model
- `model/class_names.json` — Class label order
- `model/training_curves.png` — Accuracy & loss graphs

---

## 🖥️ Running the Web Application

```bash
streamlit run app.py
```

Then open [http://localhost:8501](http://localhost:8501) in your browser.

**Features:**
- 📤 Upload JPG / PNG / WEBP plant images
- 🔍 AI-powered plant identification with confidence score
- 📖 Scientific name, plant family, description
- ✨ Key benefits shown as visual pill tags
- 💊 Medicinal uses
- 🌾 Agricultural uses
- 📜 Vrikshayurveda (ancient plant science) references
- 🌍 Explore all 10 plants in the knowledge base

---

## 🧠 Model Architecture

```
Input (224×224×3)
    │
    ▼
MobileNetV2 (Pre-trained ImageNet weights)
    │  Phase 1: Fully Frozen
    │  Phase 2: Top 30 layers unfrozen
    ▼
GlobalAveragePooling2D
    ▼
Dense(256, ReLU) + Dropout(0.5) + BatchNorm
    ▼
Dense(N_classes, Softmax)
    ▼
Output: Plant class + confidence
```

**Why MobileNetV2?**
- Lightweight (~14 MB) — fast inference on CPU
- Excellent accuracy on image classification tasks
- Transfer learning from ImageNet (1.2M diverse images)
- Suitable for academic demo and mobile deployment

---

## 🌿 Supported Plant Classes (10)

| Class | Common Name | Scientific Name |
|-------|------------|----------------|
| `Aloe_Vera` | Aloe Vera | *Aloe barbadensis miller* |
| `Banana` | Banana | *Musa acuminata* |
| `Basil` | Basil | *Ocimum basilicum* |
| `Mango` | Mango | *Mangifera indica* |
| `Neem` | Neem | *Azadirachta indica* |
| `Rose` | Rose | *Rosa damascena* |
| `Tulsi` | Holy Basil (Tulsi) | *Ocimum tenuiflorum* |
| `Turmeric` | Turmeric | *Curcuma longa* |
| `Almond` | Almond | *Prunus dulcis* |
| `Papaya` | Papaya | *Carica papaya* |

---

## ☁️ Deployment — Streamlit Cloud

1. Push the project to a **public GitHub repository**
2. Visit [share.streamlit.io](https://share.streamlit.io) → **New app**
3. Connect your GitHub repo, select `app.py` as the main file
4. Add a `model/plant_model.h5` (use Git LFS for large files) or host it on Google Drive and download via `gdown` in a startup script
5. Click **Deploy**

> **Tip**: If the model file is too large for GitHub, use `gdown` to download it from Google Drive on startup. Add a `--server.maxUploadSize 50` flag to `streamlit run` for larger image uploads.

---

## 🔧 Troubleshooting

| Problem | Solution |
|---------|----------|
| `Module not found: tensorflow` | Run `pip install -r requirements.txt` |
| `plant_model.h5 not found` | Train the model first: `python train.py` |
| Low prediction accuracy | Use more training data (Option B dataset) |
| `CUDA out of memory` | Reduce `--batch_size` to 8 or 16 |
| Dataset folder not found | Run `python download_dataset.py` |
| Slow training | Training on GPU is 10–50× faster — install `tensorflow-gpu` |

---

## 📊 Expected Results

With the **sample dataset** (demo only):
- Training accuracy: ~60–75% (very few images per class)
- Validation accuracy: ~40–60%

With the **full PlantVillage dataset** (≥500 images/class):
- Training accuracy: ~92–97%
- Validation accuracy: ~88–94%

---

## 📚 References

- [MobileNetV2 Paper (Sandler et al., 2018)](https://arxiv.org/abs/1801.04381)
- [PlantVillage Dataset — Kaggle](https://www.kaggle.com/datasets/emmarex/plantdisease)
- [Vrikshayurveda — Ancient Plant Science](https://en.wikipedia.org/wiki/Vrikshayurveda)
- [TensorFlow Documentation](https://www.tensorflow.org/api_docs)
- [Streamlit Documentation](https://docs.streamlit.io)

---

*Built for Academic Internship Demo · 2025 · PlantML Project*

# PlantML
