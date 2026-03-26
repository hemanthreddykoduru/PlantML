"""
train.py
--------
Training script for the Plant Identification CNN using MobileNetV2 (Transfer Learning).

Pipeline:
  1. Download/verify dataset from Kaggle or local folder
  2. Build data generators with augmentation
  3. Build the MobileNetV2-based model
  4. Train in two phases:
       Phase 1 – Train only the custom head (base frozen)
       Phase 2 – Fine-tune top layers of the base
  5. Plot accuracy/loss curves
  6. Save the final model as model/plant_model.h5
  7. Save class names to model/class_names.json

Usage:
  python train.py --data_dir dataset/ --epochs 20 --batch_size 32

Alternatively, if you have not downloaded the dataset yet, the script
will print clear download instructions.
"""

import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt

# ── TensorFlow / Keras ────────────────────────────────────────────────────────
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
)

# ──────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────────────────────────────────────

IMAGE_SIZE   = (224, 224)
INPUT_SHAPE  = (224, 224, 3)
BATCH_SIZE   = 32
EPOCHS_HEAD  = 10   # Phase 1: train head only
EPOCHS_FINE  = 15   # Phase 2: fine-tuning top layers
MODEL_DIR    = "model"
MODEL_PATH   = os.path.join(MODEL_DIR, "plant_model.h5")
CLASSES_PATH = os.path.join(MODEL_DIR, "class_names.json")


# ──────────────────────────────────────────────────────────────────────────────
# DATASET DOWNLOAD INSTRUCTIONS
# ──────────────────────────────────────────────────────────────────────────────

DATASET_INSTRUCTIONS = """
╔══════════════════════════════════════════════════════════════════════════════╗
║               DATASET SETUP INSTRUCTIONS                                   ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                            ║
║  Option A — Use the provided sample script (recommended):                  ║
║  ─────────────────────────────────────────────────────────                 ║
║    python download_dataset.py                                              ║
║  This will create a 'dataset/' folder with sample images for each class.   ║
║                                                                            ║
║  Option B — Use Kaggle (full PlantVillage dataset ~2 GB):                  ║
║  ─────────────────────────────────────────────────────────                 ║
║    1. Install Kaggle CLI:  pip install kaggle                               ║
║    2. Place kaggle.json in ~/.kaggle/                                      ║
║    3. Run:                                                                 ║
║       kaggle datasets download -d emmarex/plantdisease                     ║
║       unzip plantdisease.zip -d dataset/                                   ║
║                                                                            ║
║  The dataset/ folder must follow this structure:                           ║
║    dataset/                                                                ║
║      train/                                                                ║
║        Aloe_Vera/   (≥ 50 images)                                          ║
║        Banana/      (≥ 50 images)                                          ║
║        Basil/       ...                                                    ║
║        Mango/       ...                                                    ║
║        Neem/        ...                                                    ║
║        Rose/        ...                                                    ║
║        Tulsi/       ...                                                    ║
║        Turmeric/    ...                                                    ║
║        Almond/      ...                                                    ║
║        Papaya/      ...                                                    ║
║      val/           (same sub-folder structure)                            ║
║                                                                            ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""


def check_dataset(data_dir: str) -> bool:
    """Return True if the dataset directory exists and has at least two subfolders."""
    train_dir = os.path.join(data_dir, "train")
    if not os.path.isdir(train_dir):
        return False
    subfolders = [
        d for d in os.listdir(train_dir)
        if os.path.isdir(os.path.join(train_dir, d))
    ]
    return len(subfolders) >= 2


# ──────────────────────────────────────────────────────────────────────────────
# DATA GENERATORS
# ──────────────────────────────────────────────────────────────────────────────

def build_generators(data_dir: str, batch_size: int):
    """
    Build Keras ImageDataGenerators for train and validation sets.

    Data augmentation is applied ONLY to the training set to improve
    generalisation and reduce overfitting.

    Returns:
        train_gen, val_gen, num_classes
    """
    # ── Training augmentation ─────────────────────────────────────────────────
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,           # Normalise pixel values to [0, 1]
        rotation_range=30,           # Randomly rotate images up to 30°
        width_shift_range=0.2,       # Randomly shift left/right by 20%
        height_shift_range=0.2,      # Randomly shift up/down by 20%
        shear_range=0.2,             # Shear transformations
        zoom_range=0.2,              # Random zoom in/out
        horizontal_flip=True,        # Random horizontal flips
        brightness_range=[0.8, 1.2], # Random brightness variation
        fill_mode="nearest",         # Fill strategy for new pixels
    )

    # ── Validation — NO augmentation (only normalisation) ─────────────────────
    val_datagen = ImageDataGenerator(rescale=1.0 / 255)

    train_gen = train_datagen.flow_from_directory(
        os.path.join(data_dir, "train"),
        target_size=IMAGE_SIZE,
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=True,
    )

    val_gen = val_datagen.flow_from_directory(
        os.path.join(data_dir, "val"),
        target_size=IMAGE_SIZE,
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=False,
    )

    num_classes = train_gen.num_classes
    print(f"\n✅  Found {num_classes} classes: {list(train_gen.class_indices.keys())}")
    return train_gen, val_gen, num_classes, train_gen.class_indices


# ──────────────────────────────────────────────────────────────────────────────
# MODEL BUILDING
# ──────────────────────────────────────────────────────────────────────────────

def build_model(num_classes: int) -> tf.keras.Model:
    """
    Build a MobileNetV2-based transfer learning model.

    Architecture:
      MobileNetV2 (pre-trained on ImageNet, base frozen)
       └─ GlobalAveragePooling2D
       └─ Dense(256, relu) + Dropout(0.5)
       └─ Dense(num_classes, softmax)

    Args:
        num_classes: Number of plant categories in the dataset.

    Returns:
        Compiled Keras model (Phase 1 — head-only training).
    """
    # Load MobileNetV2 WITHOUT the top classification layers
    base_model = MobileNetV2(
        input_shape=INPUT_SHAPE,
        include_top=False,
        weights="imagenet",   # Start with ImageNet pre-trained weights
    )
    base_model.trainable = False   # Freeze all base layers initially

    # ── Custom classification head ────────────────────────────────────────────
    inputs = tf.keras.Input(shape=INPUT_SHAPE)
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.5)(x)                          # Regularisation
    x = layers.BatchNormalization()(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs, outputs, name="PlantClassifier_MobileNetV2")

    # Compile for Phase 1
    # Note: On Apple Silicon, the default Adam causes a Metal remapper graph crash
    # when compiling fine-tuned layers. We must use the legacy Adam instead.
    model.compile(
        optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    print("\n📐  Model summary:")
    model.summary()
    return model, base_model


def unfreeze_top_layers(model: tf.keras.Model, base_model, num_layers: int = 30):
    """
    Unfreeze the top `num_layers` of the base model for fine-tuning (Phase 2).

    Args:
        model:      The full Keras model.
        base_model: The MobileNetV2 base sub-model.
        num_layers: How many layers from the top to unfreeze.
    """
    base_model.trainable = True
    # Freeze all layers except the last `num_layers`
    for layer in base_model.layers[:-num_layers]:
        layer.trainable = False

    # Recompile with a lower learning rate for fine-tuning using legacy Adam
    model.compile(
        optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=1e-5),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    trainable_count = sum(1 for l in base_model.layers if l.trainable)
    print(f"\n🔓  Unfrozen {trainable_count} base layers for fine-tuning.")


# ──────────────────────────────────────────────────────────────────────────────
# TRAINING
# ──────────────────────────────────────────────────────────────────────────────

def get_callbacks(phase: int) -> list:
    """
    Return training callbacks appropriate for the given phase.

    Callbacks:
      - EarlyStopping:      Stop when val_loss stops improving
      - ModelCheckpoint:    Save the best model during training
      - ReduceLROnPlateau:  Reduce LR when plateau is detected
    """
    checkpoint_path = os.path.join(MODEL_DIR, f"best_phase{phase}.h5")
    return [
        EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
            verbose=1,
        ),
        ModelCheckpoint(
            filepath=checkpoint_path,
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1,
        ),
    ]


def train(data_dir: str, batch_size: int, epochs_head: int, epochs_fine: int):
    """
    Full training pipeline — Phase 1 (head) then Phase 2 (fine-tune).

    Returns:
        history_head, history_fine
    """
    os.makedirs(MODEL_DIR, exist_ok=True)

    # ── Build data generators ─────────────────────────────────────────────────
    train_gen, val_gen, num_classes, class_indices = build_generators(data_dir, batch_size)

    # Save class indices → used at inference time
    idx_to_class = {v: k for k, v in class_indices.items()}
    class_names_ordered = [idx_to_class[i] for i in range(num_classes)]
    with open(CLASSES_PATH, "w") as f:
        json.dump(class_names_ordered, f, indent=2)
    print(f"💾  Class names saved to {CLASSES_PATH}")

    # ── Build model ───────────────────────────────────────────────────────────
    model, base_model = build_model(num_classes)

    # ── Phase 1: Train head only ──────────────────────────────────────────────
    print("\n" + "="*60)
    print("  PHASE 1: Training classification head (base frozen)")
    print("="*60)
    history_head = model.fit(
        train_gen,
        epochs=epochs_head,
        validation_data=val_gen,
        callbacks=get_callbacks(phase=1),
        verbose=1,
    )

    # ── Phase 2: Fine-tune top layers ─────────────────────────────────────────
    print("\n" + "="*60)
    print("  PHASE 2: Fine-tuning top layers of MobileNetV2")
    print("="*60)
    unfreeze_top_layers(model, base_model, num_layers=30)
    history_fine = model.fit(
        train_gen,
        epochs=epochs_fine,
        validation_data=val_gen,
        callbacks=get_callbacks(phase=2),
        verbose=1,
    )

    # ── Save final model ──────────────────────────────────────────────────────
    model.save(MODEL_PATH)
    print(f"\n✅  Final model saved to: {MODEL_PATH}")

    return history_head, history_fine


# ──────────────────────────────────────────────────────────────────────────────
# PLOTTING
# ──────────────────────────────────────────────────────────────────────────────

def plot_history(history_head, history_fine, save_path: str = "model/training_curves.png"):
    """
    Plot training and validation accuracy/loss curves across both phases.

    Args:
        history_head: Keras History from Phase 1.
        history_fine: Keras History from Phase 2.
        save_path:    Path to save the plot image.
    """
    # Concatenate metrics from both phases
    acc  = history_head.history["accuracy"]      + history_fine.history["accuracy"]
    val_acc  = history_head.history["val_accuracy"]  + history_fine.history["val_accuracy"]
    loss = history_head.history["loss"]          + history_fine.history["loss"]
    val_loss = history_head.history["val_loss"]  + history_fine.history["val_loss"]

    phase1_end = len(history_head.history["accuracy"])
    epochs_range = range(1, len(acc) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle("Plant Classification — Training Curves", fontsize=16, fontweight="bold")

    # ── Accuracy plot ──────────────────────────────────────────────────────────
    axes[0].plot(epochs_range, acc,     label="Train Accuracy",      color="#2196F3")
    axes[0].plot(epochs_range, val_acc, label="Validation Accuracy", color="#4CAF50")
    axes[0].axvline(x=phase1_end, color="gray", linestyle="--", alpha=0.7,
                    label="Phase 1 → Phase 2")
    axes[0].set_title("Accuracy", fontsize=14)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # ── Loss plot ──────────────────────────────────────────────────────────────
    axes[1].plot(epochs_range, loss,     label="Train Loss",      color="#F44336")
    axes[1].plot(epochs_range, val_loss, label="Validation Loss", color="#FF9800")
    axes[1].axvline(x=phase1_end, color="gray", linestyle="--", alpha=0.7,
                    label="Phase 1 → Phase 2")
    axes[1].set_title("Loss", fontsize=14)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"📊  Training curves saved to: {save_path}")


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train Plant Classification Model (MobileNetV2 Transfer Learning)"
    )
    parser.add_argument(
        "--data_dir", type=str, default="dataset",
        help="Root directory containing train/ and val/ subfolders"
    )
    parser.add_argument(
        "--epochs_head", type=int, default=EPOCHS_HEAD,
        help="Epochs for Phase 1 (frozen base)"
    )
    parser.add_argument(
        "--epochs_fine", type=int, default=EPOCHS_FINE,
        help="Epochs for Phase 2 (fine-tuning)"
    )
    parser.add_argument(
        "--batch_size", type=int, default=BATCH_SIZE,
        help="Mini-batch size"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Check dataset availability
    if not check_dataset(args.data_dir):
        print(DATASET_INSTRUCTIONS)
        print("❌  Dataset not found. Please follow the instructions above, then re-run.")
        raise SystemExit(1)

    print(f"\n🌿  Starting Plant Classification Training")
    print(f"    Data dir   : {args.data_dir}")
    print(f"    Batch size : {args.batch_size}")
    print(f"    Phase 1    : {args.epochs_head} epochs (head only)")
    print(f"    Phase 2    : {args.epochs_fine} epochs (fine-tune)")
    print(f"    GPU        : {'✅ Available' if tf.config.list_physical_devices('GPU') else '⚠️  Not detected (using CPU)'}")

    history_head, history_fine = train(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        epochs_head=args.epochs_head,
        epochs_fine=args.epochs_fine,
    )

    plot_history(history_head, history_fine)
    print("\n🎉  Training complete!")
