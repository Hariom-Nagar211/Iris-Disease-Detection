"""
train.py
========
Train the EfficientNetB0 retinal disease classifier.

Usage
-----
    python train.py --data_dir /path/to/dataset --epochs 30

Expected dataset layout
-----------------------
    data_dir/
    ├── Central Serous Chorioretinopathy/
    ├── Diabetic Retinopathy/
    ├── Disc Edema/
    ├── Glaucoma/
    ├── Healthy/
    ├── Macular Scar/
    ├── Myopia/
    ├── Pterygium/
    ├── Retinal Detachment/
    └── Retinitis Pigmentosa/

Each sub-folder contains images for that class (jpg / png).
The script automatically splits into train / val (80/20).
"""

import argparse
import os
import tensorflow as tf
from tensorflow.keras import callbacks
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from utils.model_utils import build_model, IMG_SIZE, WEIGHTS_PATH

# ── CLI args ───────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Train retinal disease classifier")
parser.add_argument("--data_dir", type=str, required=True, help="Path to dataset root")
parser.add_argument("--epochs",   type=int, default=30)
parser.add_argument("--batch",    type=int, default=32)
parser.add_argument("--finetune_epochs", type=int, default=10,
                    help="Extra epochs with unfrozen backbone (fine-tuning phase)")
args = parser.parse_args()

# ── Data generators ────────────────────────────────────────────────────────────
train_datagen = ImageDataGenerator(
    validation_split=0.2,
    # Augmentations
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.15,
    horizontal_flip=True,
    brightness_range=[0.8, 1.2],
    fill_mode="nearest",
)

train_gen = train_datagen.flow_from_directory(
    args.data_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=args.batch,
    class_mode="categorical",
    subset="training",
    shuffle=True,
)

val_gen = train_datagen.flow_from_directory(
    args.data_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=args.batch,
    class_mode="categorical",
    subset="validation",
    shuffle=False,
)

print(f"[INFO] Classes: {train_gen.class_indices}")
print(f"[INFO] Train samples: {train_gen.n}  |  Val samples: {val_gen.n}")

# ── Build model ────────────────────────────────────────────────────────────────
model = build_model(num_classes=len(train_gen.class_indices))
model.summary()

# ── Callbacks ──────────────────────────────────────────────────────────────────
os.makedirs(os.path.dirname(WEIGHTS_PATH), exist_ok=True)

cb_list = [
    callbacks.ModelCheckpoint(
        WEIGHTS_PATH,
        monitor="val_accuracy",
        save_best_only=True,
        save_weights_only=False,
        verbose=1,
    ),
    callbacks.EarlyStopping(monitor="val_loss", patience=7, restore_best_weights=True),
    callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, verbose=1),
    callbacks.TensorBoard(log_dir="logs/phase1"),
]

# ── Phase 1: Train head only ───────────────────────────────────────────────────
print("\n[PHASE 1] Training classification head (backbone frozen) …")
history1 = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=args.epochs,
    callbacks=cb_list,
)

# ── Phase 2: Fine-tune top layers of backbone ──────────────────────────────────
print("\n[PHASE 2] Fine-tuning top 50 layers of EfficientNetB0 backbone …")
backbone = model.get_layer("efficientnetb0")
backbone.trainable = True

# Freeze all but the top 50 layers
for layer in backbone.layers[:-50]:
    layer.trainable = False

# Recompile with lower LR for fine-tuning
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-5),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

cb_list2 = [
    callbacks.ModelCheckpoint(
        WEIGHTS_PATH,
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1,
    ),
    callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
    callbacks.TensorBoard(log_dir="logs/phase2"),
]

history2 = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=args.finetune_epochs,
    callbacks=cb_list2,
)

print(f"\n[DONE] Best model saved to: {WEIGHTS_PATH}")
