import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import json

# -------------------------
# SETTINGS
# -------------------------
IMG_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 40

DATASET_PATH = "C:/Users/dines/OneDrive/Desktop/plant/plant diseases"

# -------------------------
# DATA GENERATOR (AUTO SPLIT)
# -------------------------
datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    validation_split=0.2,
    rotation_range=15,
    zoom_range=0.2,
    horizontal_flip=True
)

train_data = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training',
    shuffle=True
)

val_data = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)

NUM_CLASSES = train_data.num_classes

print("Class Indices:", train_data.class_indices)

# -------------------------
# SAVE CLASS NAMES (IMPORTANT)
# -------------------------
with open("C:/Users/dines/OneDrive/Desktop/plant/class_names.json", "w") as f:
    json.dump(train_data.class_indices, f)

# -------------------------
# CLASS WEIGHTS (BALANCE)
# -------------------------
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_data.classes),
    y=train_data.classes
)

class_weights = dict(enumerate(class_weights))

# -------------------------
# MODEL
# -------------------------
base_model = EfficientNetB0(
    weights='imagenet',
    include_top=False,
    input_shape=(224, 224, 3)
)

# Freeze base model
for layer in base_model.layers:
    layer.trainable = False

# Custom head
x = base_model.output
x = layers.GlobalAveragePooling2D()(x)
x = layers.BatchNormalization()(x)
x = layers.Dense(256, activation='relu')(x)
x = layers.Dropout(0.5)(x)
output = layers.Dense(NUM_CLASSES, activation='softmax')(x)

model = models.Model(inputs=base_model.input, outputs=output)

# -------------------------
# COMPILE (PHASE 1)
# -------------------------
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# -------------------------
# CALLBACKS
# -------------------------
early_stop = EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True)

checkpoint = ModelCheckpoint("best_model.keras", monitor='val_accuracy', save_best_only=True)

lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=3, min_lr=1e-6)

# -------------------------
# TRAIN PHASE 1
# -------------------------
model.fit(
    train_data,
    validation_data=val_data,
    epochs=20,
    callbacks=[early_stop, checkpoint, lr_scheduler],
    class_weight=class_weights
)

# -------------------------
# FINE-TUNING (IMPORTANT 🔥)
# -------------------------
for layer in base_model.layers[-100:]:
    layer.trainable = True

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# -------------------------
# TRAIN PHASE 2
# -------------------------
model.fit(
    train_data,
    validation_data=val_data,
    epochs=20,
    callbacks=[early_stop, checkpoint, lr_scheduler],
    class_weight=class_weights
)

# -------------------------
# SAVE FINAL MODEL
# -------------------------
model.save("final_plant_model.keras")

print("✅ FINAL MODEL READY (NO CLASS MISMATCH)")