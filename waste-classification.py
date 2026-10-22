import random
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                             classification_report, confusion_matrix)
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import (EarlyStopping, ModelCheckpoint,
                                        ReduceLROnPlateau)
from tensorflow.keras.preprocessing.image import ImageDataGenerator

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)
random.seed(SEED)

DATA_ROOT = '/kaggle/input/garbage-classification'
IMAGES_DIR = f'{DATA_ROOT}/Garbage classification/Garbage classification/'
TRAIN_FILE = f'{DATA_ROOT}/one-indexed-files-notrash_train.txt'
VAL_FILE = f'{DATA_ROOT}/one-indexed-files-notrash_val.txt'
TEST_FILE = f'{DATA_ROOT}/one-indexed-files-notrash_test.txt'

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 60
WEIGHTS_PATH = 'best_weights.weights.h5'

CLASS_NAMES = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
NUM_CLASSES = len(CLASS_NAMES)

LABEL_MAP = {1: 'glass', 2: 'paper', 3: 'cardboard', 4: 'plastic', 5: 'metal', 6: 'trash'}


def add_class_folder(filename):
    class_name = filename[:re.search(r'\d', filename).start()]
    return f'{class_name}/{filename}'


def load_split(split_file):
    df = pd.read_csv(split_file, sep=' ', header=None, names=['path', 'label'])
    df['path'] = df['path'].apply(add_class_folder)
    df['label'] = df['label'].apply(lambda label_id: LABEL_MAP[label_id])
    return df


df_train = load_split(TRAIN_FILE)
df_valid = load_split(VAL_FILE)
df_test = load_split(TEST_FILE)

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.3,
    height_shift_range=0.3,
    shear_range=0.3,
    zoom_range=0.3,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='nearest',
    brightness_range=[0.6, 1.4],
)

valid_test_datagen = ImageDataGenerator(rescale=1./255)


def make_generator(datagen, df, shuffle):
    return datagen.flow_from_dataframe(
        dataframe=df,
        directory=IMAGES_DIR,
        x_col='path',
        y_col='label',
        classes=CLASS_NAMES,
        target_size=IMG_SIZE,
        color_mode='rgb',
        class_mode='categorical',
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        seed=SEED,
    )


gen_train = make_generator(train_datagen, df_train, shuffle=True)
gen_valid = make_generator(valid_test_datagen, df_valid, shuffle=False)
test_gen = make_generator(valid_test_datagen, df_test, shuffle=False)


def build_optimized_model(num_classes):
    base_model = tf.keras.applications.ResNet50V2(
        weights='imagenet',
        include_top=False,
        input_shape=(*IMG_SIZE, 3)
    )

    base_model.trainable = False

    inputs = tf.keras.Input(shape=(*IMG_SIZE, 3))
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D(name='gap')(x)
    x = layers.BatchNormalization(name='bn1')(x)

    x = layers.Dense(1024, activation='relu', name='dense1')(x)
    x = layers.Dropout(0.5, name='dropout1')(x)

    x = layers.Dense(512, activation='relu', name='dense2')(x)
    x = layers.Dropout(0.4, name='dropout2')(x)

    x = layers.Dense(256, activation='relu', name='dense3')(x)
    x = layers.Dropout(0.3, name='dropout3')(x)

    outputs = layers.Dense(num_classes, activation='softmax', name='output')(x)

    return models.Model(inputs, outputs, name='waste_classifier')


model = build_optimized_model(num_classes=NUM_CLASSES)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['categorical_accuracy']
)

print('\nModel Summary:')
model.summary()

class_counts = df_train['label'].value_counts()
class_weights = {
    idx: float(len(df_train) / (NUM_CLASSES * class_counts[name]))
    for name, idx in gen_train.class_indices.items()
}
print(f'\nClass weights: {class_weights}')

early_stopping = EarlyStopping(
    monitor='val_categorical_accuracy',
    patience=20,
    restore_best_weights=True,
    mode='max',
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_categorical_accuracy',
    factor=0.5,
    patience=7,
    min_lr=1e-7,
    mode='max',
    verbose=1
)

checkpoint = ModelCheckpoint(
    WEIGHTS_PATH,
    monitor='val_categorical_accuracy',
    save_best_only=True,
    save_weights_only=True,
    mode='max',
    verbose=1
)

callbacks = [early_stopping, reduce_lr, checkpoint]

print("\n" + "=" * 70)
print("TRAINING: Single-Phase Training")
print("=" * 70)

history = model.fit(
    gen_train,
    validation_data=gen_valid,
    epochs=EPOCHS,
    callbacks=callbacks,
    class_weight=class_weights,
    verbose=1
)

model.load_weights(WEIGHTS_PATH)

fig, axes = plt.subplots(1, 2, figsize=(16, 5))

epochs_range = range(1, len(history.history['loss']) + 1)

axes[0].plot(epochs_range, history.history['loss'], 'b-', label='Training Loss', linewidth=2)
axes[0].plot(epochs_range, history.history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
axes[0].set_xlabel('Epoch', fontsize=12)
axes[0].set_ylabel('Loss', fontsize=12)
axes[0].set_title('Model Loss Over Time', fontsize=14, fontweight='bold')
axes[0].legend(fontsize=11)
axes[0].grid(True, alpha=0.3)

axes[1].plot(epochs_range, history.history['categorical_accuracy'], 'b-',
             label='Training Accuracy', linewidth=2)
axes[1].plot(epochs_range, history.history['val_categorical_accuracy'], 'r-',
             label='Validation Accuracy', linewidth=2)
axes[1].axhline(y=0.90, color='g', linestyle='--', label='90% Target', linewidth=2)
axes[1].set_xlabel('Epoch', fontsize=12)
axes[1].set_ylabel('Accuracy', fontsize=12)
axes[1].set_title('Model Accuracy Over Time', fontsize=14, fontweight='bold')
axes[1].legend(fontsize=11)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

best_epoch = int(np.argmax(history.history['val_categorical_accuracy']))
best_val_acc = history.history['val_categorical_accuracy'][best_epoch]
best_train_acc = history.history['categorical_accuracy'][best_epoch]

print(f"\nBest Results at Epoch {best_epoch + 1}:")
print(f"   Training Accuracy:   {best_train_acc*100:.2f}%")
print(f"   Validation Accuracy: {best_val_acc*100:.2f}%")

test_loss, test_acc = model.evaluate(test_gen, verbose=1)
print(f'\nTEST ACCURACY: {test_acc * 100:.2f}%')
print(f'Test Loss: {test_loss:.4f}')

test_gen.reset()
y_pred_probs = model.predict(test_gen, verbose=1)
y_pred_classes = np.argmax(y_pred_probs, axis=1)
y_true = np.array(test_gen.classes)

print("\n" + "=" * 70)
print("CLASSIFICATION REPORT")
print("=" * 70)
print(classification_report(y_true, y_pred_classes, labels=list(range(NUM_CLASSES)),
                            target_names=CLASS_NAMES, digits=4))

cm = confusion_matrix(y_true, y_pred_classes, labels=list(range(NUM_CLASSES)))
fig, ax = plt.subplots(figsize=(12, 10))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASS_NAMES)
disp.plot(cmap=plt.cm.Blues, ax=ax, values_format='d')
plt.title('Confusion Matrix - Test Set', fontsize=16, fontweight='bold', pad=20)
plt.tight_layout()
plt.show()

print("\n" + "=" * 70)
print("FINAL SUMMARY")
print("=" * 70)
overall_acc = accuracy_score(y_true, y_pred_classes)
print(f"Overall Accuracy: {overall_acc*100:.2f}%")
print(f"Correctly classified: {int(np.sum(y_pred_classes == y_true))} / {len(y_true)}")

print("\n" + "=" * 70)
print(f"Best weights saved as '{WEIGHTS_PATH}'")
print("=" * 70)