import argparse

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps
from tensorflow.keras import layers, models

CLASS_NAMES = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
IMG_SIZE = (224, 224)
WEIGHTS_PATH = 'best_weights.weights.h5'


def build_model_architecture(num_classes=len(CLASS_NAMES)):
    base_model = tf.keras.applications.ResNet50V2(
        weights=None,
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


def load_trained_model(weights_path=WEIGHTS_PATH):
    model = build_model_architecture()
    model.load_weights(weights_path)
    print(f"Loaded weights from '{weights_path}'")
    return model


def load_and_preprocess_image(image_path, target_size=IMG_SIZE, show_steps=True):
    img = Image.open(image_path)
    print(f"Loaded image: {image_path}")
    print(f"   Original size: {img.size} (width x height), mode: {img.mode}")

    img = ImageOps.exif_transpose(img)
    img = img.convert('RGB')

    height, width = target_size
    img_resized = img.resize((width, height), Image.NEAREST)
    print(f"   Resized to: {img_resized.size}")

    img_array = np.asarray(img_resized, dtype=np.float32) / 255.0
    img_batch = np.expand_dims(img_array, axis=0)

    if show_steps:
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))

        axes[0].imshow(img)
        axes[0].set_title(f'Original Image\n{img.size[0]}x{img.size[1]}',
                          fontsize=12, fontweight='bold')
        axes[0].axis('off')

        axes[1].imshow(img_resized)
        axes[1].set_title(f'Resized for Model\n{img_resized.size[0]}x{img_resized.size[1]}',
                          fontsize=12, fontweight='bold')
        axes[1].axis('off')

        plt.tight_layout()
        plt.show()

    return img_batch, img


def predict_waste_class(image_path, model, show_visualization=True):
    img_batch, display_img = load_and_preprocess_image(image_path, show_steps=show_visualization)

    probs = model.predict(img_batch, verbose=0)[0]

    best_idx = int(np.argmax(probs))
    best_class = CLASS_NAMES[best_idx]
    confidence = float(probs[best_idx] * 100)

    print("\n" + "=" * 70)
    print("PREDICTION RESULTS")
    print("=" * 70)
    print(f"Predicted Class: {best_class.upper()}")
    print(f"Confidence: {confidence:.2f}%\n")
    for i in np.argsort(probs)[::-1]:
        print(f"   {CLASS_NAMES[i]:<10} {probs[i] * 100:6.2f}%")

    if show_visualization:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        axes[0].imshow(display_img)
        axes[0].set_title('Input Image', fontsize=14, fontweight='bold')
        axes[0].axis('off')

        colors = ['green' if i == best_idx else 'steelblue' for i in range(len(CLASS_NAMES))]
        axes[1].barh(CLASS_NAMES, probs * 100, color=colors)
        axes[1].set_xlabel('Confidence (%)', fontsize=12)
        axes[1].set_title('Prediction Probabilities', fontsize=14, fontweight='bold')
        axes[1].set_xlim(0, 100)

        plt.tight_layout()
        plt.show()

    return {
        'predicted_class': best_class,
        'confidence': confidence,
        'all_probabilities': {name: float(p * 100) for name, p in zip(CLASS_NAMES, probs)},
    }


def main():
    parser = argparse.ArgumentParser(description='Predict the waste class of an image.')
    parser.add_argument('image', help='path to the image file')
    parser.add_argument('--weights', default=WEIGHTS_PATH,
                        help=f'path to the trained weights (default: {WEIGHTS_PATH})')
    parser.add_argument('--no-plot', action='store_true', help='skip the matplotlib figures')
    args = parser.parse_args()

    model = load_trained_model(args.weights)
    predict_waste_class(args.image, model, show_visualization=not args.no_plot)


if __name__ == '__main__':
    main()