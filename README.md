# Waste Classification with Transfer Learning

A 6-class waste image classifier (cardboard, glass, metal, paper, plastic, trash) built on a pretrained ResNet50V2, using data augmentation and class weighting. It reaches **88.9% accuracy on the test set**.

## Results

Test accuracy: **88.9%**

## How it works

- **Backbone:** ResNet50V2 pretrained on ImageNet and kept frozen, so only the new classification head is trained.
- **Head:** global average pooling → batch norm → dense layers of 1024, 512 and 256 units (ReLU, dropout 0.5 / 0.4 / 0.3) → softmax over the 6 classes.
- **Data augmentation:** random rotation, shifts, shear, zoom, horizontal and vertical flips, and brightness changes.
- **Class weighting:** rarer classes (especially *trash*) count more in the loss, to handle class imbalance.
- **Training:** Adam (learning rate 0.001) with early stopping and learning-rate reduction on validation accuracy. The best checkpoint is kept and evaluated on the test set.

## Dataset

[Garbage Classification](https://www.kaggle.com/asdasdasasdas/garbage-classification) on Kaggle: 2,527 images across the 6 classes. The training script uses the dataset's own train / validation / test split files.

## Project structure

```
├── waste-classification.py   training and evaluation
├── forward.py                predict the class of a new image
└── requirements.txt
```

## Setup

```bash
git clone https://github.com/YOUR-USERNAME/waste-classification.git
cd waste-classification
pip install -r requirements.txt
```

## Model weights

Training saves the model as `best_weights.weights.h5` (about 128 MB, too large to store in the repo). `forward.py` expects this file in the project folder, or you can pass its location with `--weights`.

## Usage

Predict the class of a photo:

```bash
python forward.py path/to/photo.jpg
```

Options:

- `--weights PATH` to use a weights file stored somewhere else
- `--no-plot` to print the result without showing the figures

## Training

The training script was written to run on Kaggle:

1. Create a Kaggle notebook, add the Garbage Classification dataset, and enable a GPU.
2. Run `waste-classification.py`.

It saves the best weights as `best_weights.weights.h5` and prints the test accuracy, a classification report and a confusion matrix. To train on your own machine instead, download the dataset and change `DATA_ROOT` at the top of the script.