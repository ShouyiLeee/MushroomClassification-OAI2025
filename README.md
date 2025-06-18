# MushroomClassification-OAI2025

This project leverages a custom Vision Transformer (ViT) architecture to classify mushrooms from images. The model is designed to handle multiple images per sample, improving classification accuracy by aggregating visual information.

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset Structure](#dataset-structure)
- [Model Architecture](#model-architecture)
- [Installation](#installation)
- [Training](#training)
- [Evaluation](#evaluation)
- [Inference](#inference)
- [Output Conversion](#output-conversion)
- [Acknowledgements](#acknowledgements)

---

## Project Overview

- **Goal:** Classify mushrooms into four categories using a custom ViT model.
- **Categories:**
  - Bào ngư xám + trắng
  - Đùi gà Baby (cắt ngắn)
  - Nấm mỡ
  - Linh chi trắng
- **Approach:** Each sample consists of multiple images, which are processed and pooled by the model for robust classification.

---

## Dataset Structure

The dataset should be organized as follows:

```
Source_submit/data/
  train/
    bào ngư xám + trắng/
      BN347.jpg
      ...
    Đùi gà Baby (cắt ngắn)/
      DG347.jpg
      ...
    linh chi trắng/
      LC347.jpg
      ...
    nấm mỡ/
      NM347.jpg
      ...
  test/
    image_01.jpg
    ...
```

- Each class folder contains images named with a prefix (e.g., `BN`, `DG`, `LC`, `NM`).

---

## Model Architecture

- **Backbone:** Vision Transformer (ViT) from the `timm` library.
- **Custom Head:** The default ViT classification head is replaced with a custom MLP for the four mushroom classes.
- **Multi-Image Input:** The model accepts a batch of grouped images per sample, performs mean pooling on their embeddings, and classifies the pooled representation.

See `Source_submit/libs/multi_vit_classifier.py` for implementation details.

---

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/MushroomClassification-OAI2025.git
   cd MushroomClassification-OAI2025/Source_submit
   ```

2. **Install dependencies:**
   ```bash
   pip install -r libs/requirements.txt
   ```

   **Key dependencies:**
   - torch
   - torchvision
   - timm
   - scikit-learn
   - pandas, numpy, Pillow, matplotlib, tqdm, accelerate, datasets, transformers

---

## Inference

To classify new images:

1. Prepare your test images in `Source_submit/data/test/`.
2. Use the trained model to predict classes for each sample.
3. The model expects grouped images per sample (see `MultiImageMushroomDataset` for grouping logic).

---

## Output Conversion

After inference, convert the output to the required submission format:

```bash
python libs/convert_output.py
```

This script maps class names to submission labels and outputs a CSV file in the correct format.

---

## Acknowledgements

- Built with [PyTorch](https://pytorch.org/), [timm](https://github.com/huggingface/pytorch-image-models), and [scikit-learn](https://scikit-learn.org/).
- Custom dataset and model logic inspired by the needs of multi-image mushroom classification.

---

**For more details, see the code in the `Source_submit/libs/` directory.**
