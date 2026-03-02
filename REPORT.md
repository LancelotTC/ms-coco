**Challenge**: MS-Coco Multi-label Classification

**Student**: Lancelot Tariot Camille, Sang Nguyen

## Table of Content
- [I. Introduction](#i-introduction)
- [II. Model Benchmarking](#ii-model-benchmarking-and-performance)
- [III. Model Architecture](#iii-model-architecture)
- [IV. Model Training](#iv-model-training)
- [V. Model Inference](#v-model-inference)
- [VI. Evaluation Metrics](#vi-evaluation-metrics)
- [VII. Conclusion](#vii-conclusion)


## I. Introduction

In the context of multi-label image classification, a `good` model is often defined by its ability to achieve high predictive performance, typically measured by metrics such as the F1 score. However, in practical applications, a high F1 score alone is not always sufficient to determine the overall quality or suitability of a model. Other important factors include `model complexity`, `inference speed`, `training time`, and `resource efficiency`.

For this project, our goal is to identify models that not only deliver sufficiently high accuracy and F1 scores, but also maintain a lightweight architecture and fast training times.

We prioritize models that strike a balance between predictive performance and computational efficiency. In particular, we seek architectures that are compact and efficient, enabling rapid experimentation and deployment, even if this means accepting a slight trade-off in F1 score or accuracy. This approach ensures that the selected model is practical for real-world use, where speed and resource constraints are often as critical as raw performance.

## II. Model Benchmarking

### 1. Model selection for benchmarking

In this project, we addressed the challenge of multi-label image classification on the MS-COCO dataset, where each image can contain multiple object categories simultaneously. This requires the model to predict several classes per image, rather than a single label.

To solve this, we adopted a transfer learning approach using a variety of state-of-the-art convolutional neural network backbones from torchvision:
- ConvNeXt (tiny, small, base, large)
- Swin Transformer (swin_t, swin_v2_t, etc.)
- ResNet (resnet18, resnet50)
- DenseNet
- MobileNet
- EfficientNet
- RegNet

To adapt these models for multi-label classification, we replaced the original classification head of each backbone with a custom head: a linear layer mapping to 80 output classes (corresponding to the MS-COCO label set), followed by a BatchNorm1d layer.

This design enables the model to output independent logits for each class, making it suitable for multi-label prediction.

The model selection and head replacement logic is implemented in `models_factory.py`, which ensures compatibility with different backbones and allows easy switching between architectures.

### 2. Summary of the Result

Below is a summary of the best-performing configurations:

| # | Model Architecture        | Best Val F1 | Best Epoch | Total Epochs | Learning Rate | Unfrozen Batch | Training Time |
|---|---------------------------|------------|------------|--------------|--------------|---------------|------------|
| 0 | ConvNeXt Base             | 0.633600   | 17         | 25           | 1.0e-07      | 8             | ?          |
| 1 | ConvNeXt Small            | 0.632600   | 25         | 25           | 1.0e-07      | 16            | ?          |
| 2 | ConvNeXt Tiny (v1)        | 0.624100   | 15         | 18           | 1.0e-06      | 16            | ?          |
| 3 | ConvNeXt Tiny (v2)        | 0.618100   | 14         | 25           | 1.0e-07      | 16            | ?          |
| 4 | ConvNeXt Tiny (v3)        | 0.609600   | 14         | 18           | 1.0e-06      | 32            | ?          |
| 5 | ConvNeXt Tiny (v4)        | 0.571800   | 17         | 18           | 1.0e-06      | 256           | ?          |

### 3. Model Selection
Based on the summary in the previous table, we can see that ConvNeXt Base and Small achieved the best performance in terms of validation F1 score, while the Tiny versions train faster but have lower F1 scores. 

Therefore, to balance accuracy and training cost, we decided to use four backbones for the main training: ConvNeXt, Mobile Net, Swing Transformer, RegNet.
---

## IV. Model Architecture

Before proceeding with the training using the four selected backbones, we first present an overview of the architectural features of these models. Each backbone has distinct design principles that influence its capacity, computational cost, and suitability for multi-label classification:
// gpt generated
ConvNeXt – A modernized convolutional architecture inspired by Transformer design, featuring inverted bottlenecks, large kernel sizes, and improved normalization schemes for better performance on image classification tasks.

MobileNet – A lightweight convolutional model optimized for efficiency and speed, using depthwise separable convolutions to reduce the number of parameters while maintaining reasonable accuracy.

Swin Transformer – A hierarchical Vision Transformer that processes images using shifted windows, enabling both global and local context modeling with reduced computational complexity compared to standard Transformers.

RegNet – A family of network designs with regularized width and depth configurations, offering a balance between model size, speed, and accuracy for large-scale image classification.

Understanding these architectural differences allows us to interpret their performance during training and provides insights into how they handle multi-label predictions on MS-COCO.

## V. Model Training

### 1. Model Training and Configuration

Training is orchestrated by `training.py` and utilizes the following workflow:

**Model Initialization**

The selected backbone (e.g. ResNet, MobileNet, EfficientNet, ConvNeXt, Swin Transformer, RegNet) is loaded with pretrained ImageNet weights. The final classifier layer is replaced with a custom head: a Linear layer followed by BatchNorm1d, outputting logits for 80 classes.

**Dataset Splitting**

The dataset is split into training and validation subsets using a seeded random split for reproducibility. The default validation split ratio is **5%** `(VAL_SPLIT=0.05)`, with the random seed set to **42** `(SEED=42)`.

**Loss Function and Class Imbalance**

The loss function is BCEWithLogitsLoss, with a computed pos_weight vector to address class imbalance. The positive weights are calculated from the training subset to ensure balanced learning across all classes.

**Backbone Freezing and Unfreezing**

Training starts with the backbone frozen `(FREEZE_BACKBONE=True by default)`, allowing only the classifier head to be trained initially. The backbone is unfrozen after 1 epoch `(UNFREEZE_BACKBONE_EPOCH=1)`, enabling full fine-tuning. There is also an option to partially unfreeze the last N layers `(UNFREEZE_LAST_N_BACKBONE_LAYERS=None for full unfreeze)`.

**Batch Sizes and Gradient Accumulation**

- When the backbone is frozen: `TRAIN_BATCH_SIZE_FROZEN=256`, `GRAD_ACCUM_STEPS_FROZEN=1`
- When the backbone is unfrozen: `TRAIN_BATCH_SIZE_UNFROZEN=16`, `GRAD_ACCUM_STEPS_UNFROZEN=4`
- Validation batch size: `VAL_BATCH_SIZE=256`

**Learning Rate and Scheduling**

Differential learning rates are used:
- Backbone: `BACKBONE_BASE_LR=1e-5`
- Head: `HEAD_BASE_LR=1e-4`
- The learning rate scheduler uses milestones at epoch 9 (`MILESTONES=(9,)`) with a decay factor of `1e-2`.

**Checkpointing and Artifacts**

The best model checkpoint is tracked by validation F1 score. For each run, the following artifacts are saved in a timestamped folder under trained_models:

- `best_model.pt` (model weights)
- `run_config.json` (full configuration and results)
- `confusion_matrix.png` (generated if missing)

The active checkpoint for inference is also updated at best_model.pt.

**Threshold Tuning**

The optimal multi-label threshold is tuned on the validation set by evaluating weighted F1 scores across candidate thresholds from `0.05` to `0.95` in steps of `0.05`.

**TensorBoard Logging**

Optionally, training and validation metrics can be logged to TensorBoard by setting `USE_TENSORBOARD=True` in training.py.

**Other Settings**
- Number of epochs: `NUM_EPOCHS=25`
- Number of workers for data loading: `NUM_WORKERS=4` (increase the value of this parameter to reduce the training time)
- Early stopping is implemented but disabled by default.

---

## VI. Model Inference

### 1. Model Inference and Configuration

The inference process is managed by the `testing.py` script and is designed for efficient, reproducible multi-label prediction on the MS-COCO test set. The key steps and configurations are as follows:

**Model Loading**

The script loads the best model checkpoint from the path specified by MODEL_PATH (default: `best_model.pt`). The model architecture is reconstructed using the configuration saved in the checkpoint, ensuring full consistency between training and inference.

**Batch Processing**

Test images are processed in batches with a default batch size of 32 (BATCH_SIZE=32). Data loading is performed with 0 worker processes (NUM_WORKERS=0) for maximum compatibility.

**Threshold Selection**

The inference threshold for multi-label prediction is determined in the following order of precedence:
- best_threshold saved in the checkpoint (if available)
- th_multi_label argument (if provided)
- Default value in `testing.py`: 0.5 (`TH_MULTI_LABEL=0.5`)

### 2. Output format

Predictions are saved in a JSON file, mapping each image ID (filename without extension) to a list of predicted class indices.

The output filename is automatically expanded with metadata tokens (model name, F1 score, epoch, training and test settings) for traceability. The base output path is predictions.json (OUTPUT_PATH=Path("predictions.json")).

### 3. Running Inference

You can run inference from this jupyter notebook or from terminal.

```bash
python testing.py
```

---

## VII. Evaluation Metrics and Analysis



## VIII. Conclusion

---
