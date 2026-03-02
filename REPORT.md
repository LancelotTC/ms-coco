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

| #   | Model Architecture | Best Val F1 | Best Epoch | Total Epochs | Learning Rate | Unfrozen Batch | Training Time |
| --- | ------------------ | ----------- | ---------- | ------------ | ------------- | -------------- | ------------- |
| 0   | ConvNeXt Base      | 0.633600    | 17         | 25           | 1.0e-07       | 8              | ?             |
| 1   | ConvNeXt Small     | 0.632600    | 25         | 25           | 1.0e-07       | 16             | ?             |
| 2   | ConvNeXt Tiny (v1) | 0.624100    | 15         | 18           | 1.0e-06       | 16             | ?             |
| 3   | ConvNeXt Tiny (v2) | 0.618100    | 14         | 25           | 1.0e-07       | 16             | ?             |
| 4   | ConvNeXt Tiny (v3) | 0.609600    | 14         | 18           | 1.0e-06       | 32             | ?             |
| 5   | ConvNeXt Tiny (v4) | 0.571800    | 17         | 18           | 1.0e-06       | 256            | ?             |

### 3. Model Selection

Based on the summary in the previous table, we can see that ConvNeXt Base and Small achieved the best performance in terms of validation F1 score, while the Tiny versions train faster but have lower F1 scores.

## Therefore, to balance accuracy and training cost, we decided to use four backbones for the main training: ConvNeXt, Mobile Net, Swing Transformer, RegNet.

## IV. Model Architecture

Before proceeding with the training using the four selected backbones, we first present an overview of the architectural features of these models. Each backbone has distinct design principles that influence its capacity, computational cost, and suitability for multi-label classification:

#### MobileNet: Optimizing for Computational Efficiency

While traditional models like VGG and ResNet are highly accurate, they require massive computational power. MobileNet is explicitly engineered for environments with limited resources, such as mobile phones or embedded devices.

To achieve this, MobileNet replaces standard, heavy convolutions with depthwise separable convolutions. A standard convolution filters and combines inputs in a single, expensive step. MobileNet splits this into two lighter steps:

- Depthwise Convolution: Applies a single filter to each color channel independently.

- Pointwise Convolution (1x1): Linearly combines the outputs of the first step.

By decoupling the filtering and combining phases, MobileNet drastically reduces the number of calculations required, resulting in a highly efficient and fast model.

#### Swin Transformer: Mastering Scale and Resolution

Standard Vision Transformers look at the entire image at once (global self-attention). While powerful, this requires an immense amount of computation for high-resolution images. The Swin Transformer solves this by reintroducing the hierarchical structure of a CNN into the Transformer framework.

It achieves this via a Shifted Window mechanism. Instead of computing attention globally, the Swin Transformer computes it locally within small, non-overlapping windows. To ensure the network still understands the "big picture," the window boundaries are shifted in consecutive layers, allowing information to pass between adjacent windows. Additionally, it gradually merges image patches in deeper layers to build a hierarchical feature map, much like the pooling layers in a ResNet.

#### ConvNeXt: The Modernized Pure ConvNet

Following the massive success of Vision Transformers, ConvNeXt was developed to see if a purely convolutional network could achieve the same performance if designed with modern techniques.

Internally, ConvNeXt simply takes a standard ResNet architecture and incrementally updates it using design principles borrowed from the Swin Transformer. These updates include using larger kernel sizes (e.g., 7x7) to "see" larger parts of the image at once, changing the hidden layer structures, and using modernized normalization techniques. ConvNeXt proves that pure convolutions can still compete with complex self-attention mechanisms while remaining simpler to implement.

#### RegNet: Enhancing Feature Retention via Recurrent Memory

In a standard ResNet, the shortcut (residual) connections help gradients flow, which allows us to train very deep networks. However, because these connections simply add previous outputs to current ones, the network can easily overwrite or "forget" complementary spatial features from earlier layers as it gets deeper.

RegNet addresses this by attaching a Regulator Module to the ResNet backbone. This module is built using Convolutional Recurrent Neural Networks (RNNs). In this context, the RNN acts as a spatio-temporal memory bank. It continuously extracts and holds onto important complementary features from earlier layers, feeding them back into the network to prevent information loss as the image is processed deeper into the model.

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

The following runs were evaluated from the folders under `trained_models/`. The interpretation combines:

- F1 evolution curves from `report_images/` (train vs validation dynamics),
- official platform metrics (accuracy, F1, precision, recall) reported after submission.

Extensive run-level metrics and artifacts are available in each run folder (`trained_models/<run_name>/`) and in TensorBoard (`trained_models/tensorboard_runs/`).

### 1. Common training setup (unless stated otherwise)

- Differential learning rate:
- Backbone LR: `1e-5`
- Head LR: `1e-4`
- Max epochs: `14`
- LR decay factor: `0.01`
- Validation split: `0.05` (train/val)
- Validation frequency: every epoch (`val_every_n_epochs = 1`)
- Early stopping: enabled (`patience = 4`, `min_delta = 0.0`)
- Unfrozen train batch size: `16`

For runs with full backbone unfreezing, LR milestones were set to `[5, 10]`.
For runs with partial backbone unfreezing, LR milestone was set to `[7]`.

### 2. Run-by-run analysis from F1 curves

#### ConvNeXt family

- `convnext_small_20260302-085858` (`convnext-small-f1.png`)
- Distinguishing hyperparameters for this run: `unfreeze_backbone_epoch = 5`, `unfreeze_last_n_backbone_layers = all` (full backbone unfreeze).
- Train and validation F1 increase steadily from ~0.50 and reach a stable plateau around ~0.61 (train) and ~0.60 (val).
- Gap remains relatively small, suggesting good generalization and limited overfitting.
- Platform metrics: accuracy `0.5248`, F1 `0.5904`, precision `0.4785`, recall `0.7708`.
- This run is recall-oriented and stable, but not the best F1 run overall.

- `convnext_tiny_20260302-122332` (`convnext-tiny-f1-full-unfrozen-epoch-1.png`)
- Distinguishing hyperparameters for this run: `unfreeze_backbone_epoch = 1`, `unfreeze_last_n_backbone_layers = all` (full backbone unfreeze).
- Fastest train F1 growth and highest train ceiling (~0.68), but validation saturates around ~0.62 with visible fluctuations.
- Clear train/validation gap after mid-training indicates overfitting.
- Platform metrics: accuracy `0.5585`, F1 `0.6086`, precision `0.5041`, recall `0.7679`.
- Despite visible overfitting in the curve, this run delivers the best platform F1 and the best accuracy in this experiment.

- `convnext_tiny_20260301-194510` (`convnext-tiny-f1-full-unfrozen-epoch-5.png`)
- Distinguishing hyperparameters for this run: `unfreeze_backbone_epoch = 5`, `unfreeze_last_n_backbone_layers = all` (full backbone unfreeze).
- Smoother progression than epoch-1 unfreeze; validation rises to ~0.595 with less instability.
- Smaller generalization gap than the epoch-1 full unfreeze run.
- Delaying full unfreeze reduces overfitting, but also slightly reduces peak validation F1.
- Platform metrics: accuracy `0.5079`, F1 `0.5944`, precision `0.5128`, recall `0.7070`.

- `convnext_tiny_20260301-134646` (`convnext-tiny-f1-3-layers-unfrozen.png`)
- Distinguishing hyperparameters for this run: `unfreeze_backbone_epoch = 5`, `unfreeze_last_n_backbone_layers = 3`.
- Train and validation both improve consistently and then flatten around ~0.61 (train) and ~0.59 (val).
- More conservative capacity than full unfreeze leads to stable learning with controlled gap.
- Platform metrics: accuracy `0.4919`, F1 `0.5931`, precision `0.5328`, recall `0.6688`.
- This is the highest-precision ConvNeXt run, with slightly lower recall and F1 than full unfreeze.

#### Swin / RegNet / MobileNet

- `swin_v2_t_20260301-144143` (`swint-f1.png`)
- Distinguishing hyperparameters for this run: `unfreeze_backbone_epoch = 5`, `unfreeze_last_n_backbone_layers = 3`.
- Good monotonic improvement and plateau around ~0.60 (train) and ~0.58 (val).
- Moderate and stable gap suggests balanced fine-tuning with partial unfreezing.
- Platform metrics: accuracy `0.4765`, F1 `0.5835`, precision `0.4956`, recall `0.7093`.
- Swin remains a strong middle-ground model, clearly above RegNet/MobileNet in F1 and recall.

- `regnet_y_800mf_20260301-161704` (`regnet-f1.png`)
- Distinguishing hyperparameters for this run: `unfreeze_backbone_epoch = 5`, `unfreeze_last_n_backbone_layers = 5`.
- Curves rise steadily but saturate lower (~0.55 train, ~0.52 val).
- Limited gap indicates little overfitting; main limitation appears to be lower representational performance on this setup.
- Platform metrics: accuracy `0.3794`, F1 `0.5173`, precision `0.4974`, recall `0.5389`.

- `mobilenet_v3_large_20260301-164829` (`mobilenet-f1.png`)
- Distinguishing hyperparameters for this run: `unfreeze_backbone_epoch = 5`, `unfreeze_last_n_backbone_layers = 5`.
- Lowest absolute F1 among the compared models (~0.53 train, ~0.51 val) with smooth convergence.
- Small train/val gap indicates strong regularization but underfitting relative to higher-capacity backbones.
- Platform metrics: accuracy `0.3252`, F1 `0.4896`, precision `0.5076`, recall `0.4728`.

The aggregate figure (`f1-score.png`) is consistent with the submission ranking: ConvNeXt variants are strongest, Swin is competitive but lower, and RegNet/MobileNet remain clearly behind on final F1.

### 3. Platform submission metrics

Official challenge metrics per run:

| Run folder | Run description | Accuracy | F1 | Precision | Recall |
| --- | --- | --- | --- | --- | --- |
| `convnext_small_20260302-085858` | ConvNeXt Small, Unfreeze all backbone layers at epoch 5 | 0.5248 | 0.5904 | 0.4785 | 0.7708 |
| `convnext_tiny_20260302-122332` | ConvNeXt Tiny, Unfreeze all backbone layers at epoch 1 | 0.5585 | 0.6086 | 0.5041 | 0.7679 |
| `convnext_tiny_20260301-194510` | ConvNeXt Tiny, Unfreeze all backbone layers at epoch 5 | 0.5079 | 0.5944 | 0.5128 | 0.7070 |
| `convnext_tiny_20260301-134646` | ConvNeXt Tiny, Unfreeze 3 backbone layers at epoch 5 | 0.4919 | 0.5931 | 0.5328 | 0.6688 |
| `swin_v2_t_20260301-144143` | Swin V2 Tiny, Unfreeze 3 backbone layers at epoch 5 | 0.4765 | 0.5835 | 0.4956 | 0.7093 |
| `regnet_y_800mf_20260301-161704` | RegNet Y 800MF, Unfreeze 5 backbone layers at epoch 5 | 0.3794 | 0.5173 | 0.4974 | 0.5389 |
| `mobilenet_v3_large_20260301-164829` | MobileNet V3 Large, Unfreeze 5 backbone layers at epoch 5 | 0.3252 | 0.4896 | 0.5076 | 0.4728 |

## VIII. Conclusion

---
