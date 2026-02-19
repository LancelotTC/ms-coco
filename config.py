from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
LOCAL_FOLDER = Path.home() / "ms-coco"

DATASET_FOLDER = LOCAL_FOLDER / "ms-coco-dataset"
PRETRAINED_MODELS_FOLDER = LOCAL_FOLDER / "pre-trained_models"
TRAINED_MODELS_FOLDER = PROJECT_ROOT / "trained_models"

IMAGES_DIR = DATASET_FOLDER / "images"
LABELS_DIR = DATASET_FOLDER / "labels"

TRAIN_IMAGES_DIR = IMAGES_DIR / "train-resized"
TEST_IMAGES_DIR = IMAGES_DIR / "test-resized"
TRAIN_LABELS_DIR = LABELS_DIR / "train"


NUM_CLASSES = 80

# All tested models retained for ease of use. Only last is considered.
MODEL_NAME = "efficientnet_b0"
MODEL_NAME = "regnet_y_800mf"
MODEL_NAME = "mobilenet_v3_large"
MODEL_NAME = "swin_v2_t"
MODEL_NAME = "efficientnet_v2_s"
MODEL_NAME = "convnext_tiny"
MODEL_NAME = "convnext_small"

BEST_MODEL_PATH = TRAINED_MODELS_FOLDER / "best_model.pt"
FREEZE_BACKBONE = True
TRAIN_METRICS_EVERY_N_EPOCHS = 0
VAL_EVERY_N_EPOCHS = 1

CLASSES = (
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
)
