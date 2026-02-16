from torchvision.models import ResNet50_Weights
from config import CLASSES
from models_factory import MODEL_SPECS
from pprint import pprint

models_common = {}

for name, spec in MODEL_SPECS.items():
    models_common[name] = (set(spec.weights.meta["categories"]).intersection(set(CLASSES)))

pprint(models_common)
