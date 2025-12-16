# Handles loading the model and processing images
import torch
import torch.nn as nn
from torchvision.models import resnet34
from utils import transform_image

class_names = {
    0: "apple",
    1: "aquarium_fish",
    2: "baby",
    3: "bear",
    4: "beaver",
    5: "bed",
    6: "bee",
    7: "beetle",
    8: "bicycle",
    9: "bottle",
    10: "bowl",
    11: "boy",
    12: "bridge",
    13: "bus",
    14: "butterfly",
    15: "camel",
    16: "can",
    17: "castle",
    18: "caterpillar",
    19: "cattle",
    20: "chair",
    21: "chimpanzee",
    22: "clock",
    23: "cloud",
    24: "cockroach",
    25: "couch",
    26: "crab",
    27: "crocodile",
    28: "cup",
    29: "dinosaur",
    30: "dolphin",
    31: "elephant",
    32: "flatfish",
    33: "forest",
    34: "fox",
    35: "girl",
    36: "hamster",
    37: "house",
    38: "kangaroo",
    39: "keyboard",
    40: "lamp",
    41: "lawn_mower",
    42: "leopard",
    43: "lion",
    44: "lizard",
    45: "lobster",
    46: "man",
    47: "maple_tree",
    48: "motorcycle",
    49: "mountain",
    50: "mouse",
    51: "mushroom",
    52: "oak_tree",
    53: "orange",
    54: "orchid",
    55: "otter",
    56: "palm_tree",
    57: "pear",
    58: "pickup_truck",
    59: "pine_tree",
    60: "plain",
    61: "plate",
    62: "poppy",
    63: "porcupine",
    64: "possum",
    65: "rabbit",
    66: "raccoon",
    67: "ray",
    68: "road",
    69: "rocket",
    70: "rose",
    71: "sea",
    72: "seal",
    73: "shark",
    74: "shrew",
    75: "skunk",
    76: "skyscraper",
    77: "snail",
    78: "snake",
    79: "spider",
    80: "squirrel",
    81: "streetcar",
    82: "sunflower",
    83: "sweet_pepper",
    84: "table",
    85: "tank",
    86: "telephone",
    87: "television",
    88: "tiger",
    89: "tractor",
    90: "train",
    91: "trout",
    92: "tulip",
    93: "turtle",
    94: "wardrobe",
    95: "whale",
    96: "willow_tree",
    97: "wolf",
    98: "woman",
    99: "worm",
}


# Define architecture
class CifarResNet34PM(nn.Module):
    def __init__(self, num_classes=100):
        super(CifarResNet34PM, self).__init__()
        # Load standard ResNet-34 backbone
        self.backbone = resnet34(weights=None)

        # Modify
        # Replace first layer: 3x3 kernel rather than 7x7
        # Keeps spatial res high
        self.backbone.conv1 = nn.Conv2d(
            3, 64, kernel_size=3, stride=1, padding=1, bias=False
        )

        # Remove first pooling layer (lose info on small images)
        self.backbone.maxpool = nn.Identity()

        # Replace final layer with 100 classes
        self.backbone.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        return self.backbone(x)


class ModelPredictor:
    def __init__(self, path_to_weights: str, device: str = "cpu"):
        self.device = device

        # Load model
        self.model = CifarResNet34PM(num_classes=100)
        self.model.load_state_dict(torch.load(path_to_weights, map_location=device))
        self.model.to(device)
        self.model.eval()

    def predict(self, image):
        """
        Accepts raw image
        Applies necessary transformations
        Runs forward pass through model
        Returns predicted class name and confidence score

        :param self:
        :param image: Input image
        """
        # Transform image
        input_tensor = transform_image(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(input_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted_idx = torch.max(probabilities, dim=1)

        predicted_class = class_names[predicted_idx.item()]
        confidence_score = confidence.item()

        return predicted_class, confidence_score
