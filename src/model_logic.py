# Handles loading the model and processing images
import torch
import torch.nn as nn
from torchvision.models import resnet34
from src.utils import transform_image, GradCAM, overlay_heatmap
import numpy as np
import io
from PIL import Image


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
        self.labels = self._get_cifar100_labels()
        self.model = CifarResNet34PM(num_classes=100)
        self.model.load_state_dict(torch.load(path_to_weights, map_location=device))
        self.model.to(device)
        self.model.eval()

    def predict(self, image_file):
        """
        Accepts raw image
        Applies necessary transformations
        Runs forward pass through model
        Returns predicted class name and confidence score

        :param self:
        :param image_file: Input image

        Returns:
            top_class (str): The name of the highest probability class.
            top_conf (float): The confidence score (0-1).
            top_5_dict (dict): Mapping of {Class Name: Confidence} for plotting.
        """
        # Preprocess Image
        # Handle cases where image_file is in bytes
        image = Image.open(image_file).convert("RGB")

        # Transform image
        input_tensor = transform_image(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(input_tensor)
            probabilities = torch.softmax(outputs, dim=1)

            # Get Top K results
            # values: the probabilities
            # indices: the class ID's
            top_probs, top_indices = torch.topk(probabilities, k=5)

            # confidence, predicted_idx = torch.max(probabilities, dim=1)

        # Format results
        top_probs = top_probs.cpu().numpy()[0]
        top_indices = top_indices.cpu().numpy()[0]

        top_5_dict = {}
        for i in range(len(top_indices)):
            class_name = self.labels[top_indices[i]]
            conf = float(top_probs[i])
            top_5_dict[class_name] = conf

        # predicted_class = class_names[predicted_idx.item()]
        # confidence_score = confidence.item()

        # The winner
        top_class = self.labels[top_indices[0]]
        top_conf = float(top_probs[0])

        return top_class, top_conf, top_5_dict

        return predicted_class, confidence_score

    def predict_with_heatmap(self, image_file):
        # Preprocess Image
        orig_image = Image.open(image_file).convert("RGB")

        # Create tensor
        input_tensor = transform_image(orig_image).unsqueeze(0).to(self.device)
        input_tensor.requires_grad = True

        # Init GradCAM on last layer of ResNet
        target_layer = self.model.layer4[-1]
        grad_cam = GradCAM(self.model, target_layer)

        # Generate heatmap
        heatmap = grad_cam.generate_heatmap(input_tensor)

        # Overlay
        heatmap_overlay = overlay_heatmap(heatmap, orig_image)

        # Get prediction as before
        top_class, top_conf, top_5_dict = self.predict(image_file)

        return top_class, top_conf, top_5_dict, heatmap_overlay

    def _get_cifar100_labels(self):
        # Ordered list of CIFAR-100 Fine Labels (Indices 0-99)
        return [
            "apple",
            "aquarium_fish",
            "baby",
            "bear",
            "beaver",
            "bed",
            "bee",
            "beetle",
            "bicycle",
            "bottle",
            "bowl",
            "boy",
            "bridge",
            "bus",
            "butterfly",
            "camel",
            "can",
            "castle",
            "caterpillar",
            "cattle",
            "chair",
            "chimpanzee",
            "clock",
            "cloud",
            "cockroach",
            "couch",
            "crab",
            "crocodile",
            "cup",
            "dinosaur",
            "dolphin",
            "elephant",
            "flatfish",
            "forest",
            "fox",
            "girl",
            "hamster",
            "house",
            "kangaroo",
            "keyboard",
            "lamp",
            "lawn_mower",
            "leopard",
            "lion",
            "lizard",
            "lobster",
            "man",
            "maple_tree",
            "motorcycle",
            "mountain",
            "mouse",
            "mushroom",
            "oak_tree",
            "orange",
            "orchid",
            "otter",
            "palm_tree",
            "pear",
            "pickup_truck",
            "pine_tree",
            "plain",
            "plate",
            "poppy",
            "porcupine",
            "possum",
            "rabbit",
            "raccoon",
            "ray",
            "road",
            "rocket",
            "rose",
            "sea",
            "seal",
            "shark",
            "shrew",
            "skunk",
            "skyscraper",
            "snail",
            "snake",
            "spider",
            "squirrel",
            "streetcar",
            "sunflower",
            "sweet_pepper",
            "table",
            "tank",
            "telephone",
            "television",
            "tiger",
            "tractor",
            "train",
            "trout",
            "tulip",
            "turtle",
            "wardrobe",
            "whale",
            "willow_tree",
            "wolf",
            "woman",
            "worm",
        ]
