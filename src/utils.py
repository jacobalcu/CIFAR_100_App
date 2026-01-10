# Includes helpter function (e.g. image transformations)
from torchvision import transforms
import numpy as np
import cv2
import torch


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # Hook into backward pass to grab gradients
        self.target_layer.register_full_backward_hook(self.save_gradients)
        # Hook into forward pass to grab activations
        self.target_layer.register_forward_hook(self.save_activations)

    def save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def save_activations(self, module, input, output):
        self.activations = output

    def generate_heatmap(self, input_tensor, class_idx=None):
        # Zero gradients
        self.model.zero_grad()

        # Forward Pass
        output = self.model(input_tensor)

        # If not class spec, viz highest prob class
        if class_idx is None:
            class_idx = torch.argmax(output, dim=1).item()

        # Backward pass to get grads for specific class
        # Set target to 1.0 for class we want, 0 for others
        one_hot = torch.zeros_like(output)
        one_hot[0, class_idx] = 1.0
        output.backward(gradient=one_hot, retain_graph=True)

        # Global Average Pooling of gradients
        pooled_grads = torch.mean(self.gradients, dim=[0, 2, 3])

        # Weight activations by gradients
        activations = self.activations[0]
        for i in range(activations.shape[0]):
            activations[i, :, :] *= pooled_grads[i]

        # Avg the channels to get heatmap
        heatmap = torch.mean(activations, dim=0).cpu().detach().numpy()

        # Apply ReLU for positive influence
        heatmap = np.maximum(heatmap, 0)

        # Norm between 0-1
        heatmap /= np.max(heatmap) if np.max(heatmap) != 0 else 1

        return heatmap


def overlay_heatmap(heatmap, original_image_pil, alpha=0.4):
    # Convert PIL image to OpenCV format (rgb->bgr)
    img_np = np.array(original_image_pil)
    img_cv = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    # Resize heatmap to match image size
    heatmap_resized = cv2.resize(heatmap, (img_cv.shape[1], img_cv.shape[0]))

    # Colorize heatmap
    heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)

    # Overlay heatmap on original image
    overlayed_image = cv2.addWeighted(img_cv, 1 - alpha, heatmap_color, alpha, 0)

    # Convert back to RGB for PIL
    overlayed_image = cv2.cvtColor(overlayed_image, cv2.COLOR_BGR2RGB)

    return overlayed_image


def transform_image(image):
    stats = ((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))

    test_transform = transforms.Compose(
        [
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize(*stats),
        ]
    )

    return test_transform(image)
