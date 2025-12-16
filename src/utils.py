# Includes helpter function (e.g. image transformations)
from torchvision import transforms


def transform_image(image):
    stats = ((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))

    test_transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(*stats)]
    )

    return test_transform(image)
