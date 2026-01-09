import os
import sys
import numpy as np
from PIL import Image
import torch

# Ensure we can import from src/
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

from src.model_logic import ModelPredictor


def run_sanity_check():
    print("Running Sanity Check...")

    # Verify paths
    model_path = "cifar100_resnet34_v1.0.0.pth"
    if not os.path.exists(model_path):
        print(f"Model weights not found at {model_path}")
        return
    print("Model weights found.")

    # Init Predictor
    print("Initializing ModelPredictor...")
    try:
        predictor = ModelPredictor(model_path, device="cpu")
        print("ModelPredictor initialized successfully.")
    except Exception as e:
        print(f"Error initializing ModelPredictor: {e}")
        return

    # Create synthetic image (32x32 RGB)
    print("Creating synthetic image...")
    rand_array = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
    synthetic_image = Image.fromarray(rand_array)

    # Save temp to test full file-loading pipeline
    temp_image_path = "temp_test_image.jpg"
    synthetic_image.save(temp_image_path)

    # Run prediction
    print("Running prediction on synthetic image...")
    try:
        prediction, conf, top_5_dict = predictor.predict(temp_image_path)

        # Validation
        assert isinstance(prediction, str), "Prediction should be a string."
        assert isinstance(conf, float), "Confidence should be a float."
        assert isinstance(top_5_dict, dict), "Top 5 should be a dict."
        assert len(top_5_dict) == 5, "Top 5 dict should have 5 entries."

        print("Prediction successful and validated.")
        print(f"Prediction: {prediction}, Confidence: {conf:.4f}")
        print("Top 5 Predictions:")
        print(top_5_dict)

    except AssertionError as ae:
        print(f"Assertion Error: {ae}")
    except Exception as e:
        print(f"Error during prediction: {e}")
    finally:
        # Clean up temp image
        if os.path.exists(temp_image_path):
            os.remove(temp_image_path)
        print("Sanity Check Completed.")


if __name__ == "__main__":
    run_sanity_check()
