# Entry point for frontend application
import streamlit as st
import pandas as pd
from src.model_logic import ModelPredictor
from PIL import Image

# 1. Get the transform from your new utils
from src.utils import transform_image
from torchvision import transforms

# Page configuration
st.set_page_config(
    page_title="CIFAR-100 Image Classifier",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .main { padding-top: 2rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


# Singleton Pattern. Runs only once per server restart
@st.cache_resource
def get_model():
    # Point to where stored weights
    return ModelPredictor(path_to_weights="cifar100_resnet34_v1.0.0.pth")


# Init model immediately
model = get_model()

# UI Header
st.title("CIFAR-100 Image Classifier")
st.markdown("Upload an image to see how ResNet-34 classifies it among 100 classes")

# Main Interface
col1, col2 = st.columns([1, 1.2], gap="large")

with col1:
    st.subheader("1. Input Image")
    # Tabs for different input methods
    tab1, tab2 = st.tabs(["Upload Image", "Image URL"])

    image_source = None

    # Tab 1
    with tab1:
        uploaded_file = st.file_uploader(
            "Choose an image file (PNG, JPG, JPEG)", type=["png", "jpg", "jpeg"]
        )
        if uploaded_file:
            image_source = uploaded_file

    # Tab 2
    with tab2:
        url = st.text_input("Past an image link (Right-click and copy image address)")
        if url:
            try:
                import requests
                from io import BytesIO

                # Download image in memory
                response = requests.get(url, timeout=5)
                response.raise_for_status()  # Check for 404
                image_source = BytesIO(response.content)
            except Exception as e:
                st.error(f"Error fetching image from URL: {e}")

    if image_source:
        st.image(image_source, caption="Input Image", use_column_width=True)

with col2:
    st.subheader("2. Prediction Results")

    if image_source:
        show_cam = st.checkbox("Show AI Attention Heatmap", value=False)
        if st.button("Run Inference", type="primary"):

            # --- DEBUGGING SECTION ---
            with st.expander("Show Internal Model Representation"):

                # 2. Apply it manually to see the result
                image = Image.open(image_source).convert("RGB")
                tensor_img = transform_image(image)

                # 3. Undo the normalization so we can view it as a human
                # (Multiply by std, add mean)
                inv_normalize = transforms.Normalize(
                    mean=[-0.5071 / 0.2675, -0.4867 / 0.2565, -0.4408 / 0.2761],
                    std=[1 / 0.2675, 1 / 0.2565, 1 / 0.2761],
                )
                debug_img = inv_normalize(tensor_img)

                # 4. Convert back to image for display
                to_pil = transforms.ToPILImage()
                st.image(
                    to_pil(debug_img),
                    caption="What the model actually sees (32x32)",
                    width=100,
                )

                st.write(f"Tensor Min: {tensor_img.min():.2f}")
                st.write(f"Tensor Max: {tensor_img.max():.2f}")
            # Proc and Feedback
            with st.spinner("Classifying..."):
                if show_cam:
                    top_class, conf, top_5_dict, heatmap = model.predict_with_heatmap(
                        image_source
                    )

                    # Show side by side
                    c1, c2 = st.columns(2)
                    c1.image(image_source, caption="Original Image")
                    c2.image(heatmap, caption="AI Attention Heatmap")
                else:
                    # Mocking response for structure ex:
                    prediction, conf, top_5_dict = model.predict(image_source)

                    # Display Results
                    st.success(f"Prediction: {prediction} ({conf*100:0.1f}%)")

                    # Progress bar for confidence
                    st.progress(int(conf * 100))

                    # Dataframe for breakdown
                    st.markdown("**Top 5 Predictions:**")
                    df = pd.DataFrame(
                        list(top_5_dict.items()), columns=["Class", "Confidence"]
                    )
                    st.bar_chart(df.set_index("Class"))

                # except Exception as e:
                #     st.error(f"Error analyzing image: {e}")

    else:
        st.info("Please upload an image to see predictions.")

# --- 5. SIDEBAR INFORMATION ---
with st.sidebar:
    st.title("Project Details")

    st.info(
        """
        **Architecture:** ResNet-34 (Modified)
        \n**Dataset:** CIFAR-100
        \n**Status:** Trained from Scratch
        \n**Accuracy:** ~80% (Top-1)
        """
    )

    st.markdown("### Description")
    st.write(
        """
        This project demonstrates a full deep learning pipeline, from architecture design to cloud deployment.
        
        The model is a **ResNet-34** CNN, modified to handle small $32\\times32$ images (removing the initial 7x7 pooling layers to preserve feature spatial resolution).
        
        It was trained for **200 epochs** using:
        * **SGD with Momentum**
        * **Mixup Augmentation**
        * **Cosine Annealing / Step LR**
        """
    )

    st.markdown("### 🛠️ Tech Stack")
    st.write(
        """
        * **PyTorch** (Model Training)
        * **Streamlit** (Web Interface)
        * **Grad-CAM** (Explainability)
        * **OpenCV** (Image Processing)
        """
    )

    st.markdown("---")
    st.caption("Built by Jacob Alcumbrack")
    st.caption("© 2026")
