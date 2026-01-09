# Entry point for frontend application
import streamlit as st
import pandas as pd
from src.model_logic import ModelPredictor

# Page configuration
st.set_page_config(
    page_title="CIFAR-100 Image Classifier",
    page_icon="🤖",
    layout="wide",
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
    return ModelPredictor(model_path="cifar100_resnet34_v1.0.0.pth")


# Init model immediately
model = get_model()

# UI Header
st.title("CIFAR-100 Image Classifier")
st.markdown("Upload an image to see how ResNet-34 classifies it among 100 classes")

# Main Interface
col1, col2 = st.columns([1, 1.2], gap="large")

with col1:
    st.subheader("1. Input Image")
    uploaded_file = st.file_uploader(
        "Choose an image file (PNG, JPG, JPEG)", type=["png", "jpg", "jpeg"]
    )

    if uploaded_file:
        st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

with col2:
    st.subheader("2. Prediction Results")

    if uploaded_file:
        if st.button("Run Inference", type="primary"):
            # Proc and Feedback
            with st.spinner("Classifying..."):
                try:
                    # Mocking response for structure ex:
                    prediction, conf, top_5_dict = model.predict(uploaded_file)
                    # MOCK DATA
                    # prediction = "Beaver"
                    # confidence = 0.85
                    # top_5_data = {
                    #     "Beaver": 0.85,
                    #     "Otter": 0.10,
                    #     "Hamster": 0.03,
                    #     "Mouse": 0.01,
                    #     "Shrew": 0.01,
                    # }

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

                except Exception as e:
                    st.error(f"Error analyzing image: {e}")

    else:
        st.info("Please upload an image to see predictions.")
