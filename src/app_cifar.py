from pathlib import Path

import streamlit as st
import torch
from PIL import Image

from utils import (
    build_inference_transform,
    build_model,
    load_checkpoint,
    predict,
    preprocess_image,
)


CHECKPOINT = (
    Path(__file__).resolve().parents[1]
    / "models"
    / "resnet18_cifar10.pth"
)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@st.cache_resource
def load_model_for_inference(checkpoint_path: Path, device: torch.device):
    checkpoint = load_checkpoint(checkpoint_path, device)
    model = build_model(checkpoint, device)
    transform = build_inference_transform(checkpoint)
    return model, checkpoint["class_names"], transform


def main() -> None:
    st.set_page_config(page_title="CIFAR-10 classifier", page_icon="🖼️")
    st.title("CIFAR-10 image classifier")

    if not CHECKPOINT.exists():
        st.error(f"Checkpoint not found: {CHECKPOINT}")
        st.stop()

    model, class_names, transform = load_model_for_inference(
        CHECKPOINT,
        DEVICE,
    )

    uploaded_file = st.file_uploader(
        "Upload an image",
        type=["jpg", "jpeg", "png"],
    )
    if uploaded_file is None:
        return

    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded image", use_container_width=True)

    image_tensor = preprocess_image(image, transform)
    predictions = predict(
        model,
        image_tensor,
        class_names,
        DEVICE,
    )

    if st.button("What is this?"):
        # Display the top prediction
        top_prediction = predictions[0]
        st.write(f"This is a/an {top_prediction[0]}")

if __name__ == "__main__":
    main()