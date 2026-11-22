from pathlib import Path

import streamlit as st
import torch
from PIL import Image
from torchvision import models, transforms


CHECKPOINT = Path(__file__).resolve().parents[1] / "models" / "mobilenet_food101_interrupted_best.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@st.cache_resource
def load_model():
	checkpoint = torch.load(CHECKPOINT, map_location=DEVICE)

	if isinstance(checkpoint, torch.nn.Module):
		model = checkpoint
	else:
		num_classes = checkpoint.get("num_classes", 101)
		model = models.mobilenet_v2(weights=None)
		model.classifier[1] = torch.nn.Linear(
			model.classifier[1].in_features,
			num_classes,
		)
		state_dict = checkpoint.get("state_dict", checkpoint.get("model_state_dict", checkpoint))
		state_dict = {
			key.removeprefix("module."): value for key, value in state_dict.items()
		}
		model.load_state_dict(state_dict)

	model.to(DEVICE).eval()
	class_names = checkpoint.get("class_names") if isinstance(checkpoint, dict) else None
	return model, class_names


st.set_page_config(page_title="Food-101 classifier", page_icon="🍴")
st.title("Food-101 image classifier")

if not CHECKPOINT.exists():
	st.error(f"Checkpoint not found: {CHECKPOINT}")
	st.stop()

uploaded_file = st.file_uploader("Upload a food image", type=["jpg", "jpeg", "png"])

if uploaded_file:
	image = Image.open(uploaded_file).convert("RGB")
	st.image(image, caption="Uploaded image", use_container_width=True)

	preprocess = transforms.Compose(
		[
			transforms.Resize(256),
			transforms.CenterCrop(224),
			transforms.ToTensor(),
			transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
		]
	)

	with st.spinner("Classifying..."):
		model, class_names = load_model()
		with torch.inference_mode():
			probabilities = model(preprocess(image).unsqueeze(0).to(DEVICE)).softmax(1)[0]
		scores, indices = probabilities.topk(5)

	st.subheader("Food Type")
	top_index = indices[0].item()
	top_score = scores[0].item()
	st.write(
		f"**{class_names[top_index]}** "
		f"with a confidence of {top_score * 100:.2f}%"
	)