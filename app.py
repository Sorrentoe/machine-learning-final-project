import streamlit as st
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from src.model import create_pneumonia_model

# --- Decision thresholds (tunable) ---
# Binary gate: probability the image is a chest X-ray (vs junk) from dedicated head.
XRAY_GATE_THRESHOLD = 0.52
# OTHER softmax % above which we treat the upload as non–X-ray.
OTHER_CONFIDENCE_THRESHOLD_PCT = 32.0
# Medical class must beat OTHER by at least this margin (percentage points).
MEDICAL_MARGIN_OVER_OTHER_PCT = 15.0
# Normalized entropy above this → do not show a crisp diagnosis (uniform 3-way ≈ 1.099).
ENTROPY_UNCERTAIN_MIN = 0.92

st.title("🫁 Medical Imaging: Pneumonia Detector")
st.write("Upload a pediatric chest X-ray to predict the probability of Pneumonia.")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


@st.cache_resource
def load_model():
    model = create_pneumonia_model()
    model.load_state_dict(torch.load("pneumonia_model.pth", map_location=torch.device("cpu")))
    model.eval()
    return model


def softmax_entropy(probs_1d: torch.Tensor) -> float:
    p = probs_1d.clamp(min=1e-9)
    return float(-(p * p.log()).sum().item())


model = load_model()

uploaded_file = st.file_uploader("Choose an X-ray image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    st.write("Analyzing...")

    img_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        logits_3, xray_logit = model(img_tensor)
        probs = F.softmax(logits_3, dim=1)[0]
        probabilities_pct = probs * 100

        prob_normal = probabilities_pct[0].item()
        prob_other = probabilities_pct[1].item()
        prob_pneumonia = probabilities_pct[2].item()

        xray_prob = torch.sigmoid(xray_logit).item()
        top_medical = max(prob_normal, prob_pneumonia)
        medical_margin_vs_other = top_medical - prob_other
        entropy = softmax_entropy(probs)

    st.subheader("Diagnosis Results")

    is_invalid = (
        xray_prob < XRAY_GATE_THRESHOLD
        or prob_other >= OTHER_CONFIDENCE_THRESHOLD_PCT
        or medical_margin_vs_other < MEDICAL_MARGIN_OVER_OTHER_PCT
    )
    is_uncertain = (not is_invalid) and entropy >= ENTROPY_UNCERTAIN_MIN

    if is_invalid:
        st.warning("**INVALID OR UNSUITABLE IMAGE**")
        st.write("This does not appear to be a valid chest X-ray. Please upload a medical scan.")
    elif is_uncertain:
        st.info(
            f"**LOW CONFIDENCE — NO CLEAR DIAGNOSIS** "
            f"(distribution entropy {entropy:.2f}; model is not confident this is a typical scan.)"
        )
        st.write(
            "If this is a real pediatric chest X-ray, consider a clinical read; "
            "otherwise upload a clearer image."
        )
    else:
        if prob_pneumonia > prob_normal:
            st.error(f"**PREDICTION: PNEUMONIA** ({prob_pneumonia:.2f}% confidence)")
        else:
            st.success(f"**PREDICTION: NORMAL** ({prob_normal:.2f}% confidence)")

    st.write("---")
    st.write("Confidence breakdown:")
    st.write(f"- Normal: {prob_normal:.2f}%")
    st.write(f"- Pneumonia: {prob_pneumonia:.2f}%")
    st.write(f"- Invalid / other: {prob_other:.2f}%")
