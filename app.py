import streamlit as st
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from src.model import create_pneumonia_model

st.title("🫁 Medical Imaging: Pneumonia Detector")
st.write("Upload a pediatric chest X-ray to predict the probability of Pneumonia.")

# Transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

@st.cache_resource
def load_model():
    model = create_pneumonia_model()
    model.load_state_dict(torch.load("pneumonia_model.pth", map_location=torch.device('cpu')))
    model.eval() 
    return model

model = load_model()

uploaded_file = st.file_uploader("Choose an X-ray image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Image', use_container_width=True)
    
    st.write("Analyzing...")
    
    img_tensor = transform(image).unsqueeze(0) 
    
    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = F.softmax(outputs, dim=1)[0] * 100
        
        # PyTorch alphabetizes classes: 0=NORMAL, 1=OTHER, 2=PNEUMONIA
        prob_normal = probabilities[0].item()
        prob_other = probabilities[1].item()
        prob_pneumonia = probabilities[2].item()
        
    st.subheader("Diagnosis Results")
    
    # 1. Check if the model thinks this is "Garbage" (OTHER)
    if prob_other > prob_normal and prob_other > prob_pneumonia:
        st.warning(f"**INVALID IMAGE DETECTED** ({prob_other:.2f}% confidence)")
        st.write("This does not appear to be a valid chest X-ray. Please upload a medical scan.")
        
    # 2. Otherwise, proceed with the medical diagnosis
    else:
        if prob_pneumonia > prob_normal:
            st.error(f"**PREDICTION: PNEUMONIA** ({prob_pneumonia:.2f}% confidence)")
        else:
            st.success(f"**PREDICTION: NORMAL** ({prob_normal:.2f}% confidence)")
            
    # Always show the math behind the decision
    st.write("---")
    st.write("Confidence Breakdown:")
    st.write(f"- Normal: {prob_normal:.2f}%")
    st.write(f"- Pneumonia: {prob_pneumonia:.2f}%")
    st.write(f"- Invalid/Other: {prob_other:.2f}%")