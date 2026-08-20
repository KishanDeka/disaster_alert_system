import os
import random
import streamlit as st
import torch
from PIL import Image

# Import from your package structure
from src.model import ScratchCNN
from src.utils import CLASS_NAMES, get_device, load_dataset_stats

# --- Page Configuration ---
st.set_page_config(
    page_title="Satellite Disaster Classifier",
    page_icon="🛰️",
    layout="wide"
)


# --- Load Model & Config (Cached) ---
@st.cache_resource
def load_model_and_config():
    device = get_device()
    weights_path = "data/best_model.pth"
    stats_path = "data/data_stats.csv"

    model = ScratchCNN(num_classes=len(CLASS_NAMES))
    
    if os.path.exists(weights_path):
        model.load_state_dict(torch.load(weights_path, map_location=device))
        model.to(device)
        model.eval()
    else:
        st.error(f"Model weights file `{weights_path}` not found! Please train the model first.")

    mean, std = None, None
    if os.path.exists(stats_path):
        mean, std = load_dataset_stats(stats_path)

    return model, device, mean, std


# --- Helper to sample random test images ---
def get_random_test_images(data_dir="./data", num_samples=4):
    """Recursively collects image paths from the Test directory and samples random paths."""
    test_dir = os.path.join(data_dir)
    valid_extensions = ('.png', '.jpg', '.jpeg', '.tif', '.tiff')
    
    image_paths = []
    if os.path.exists(data_dir):
        for root, _, files in os.walk(data_dir):
            for file in files:
                if file.lower().endswith(valid_extensions):
                    image_paths.append(os.path.join(root, file))
                    
    if len(image_paths) >= num_samples:
        return random.sample(image_paths, num_samples)
    return image_paths


# --- Main App Execution ---
st.title(" Satellite Imagery Disaster Detection")
st.markdown(
    "Upload a satellite image to classify natural disaster events "
    "*(Earthquake, Fire, Flood, Normal)* or run a batch prediction on test set samples."
)

model, device, mean, std = load_model_and_config()

# --- Sidebar Controls ---
st.sidebar.header(" Controls")

# Upload single image
uploaded_file = st.sidebar.file_uploader(
    "Upload Single Image...", 
    type=["jpg", "jpeg", "png", "tif", "tiff"]
)

# Button to trigger batch predictions
st.sidebar.markdown("---")
st.sidebar.subheader(" Batch Test Preview")
run_random_batch = st.sidebar.button("🎲 Predict 4 Random Test Images")


# --- Option A: Predict 4 Random Test Images ---
if run_random_batch:
    st.subheader(" Batch Test Results with Ground Truth")
    test_images = get_random_test_images(data_dir="./data", num_samples=4)
    
    if not test_images:
        st.warning("No images found in `./data/Test/` directory.")
    else:
        cols = st.columns(4)
        
        for idx, img_path in enumerate(test_images):
            image = Image.open(img_path).convert('RGB')
            
            # Extract true label from folder structure (e.g., ./data/Test/Fire/image1.jpg -> "Fire")
            true_label = os.path.basename(os.path.dirname(img_path))
            
            # Run inference
            pred_class, confidence, prob_dict = model.predict(image_input=image, mean=mean, std=std)
            
            with cols[idx]:
                st.image(image, use_container_width=True)
                
                # Check prediction accuracy for dynamic styling
                is_correct = pred_class.strip().lower() == true_label.strip().lower()
                status_icon = "✅" if is_correct else "❌"
                
                # Display Comparison Metrics
                st.markdown(f"**True Label:** `{true_label}`")
                st.markdown(f"**Predicted:** `{pred_class}` {status_icon}")
                
                # Top Confidence Progress Bar
                st.caption(f"Top Confidence: **{confidence:.1f}%**")
                st.progress(int(confidence))
                
                # Expander for full probability distribution
                with st.expander(" Probability Details"):
                    for c_name, prob in prob_dict.items():
                        prob_pct = int(prob * 100)
                        st.text(f"{c_name}: {prob_pct}%")
                        st.progress(prob_pct)

# --- Option B: Predict Single Uploaded Image ---
elif uploaded_file is not None:
    st.markdown("---")
    col1, col2 = st.columns(2)
    
    image = Image.open(uploaded_file).convert('RGB')
    
    with col1:
        st.subheader("Uploaded Image")
        st.image(image, use_container_width=True)

    with st.spinner("Analyzing image..."):
        pred_class, confidence, prob_dict = model.predict(image_input=image, mean=mean, std=std)

    with col2:
        st.subheader("Prediction Results")
        st.metric(
            label="Predicted Event", 
            value= pred_class, 
            delta= f"{confidence:.2f}% Confidence"
        )

        st.markdown("---")
        st.write("### Class Probabilities")
        for class_name, prob in prob_dict.items():
            prob_percent = prob * 100
            st.write(f"**{class_name}**: `{prob_percent:.2f}%`")
            st.progress(int(prob_percent))

# --- Default State ---
else:
    st.info("Upload an image or click **'Predict 4 Random Test Images'** in the sidebar to test predictions.")
