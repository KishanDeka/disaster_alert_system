import streamlit as st

# Import from your package structure
from src.utils import *
from src.dataset import CSVDataset
from src.model import ScratchCNN

# --- Page Configuration ---
st.set_page_config(
    page_title="Satellite Disaster Classifier",
    page_icon="🛰️",
    layout="wide"
)
    

# --- Helper to sample random test images ---
def get_random_test_images(data_dir="data/", num_samples=4):
    """Recursively collects image paths from the Test directory and samples random paths."""
    test_dir = os.path.join(data_dir,'testdata/')
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
    

# Function to render custom-colored progress bar on the same line
def render_custom_probability_bar(class_name, prob_percent, color="#4A90E2"):
    """Renders class name, progress bar, and percentage tag on a single flex line."""
    st.markdown(
        f"""
        <div style="display: flex; align-items: center; justify-content: space-between; gap: 8px; margin-bottom: 6px; font-size: 0.85rem;">
            <span style="font-weight: 600; min-width: 65px; white-space: nowrap;">{class_name}</span>
            <div style="background-color: #e0e0e0; border-radius: 6px; height: 12px; flex-grow: 1; overflow: hidden;">
                <div style="background-color: {color}; width: {prob_percent}%; height: 100%; border-radius: 6px; transition: width 0.4s ease;"></div>
            </div>
            <span style="font-family: monospace; min-width: 45px; text-align: right; white-space: nowrap;">{prob_percent:.1f}%</span>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    
# --- Main App Execution ---
st.title(" Satellite Imagery Disaster Detection")
st.markdown(
    "Upload a satellite image to classify natural disaster events "
    "*(Earthquake, Fire, Flood, Normal)* or run a batch prediction on test set samples."
)

# --- Load Model & Config ---
@st.cache_resource
model, device, mean, std = load_model_and_config(weights_path = 'data/model/best_model.pth', stats_path = "data/data_stats.csv")
fig_path, metrics_df, summary = load_saved_evaluation_assets(output_dir="data/summary/")

tab_inference, tab_analytics = st.tabs([" Real-Time Inference", " Model Performance & Metrics"])


# ==========================================
# TAB 1: REAL-TIME INFERENCE
# ==========================================
with tab_inference:

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
        test_images = get_random_test_images(data_dir="data/", num_samples=4)
        
        if not test_images:
            st.warning("No images found in `data/` directory.")
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
                    
                    # Inside run_random_batch column loop:
                    with st.expander("📊 Probabilities"):
                        for idx, (class_name, prob) in enumerate(prob_dict.items()):
                            prob_percent = prob * 100
                            # Safe indexing into CLASS_COLORS array
                            bar_color = CLASS_COLORS[idx] if idx < len(CLASS_COLORS) else "#4A90E2"
                            
                            render_custom_probability_bar(
                                class_name=class_name,
                                prob_percent=prob_percent,
                                color=bar_color
                            )


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
            for idx, (class_name, prob) in enumerate(prob_dict.items()):
                prob_percent = prob * 100
                # Safe indexing into CLASS_COLORS array
                bar_color = CLASS_COLORS[idx] if idx < len(CLASS_COLORS) else "#4A90E2"
                
                render_custom_probability_bar(
                    class_name=class_name,
                    prob_percent=prob_percent,
                    color=bar_color
                )

    # --- Default State ---
    else:
        st.info("Upload an image or click **'Predict 4 Random Test Images'** in the sidebar to test predictions.")
        

# ==========================================
# TAB 2: MODEL PERFORMANCE & METRICS
# ==========================================
with tab_analytics:
    st.subheader(" Test Set Evaluation & Proof of Performance")
    st.markdown("Evaluate how the model performs across the full test set using the evaluation pipeline.")

    if st.button(" Run Full Test Set Evaluation"):
        csv_file = "data/data_list.csv"
        
        if not os.path.exists(csv_file):
            st.error(f"Metadata file `{csv_file}` not found. Cannot load test dataset.")
        else:
            with st.spinner("Evaluating entire test dataset..."):
                # Define evaluation transforms
                test_transform = T.Compose([
                    T.Resize((128, 128)),
                    T.ToTensor(),
                    T.Normalize(mean=mean, std=std)
                    ])

                # Top-level key performance metrics
                st.markdown("---")
                m_col1, m_col2 = st.columns(2)
                m_col1.metric("Overall Accuracy", f"{summary['accuracy']:.2f}%")
                m_col2.metric("Macro F1-Score", f"{summary['macro_f1']:.2f}%")
                st.markdown("---")

                # Layout: Confusion Matrix & Class Table
                eval_col1, eval_col2 = st.columns([1, 1.2])

                with eval_col1:
                    st.markdown("### Confusion Matrix")
                    st.image(fig_path, use_container_width=True)

                with eval_col2:
                    st.markdown("### Per-Class Performance Metrics")
                    st.dataframe(metrics_df, hide_index=True, use_container_width=True)

                    # Metric Explanation Box
                    with st.expander(" What do these metrics mean?", expanded=True):
                        st.markdown("""
                        * **Precision (Exactness):** Measures how many of the images predicted as a specific class actually belonged to that class.
                          $$\\text{Precision} = \\frac{\\text{True Positives}}{\\text{True Positives} + \\text{False Positives}}$$
                          *High precision means low false alarms.*

                        * **Recall (Completeness):** Measures how many of the actual disaster images in the dataset were correctly captured by the model.
                          $$\\text{Recall} = \\frac{\\text{True Positives}}{\\text{True Positives} + \\text{False Negatives}}$$
                          *High recall means few missed disaster cases.*

                        * **F1-Score (Balance):** The harmonic mean of Precision and Recall. It provides a single robust score, especially if classes are imbalanced.

                        * **Support:** The total count of ground-truth test images evaluated for each category.
                        """)
