import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image, ImageDraw
from streamlit_drawable_canvas import st_canvas
import matplotlib.pyplot as plt
import time
import base64
from io import BytesIO

# --- Page Configuration ---
st.set_page_config(layout="wide", page_title="BioVision Analytics")

# --- Session State Initialization ---
if 'models' not in st.session_state:
    st.session_state.models = [
        {"name": "Nuclei Counter v1", "author": "System", "type": "Nuclei", "accuracy": "94%"},
        {"name": "Cytoplasm Segmenter", "author": "System", "type": "Cytoplasm", "accuracy": "89%"}
    ]
if 'generated_data' not in st.session_state:
    st.session_state.generated_data = None

# --- DYNAMIC CLASS STATE ---
if 'classes' not in st.session_state:
    st.session_state.classes = [
        {"name": "Nuclei", "color": "#0000FF"},      # Blue
        {"name": "Cytoplasm", "color": "#FFFF00"},   # Yellow
        {"name": "Background", "color": "#FF0000"}   # Red
    ]

# --- Helper Functions ---
def mock_predict_image(image, brush_data):
    img_array = np.array(image)
    mask = np.zeros_like(img_array)
    mask[:, :, 0] = 100 
    return Image.fromarray(mask)

def image_to_base64(image):
    """Converts a PIL Image to a base64 string for st_canvas."""
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return f"data:image/png;base64,{img_str}"

# --- Navigation ---
st.sidebar.title("🧬 VolkCell Analytics")
page = st.sidebar.radio("Navigation", ["Model Studio", "Batch Runner", "Marketplace", "Analysis Lab"])

# ==========================================
# 1. MODEL STUDIO
# ==========================================
if page == "Model Studio":
    st.title("🖌️ Model Studio")
    st.markdown("Paint on the image to train your classifier. The model learns in real-time.")

    col1, col2, col3 = st.columns([1, 3, 1])

    with col1:
        st.subheader("Palette")
        
        # Class Selection
        class_options = [c['name'] for c in st.session_state.classes]
        
        if class_options:
            selected_class_name = st.radio("Active Class", class_options)
            
            # Find color
            selected_class = next((c for c in st.session_state.classes if c['name'] == selected_class_name), None)
            stroke_color = selected_class['color'] if selected_class else "#FFFFFF"
            
            st.caption(f"Color: {stroke_color}")
            
            if st.button(f"🗑️ Remove '{selected_class_name}'"):
                st.session_state.classes = [c for c in st.session_state.classes if c['name'] != selected_class_name]
                st.rerun()
        else:
            st.warning("No classes defined. Add one below.")
            stroke_color = "#FFFFFF"

        st.markdown("---")
        
        # Add New Class
        with st.expander("➕ Add New Class", expanded=False):
            new_name = st.text_input("Class Name", "Mitochondria")
            new_color = st.color_picker("Class Color", "#00FF00")
            
            if st.button("Add Class"):
                if any(c['name'] == new_name for c in st.session_state.classes):
                    st.error("Class already exists!")
                else:
                    st.session_state.classes.append({"name": new_name, "color": new_color})
                    st.rerun()

        st.markdown("---")
        brush_size = st.slider("Brush Size", 1, 50, 10)
        
        if st.button("Train Model"):
            with st.spinner("Updating weights..."):
                time.sleep(1.0)
            st.success("Model Updated!")

    with col2:
        st.subheader("Viewport")
        upload_img = st.file_uploader("Upload Training Image", type=["png", "jpg", "tif"])
        
        if upload_img:
            bg_image = Image.open(upload_img).resize((600, 400))
        else:
            bg_image = Image.new('RGB', (600, 400), color = (73, 109, 137))
            d = ImageDraw.Draw(bg_image)
            d.text((250,200), "Upload an Image", fill=(255,255,0))

        # FIX: Convert PIL image to Base64 string before passing to canvas
        # This bypasses the broken 'image_to_url' function in Streamlit
        bg_image_url = image_to_base64(bg_image)

        # Canvas
        canvas_result = st_canvas(
            fill_color="rgba(255, 165, 0, 0.3)",  
            stroke_width=brush_size,
            stroke_color=stroke_color,
            background_image=bg_image_url, # Changed from bg_image to bg_image_url
            update_streamlit=True,
            height=400,
            width=600,
            drawing_mode="freedraw",
            key="canvas",
        )

    with col3:
        st.subheader("Validation")
        # Fixed width parameter
        st.image(bg_image, caption="Live Prediction", use_container_width=True)
        
        st.metric("Model Confidence", "88%", "+12%")
        
        st.markdown("---")
        model_name = st.text_input("Model Name", "My Custom Model")
        if st.button("Freeze & Deploy", type="primary"):
            new_model = {"name": model_name, "author": "You", "type": "Custom", "accuracy": "Untested"}
            st.session_state.models.append(new_model)
            st.balloons()
            st.success(f"Deployed {model_name}!")

# ==========================================
# 2. BATCH RUNNER
# ==========================================
elif page == "Batch Runner":
    st.title("🚀 Batch Runner")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("1. Select Model")
        model_names = [m['name'] for m in st.session_state.models]
        selected_model = st.selectbox("Choose a model", model_names)
    
    with col2:
        st.subheader("2. Upload Data")
        uploaded_files = st.file_uploader("Upload images", accept_multiple_files=True)
    
    st.markdown("---")
    
    if st.button("Start Analysis"):
        if not uploaded_files:
            st.warning("Please upload files first.")
        else:
            progress_bar = st.progress(0)
            data_rows = []
            for i, file in enumerate(uploaded_files):
                time.sleep(0.5) 
                progress_bar.progress((i + 1) / len(uploaded_files))
                
                data_rows.append({
                    "Filename": file.name,
                    "Cell_Count": np.random.randint(50, 200),
                    "Avg_Intensity": np.random.uniform(0.2, 0.9),
                })
            
            st.session_state.generated_data = pd.DataFrame(data_rows)
            st.success("Done!")

    if st.session_state.generated_data is not None:
        st.subheader("Results")
        st.dataframe(st.session_state.generated_data)

# ==========================================
# 3. MARKETPLACE
# ==========================================
elif page == "Marketplace":
    st.title("🌍 Model Marketplace")
    st.markdown("Discover models trained by the community.")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        search_query = st.text_input("Search models...", placeholder="e.g. Brain, HeLa, Nuclei").lower()
    with col2:
        st.selectbox("Sort by", ["Popularity", "Newest", "Rating"])
    
    st.markdown("---")

    # Mock Data with Specific Cell Images
    all_market_models = [
        {
            "name": "H&E Histology Segmenter", 
            "tags": "#Tissue #H&E", 
            "stars": "⭐⭐⭐⭐⭐",
            # Pink/Purple Tissue Staining
            "img": "https://images.unsplash.com/photo-1576086213369-97a306d36557?auto=format&fit=crop&w=400&q=80"
        },
        {
            "name": "Fluorescent Nuclei (DAPI)", 
            "tags": "#Nuclei #Fluorescence", 
            "stars": "⭐⭐⭐⭐",
            # Blue glowing dots/cells
            "img": "data:image/jpeg;base64,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 Ihrer"
        },
        {
            "name": "Stem Cell Colony Tracker", 
            "tags": "#LiveCell #PhaseContrast", 
            "stars": "⭐⭐⭐⭐⭐",
            # Bubbly cellular structures
            "img": "https://www.whatisbiotechnology.org/assets/images/science/pages/stem.jpg"
        },
        {
            "name": "Bacteria/Microbe Counter", 
            "tags": "#Microbiology #100x", 
            "stars": "⭐⭐⭐",
            # Small particles/cells
            "img": "https://images.unsplash.com/photo-1581093450021-4a7360e9a6b5?auto=format&fit=crop&w=400&q=80"
        },
    ]

    filtered_models = [
        m for m in all_market_models 
        if search_query in m['name'].lower() or search_query in m['tags'].lower()
    ]

    if not filtered_models:
        st.info("No models found matching your search.")
    else:
        cols = st.columns(4)
        for i, model in enumerate(filtered_models):
            with cols[i % 4]:
                st.image(model['img'], use_container_width=True) 
                st.subheader(model['name'])
                st.caption(model['tags'])
                st.write(model['stars'])
                
                if st.button(f"Add to Library", key=f"add_{i}"):
                    st.session_state.models.append({
                        "name": model['name'], 
                        "author": "Community", 
                        "type": "Imported", 
                        "accuracy": "Unknown"
                    })
                    st.toast(f"Added {model['name']}!")

# ==========================================
# 4. ANALYSIS LAB
# ==========================================
elif page == "Analysis Lab":
    st.title("🔬 Analysis Lab")
    
    if st.session_state.generated_data is not None:
        df = st.session_state.generated_data
        
        c1, c2 = st.columns(2)
        with c1:
            x_axis = st.selectbox("X Axis", df.columns)
        with c2:
            y_axis = st.selectbox("Y Axis", df.columns)
        
        fig, ax = plt.subplots()
        ax.scatter(df[x_axis], df[y_axis])
        ax.set_title(f"{y_axis} vs {x_axis}")
        st.pyplot(fig)
    else:
        st.info("No data available. Run an analysis first.")
