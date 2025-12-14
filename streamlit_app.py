import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image, ImageDraw
from streamlit_drawable_canvas import st_canvas
import matplotlib.pyplot as plt
import time

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

        # Canvas
        canvas_result = st_canvas(
            fill_color="rgba(255, 165, 0, 0.3)",  
            stroke_width=brush_size,
            stroke_color=stroke_color,
            background_image=bg_image,
            update_streamlit=True,
            height=400,
            width=600,
            drawing_mode="freedraw",
            key="canvas",
        )

    with col3:
        st.subheader("Validation")
        # Fixed width parameter
        st.image(bg_image, caption="Live Prediction", width="stretch")
        
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
# 3. MARKETPLACE (Updated)
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
            "img": "data:image/jpeg;base64,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"
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
                # FIX: Replaced use_column_width=True with use_container_width=True
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
