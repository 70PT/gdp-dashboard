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

        # FIX: Pass PIL Image directly - streamlit-drawable-canvas handles conversion
        canvas_result = st_canvas(
            fill_color="rgba(255, 165, 0, 0.3)",  
            stroke_width=brush_size,
            stroke_color=stroke_color,
            background_image=bg_image,  # Pass PIL Image directly
            update_streamlit=True,
            height=400,
            width=600,
            drawing_mode="freedraw",
            key="canvas",
        )

    with col3:
        st.subheader("Validation")
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
            "img": "https://images.unsplash.com/photo-1576086213369-97a306d36557?auto=format&fit=crop&w=400&q=80"
        },
        {
            "name": "Fluorescent Nuclei (DAPI)", 
            "tags": "#Nuclei #Fluorescence", 
            "stars": "⭐⭐⭐⭐",
            "img": "https://images.unsplash.com/photo-1532187863486-abf9dbad1b69?auto=format&fit=crop&w=400&q=80"
        },
        {
            "name": "Stem Cell Colony Tracker", 
            "tags": "#LiveCell #PhaseContrast", 
            "stars": "⭐⭐⭐⭐⭐",
            "img": "https://images.unsplash.com/photo-1579154204601-01588f351e67?auto=format&fit=crop&w=400&q=80"
        },
        {
            "name": "Bacteria/Microbe Counter", 
            "tags": "#Microbiology #100x", 
            "stars": "⭐⭐⭐",
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
