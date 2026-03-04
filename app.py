import streamlit as st
import rasterio
import numpy as np
import joblib
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import tempfile
import os

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Satellite Image Classifier", 
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1rem;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .main-header h1 {
        color: white;
        font-size: 2.5rem;
        margin-bottom: 0.5rem;
    }
    .main-header p {
        color: #f0f0f0;
        font-size: 1.1rem;
    }
    .section-header {
        font-size: 1.3rem;
        font-weight: 600;
        margin: 1.5rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #667eea;
    }
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
    }
    .stButton button {
        width: 100%;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: 600;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 5px;
        transition: all 0.3s ease;
    }
    .stButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    .upload-section {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 1.5rem;
        border: 1px solid #e0e0e0;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Header Section
# -----------------------------
st.markdown("""
<div class="main-header">
    <h1>🛰️ Satellite Image Classification</h1>
    <p>Decision Tree Classifier for Land Cover Mapping</p>
</div>
""", unsafe_allow_html=True)

# -----------------------------
# Upload Section in Main Page
# -----------------------------
st.markdown('<div class="section-header">📥 Upload Section</div>', unsafe_allow_html=True)

upload_col1, upload_col2 = st.columns(2)

with upload_col1:
    st.markdown('<div class="upload-section">', unsafe_allow_html=True)
    st.markdown("### 📁 **Load Trained Model**")
    uploaded_model = st.file_uploader(
        "Upload your trained Decision Tree model (.pkl file)", 
        type=["pkl"],
        help="Select the model file you trained earlier"
    )
    
    model = None
    if uploaded_model is not None:
        model = joblib.load(uploaded_model)
        st.success("✅ Model loaded successfully!")
    st.markdown('</div>', unsafe_allow_html=True)

with upload_col2:
    st.markdown('<div class="upload-section">', unsafe_allow_html=True)
    st.markdown("### 🖼️ **Load Satellite Image**")
    uploaded_file = st.file_uploader(
        "Upload a satellite image in GeoTIFF format", 
        type=["tif", "tiff"],
        help="Select the image you want to classify"
    )
    st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------
# Sidebar (only model info and settings)
# -----------------------------
with st.sidebar:
    st.markdown("## 📊 **Model Information**")
    
    with st.expander("📘 **About the Model**", expanded=True):
        st.markdown("""
        This is a **Decision Tree** classifier trained to identify three land cover classes:
        
        - 🏙️ **Urban Areas** (Red)
        - 🌾 **Agricultural Areas** (Green)
        - 💧 **Water Bodies** (Blue)
        
        The model analyzes **three spectral bands** (e.g., Red, Green, Blue) to classify each pixel.
        """)
    
    st.markdown("---")
    
    st.markdown("## 📝 **Instructions**")
    st.markdown("""
    1. Upload your trained model (.pkl)
    2. Upload a GeoTIFF image
    3. Select the bands for RGB
    4. Click 'Run Classification'
    5. Download the classified image
    """)

# -----------------------------
# Main Content Area (only shows if image is uploaded)
# -----------------------------
if uploaded_file is not None:
    with rasterio.open(uploaded_file) as src:
        image = src.read()
        crs = src.crs
        transform = src.transform
        
        # Display image info in a nice card
        st.markdown('<div class="section-header">📊 Image Properties</div>', unsafe_allow_html=True)
        info_col1, info_col2, info_col3, info_col4 = st.columns(4)
        with info_col1:
            st.markdown('<div class="metric-card">📏 **Width**<br>{} px</div>'.format(image.shape[2]), unsafe_allow_html=True)
        with info_col2:
            st.markdown('<div class="metric-card">📏 **Height**<br>{} px</div>'.format(image.shape[1]), unsafe_allow_html=True)
        with info_col3:
            st.markdown('<div class="metric-card">🎨 **Bands**<br>{}</div>'.format(image.shape[0]), unsafe_allow_html=True)
        with info_col4:
            crs_str = str(crs).split(':')[-1] if crs else 'Unknown'
            st.markdown('<div class="metric-card">📍 **CRS**<br>{}</div>'.format(crs_str), unsafe_allow_html=True)

    # -----------------------------
    # Original Image Display
    # -----------------------------
    if image.shape[0] >= 3:
        st.markdown('<div class="section-header">🖼️ Original Image (RGB)</div>', unsafe_allow_html=True)
        
        # Display original image
        rgb = np.stack([image[3], image[2], image[1]], axis=2)
        rgb_norm = (rgb - rgb.min()) / (rgb.max() - rgb.min())
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.imshow(rgb_norm)
        ax.set_title("Original RGB Composite", fontsize=14, pad=10)
        ax.axis('off')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    # -----------------------------
    # Band Selection
    # -----------------------------
    st.markdown('<div class="section-header">🎚️ Band Selection</div>', unsafe_allow_html=True)
    
    bands_options = list(range(1, image.shape[0] + 1))
    
    band_col1, band_col2, band_col3 = st.columns(3)
    
    with band_col1:
        band_r = st.selectbox(
            "🔴 **Red Band**", 
            bands_options, 
            index=3 if len(bands_options) >= 4 else 0,
            help="Select the band corresponding to Red wavelength"
        )
    
    with band_col2:
        band_g = st.selectbox(
            "🟢 **Green Band**", 
            bands_options, 
            index=2 if len(bands_options) >= 3 else 0,
            help="Select the band corresponding to Green wavelength"
        )
    
    with band_col3:
        band_b = st.selectbox(
            "🔵 **Blue Band**", 
            bands_options, 
            index=1 if len(bands_options) >= 2 else 0,
            help="Select the band corresponding to Blue wavelength"
        )
    
    st.markdown("---")
    
    # -----------------------------
    # Classification Button
    # -----------------------------
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        classify_button = st.button("🚀 **RUN CLASSIFICATION**", use_container_width=True)
    
    if classify_button:
        if model is None:
            st.error("❌ Please load a model first in the Upload Section above!")
        else:
            with st.spinner("🔄 Classifying image... Please wait..."):
                # Prepare data
                height, width = image.shape[1], image.shape[2]
                X = np.stack([
                    image[band_r - 1],
                    image[band_g - 1],
                    image[band_b - 1]
                ], axis=2)
                X_reshaped = X.reshape(-1, 3)
                
                # Predict
                y_pred = model.predict(X_reshaped)
                y_pred_image = y_pred.reshape(height, width)
                
                # Visualization
                colors = ['red', 'green', 'blue']
                cmap_custom = ListedColormap(colors)
                y_pred_image_0based = y_pred_image - 1
                
                # Display results - JUST THE IMAGE ALONE
                st.markdown('<div class="section-header">📊 Classification Results</div>', unsafe_allow_html=True)
                
                # عرض الصورة فقط بدون أعمدة
                st.markdown("### 🗺️ Classified Image")
                fig2, ax2 = plt.subplots(figsize=(10, 8))
                im = ax2.imshow(y_pred_image_0based, cmap=cmap_custom, vmin=0, vmax=2)
                ax2.axis('off')
                
                # Legend
                legend_labels = ['🏙️ Urban', '🌾 Agricultural', '💧 Water']
                legend_colors = colors
                patches = [plt.Line2D([0], [0], marker='o', color='w', 
                                     markerfacecolor=legend_colors[i], 
                                     markersize=10, 
                                     label=legend_labels[i]) for i in range(3)]
                ax2.legend(handles=patches, loc='lower right', fontsize=9, framealpha=0.9)
                plt.tight_layout()
                st.pyplot(fig2)
                plt.close()
                
                # -----------------------------
                # Save and Download
                # -----------------------------
                st.markdown('<div class="section-header">💾 Download Results</div>', unsafe_allow_html=True)
                
                with tempfile.NamedTemporaryFile(delete=False, suffix='.tif') as tmp:
                    output_path = tmp.name
                
                with rasterio.open(
                    output_path,
                    'w',
                    driver='GTiff',
                    height=height,
                    width=width,
                    count=1,
                    dtype=y_pred_image.dtype,
                    crs=crs,
                    transform=transform
                ) as dst:
                    dst.write(y_pred_image, 1)
                
                download_col1, download_col2, download_col3 = st.columns([1, 2, 1])
                with download_col2:
                    with open(output_path, 'rb') as f:
                        st.download_button(
                            label="📥 **Download Classified Image (GeoTIFF)**",
                            data=f,
                            file_name="classified_image.tif",
                            mime="image/tiff",
                            use_container_width=True
                        )
                
                st.success("✅ Classification completed successfully!")

