import streamlit as st
from PIL import Image
import tempfile
import os
from pathlib import Path
import sys
import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from training.config import get_config
from training.inference import Inference

st.set_page_config(page_title="EleGANt Makeup Transfer", layout="wide")

@st.cache_resource
def load_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = get_config()
    model = Inference(config, type('Args', (), {'device': device})(),
                     'sow_pyramid_a5_e3d2_remapped.pth')
    return model, device

@st.cache_resource
def load_styles():
    styles_dir = Path('assets/images/makeup')
    styles = sorted([f for f in styles_dir.glob('make_styles_*.jpg')])
    return {f'style_{i+1}': str(f) for i, f in enumerate(styles)}

st.title("🎨 EleGANt Makeup Transfer")

model, device = load_model()
styles = load_styles()

col1, col2 = st.columns(2)

with col1:
    st.subheader("Source Image")
    source_file = st.file_uploader("Upload image with face", type=['jpg', 'png', 'jpeg'])

with col2:
    st.subheader("Makeup Style")
    style_option = st.selectbox("Choose style", list(styles.keys()))

if source_file and style_option:
    col1, col2, col3 = st.columns(3)

    with col1:
        source_img = Image.open(source_file).convert('RGB')
        st.image(source_img, caption="Source", use_column_width=True)

    with col2:
        style_img = Image.open(styles[style_option]).convert('RGB')
        st.image(style_img, caption="Style", use_column_width=True)

    if st.button("🚀 Apply Makeup"):
        with st.spinner("Processing..."):
            result = model.transfer(source_img, style_img, postprocess=True)

            if result is not None:
                with col3:
                    st.image(result, caption="Result", use_column_width=True)

                buf = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
                result.save(buf.name, quality=95)

                with open(buf.name, 'rb') as f:
                    st.download_button("Download Result", f.read(),
                                     file_name="result.jpg", mime="image/jpeg")

                os.unlink(buf.name)
            else:
                st.error("Transfer failed")

st.sidebar.markdown("### About\nEleGANt: High-quality makeup transfer using deep learning")
st.sidebar.markdown(f"🖥️ Device: {device}")
st.sidebar.markdown(f"📦 Styles: {len(styles)}")
