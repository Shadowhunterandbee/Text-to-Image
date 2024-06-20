import streamlit as st
import torch
from transformers import CLIPTokenizer
from PIL import Image
import model_loader  # Ensure this file is in the same directory
import pipeline  # Ensure this file is in the same directory

# Device configuration
DEVICE = "cpu"
ALLOW_CUDA = False
ALLOW_MPS = False

if torch.cuda.is_available() and ALLOW_CUDA:
    DEVICE = "cuda"
elif (torch.has_mps or torch.backends.mps.is_available()) and ALLOW_MPS:
    DEVICE = "mps"

# Streamlit UI
st.title("Text-to-Image Generation")
prompt = st.text_input("Enter a detailed prompt:")
uncond_prompt = st.text_input("Enter a negative prompt (optional):", "")
do_cfg = st.checkbox("Use Classifier-Free Guidance", True)
cfg_scale = st.slider("CFG Scale", min_value=1, max_value=14, value=8)
sampler_name = st.selectbox("Sampler", ["ddpm"])
num_inference_steps = st.slider("Number of Inference Steps", min_value=1, max_value=90, value=50)
seed = st.number_input("Random Seed", value=40, step=1)

# Load models and tokenizer
@st.cache_resource
def load_models():
    model_file = r"C:/Users/Shambhavi Deo/PycharmProjects/text-to-image/data/v1-5.ckpt"
    return model_loader.load_models(model_file, DEVICE)

@st.cache_resource
def load_tokenizer():
    return CLIPTokenizer(
        vocab_file="C:/Users/Shambhavi Deo/PycharmProjects/text-to-image/data/tokenizer_vocab.json",
        merges_file="C:/Users/Shambhavi Deo/PycharmProjects/text-to-image/data/tokenizer_merges.txt"
    )

models = load_models()
tokenizer = load_tokenizer()

# Generate the image
if st.button("Generate Image"):
    with st.spinner("Generating image..."):
        try:
            output_image = pipeline.generate(
                prompt=prompt,
                uncond_prompt=uncond_prompt,
                input_image=None,
                strength=None,
                do_cfg=do_cfg,
                cfg_scale=cfg_scale,
                sampler_name=sampler_name,
                n_inference_steps=num_inference_steps,
                seed=seed,
                models=models,
                device=DEVICE,
                idle_device="cpu",
                tokenizer=tokenizer,
            )
            image = Image.fromarray(output_image)
            st.image(image, caption="Generated Image")
        except TypeError:
            output_image = pipeline.generate(
                prompt=prompt,
                uncond_prompt=uncond_prompt,
                do_cfg=do_cfg,
                cfg_scale=cfg_scale,
                sampler_name=sampler_name,
                n_inference_steps=num_inference_steps,
                seed=seed,
                models=models,
                device=DEVICE,
                idle_device="cpu",
                tokenizer=tokenizer,
            )
            image = Image.fromarray(output_image)
            st.image(image, caption="Generated Image")

# Run Streamlit app
if __name__ == "__main__":
    st.write("Using device:", DEVICE)
