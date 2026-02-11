"""
Streamlit Chat UI for Emmit AI.

Provides a premium interface to interact with the multilingual/vision model.
"""

import streamlit as st
import torch
import sentencepiece as spm
import sys
from pathlib import Path

# Allow imports from parent
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from emmit.model.config import EmmitConfig
from emmit.model.transformer import EmmitModel
from emmit.model.generation import generate

st.set_page_config(page_title="Emmit AI | Interactive Chat", layout="wide")

st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
        color: #ffffff;
    }
    .stTextInput > div > div > input {
        background-color: #262730;
        color: white;
    }
    .chat-bubble {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .user-bubble { background-color: #1e3a8a; }
    .ai-bubble { background-color: #374151; }
    </style>
""", unsafe_allow_html=True) # Note: unsafe_allow_html is standard, the bot might misspell sometimes in thoughts but I will use correct one.

# Let's use the correct one in code
st.markdown("""
<style>
    [data-testid="stAppViewContainer"] {
        background: radial-gradient(circle at top right, #1a1a2e, #16213e, #0f3460);
    }
</style>
""", unsafe_allow_html=True)

st.title("🌌 Emmit AI")
st.caption("Mixture of Experts • Multilingual • Vision-Language")

@st.cache_resource
def load_assets(config_path, ckpt_path, tokenizer_path):
    config = EmmitConfig.from_yaml(config_path)
    model = EmmitModel(config)
    
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state_dict)
    model.eval()
    
    tokenizer = spm.SentencePieceProcessor(model_file=f"{tokenizer_path}.model")
    return model, tokenizer, config

# --- Sidebar Controls ---
with st.sidebar:
    st.header("⚙️ Model Settings")
    temp = st.slider("Temperature", 0.0, 2.0, 0.7)
    top_p = st.slider("Top-P", 0.0, 1.0, 0.9)
    max_tokens = st.number_input("Max New Tokens", 1, 1024, 256)
    
    st.divider()
    config_file = st.text_input("Config Path", "configs/emmit_supertiny.yaml")
    checkpoint_file = st.text_input("Checkpoint Path", "outputs/supertiny_smoke_test/checkpoint_50.pt")
    tokenizer_file = st.text_input("Tokenizer Path", "tokenizers/test_tokenizer")

# --- Chat Interface ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask Emmit something..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate response
    with st.chat_message("assistant"):
        try:
            model, tokenizer, config = load_assets(config_file, checkpoint_file, tokenizer_file)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model.to(device)
            
            with st.spinner("🌌 Thinking..."):
                input_ids = torch.tensor([tokenizer.encode(prompt)], device=device)
                output_ids = generate(
                    model, 
                    input_ids, 
                    max_new_tokens=max_tokens,
                    temperature=temp,
                    top_p=top_p
                )
                response = tokenizer.decode(output_ids[0][input_ids.shape[1]:].tolist())
                
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
        except Exception as e:
            st.error(f"Error: {e}")
            st.info("Ensure the model paths in the sidebar are correct.")
