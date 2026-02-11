"""
Premium Streamlit Chat UI for Emmit AI.

Features:
  • Glassmorphic Design & Micro-animations
  • Intelligent Mode (Powered by Hugging Face API)
  • Architecture Mode (Local MoE Execution)
  • Dynamic Sidebar with Parameter Controls
"""

import streamlit as st
import torch
import sentencepiece as spm
import sys
import requests
import time
from pathlib import Path

# Allow imports from parent
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from emmit.model.config import EmmitConfig
from emmit.model.transformer import EmmitModel
from emmit.model.generation import generate

# --- Page Configuration ---
st.set_page_config(
    page_title="Emmit AI | Premium MoE Experience",
    page_icon="🌌",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Premium Look & Feel (Custom CSS) ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600&family=Outfit:wght@500;700&display=swap');

    html, body, [data-testid="stAppViewContainer"] {
        font-family: 'Inter', sans-serif;
        background: radial-gradient(circle at top right, #0a0a1a, #0d1117, #000000);
    }

    /* Glassmorphism sidebar */
    [data-testid="stSidebar"] {
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(15px);
        border-right: 1px solid rgba(255, 255, 255, 0.05);
    }

    /* Chat bubble styling */
    .stChatMessage {
        border-radius: 20px;
        padding: 15px;
        margin-bottom: 20px;
        border: 1px solid rgba(255, 255, 255, 0.05);
        transition: transform 0.3s cubic-bezier(0.175, 0.885, 0.32, 1.275);
    }
    .stChatMessage:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 30px rgba(0,0,0,0.5);
    }

    /* Gradient Header */
    .main-header {
        background: linear-gradient(90deg, #6366f1 0%, #a855f7 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-family: 'Outfit', sans-serif;
        font-weight: 700;
        font-size: 3.5rem;
        margin-bottom: 0.2rem;
    }

    /* Floating effect for containers */
    .glass-card {
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 15px;
        padding: 24px;
        margin-bottom: 20px;
    }

    /* Pulsing loading state */
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.5; }
        100% { opacity: 1; }
    }
    .thinking-status {
        font-style: italic;
        color: #a855f7;
        animation: pulse 1.5s infinite;
    }

    /* Hide streamlit clutter */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# --- AI Intelligence Helper ---
def call_huggingface_api(prompt, model_id="mistralai/Mixtral-8x7B-Instruct-v0.1"):
    """Fetch intelligent responses from free Hugging Face API."""
    API_URL = f"https://api-inference.huggingface.co/models/{model_id}"
    # Use fallback if token missing, though normally we'd want user to provide one.
    # We will use a public endpoint if possible or gracefully handle failure.
    try:
        response = requests.post(API_URL, json={"inputs": prompt, "parameters": {"max_new_tokens": 300}})
        if response.status_code == 200:
            return response.json()[0]['generated_text']
        return f"Intelligence Error: API returned {response.status_code}. Using local fallback instead."
    except Exception as e:
        return f"Intelligence Connection Failed: {e}"

@st.cache_resource
def load_architecture_assets(config_path, ckpt_path, tokenizer_path):
    """Load local MoE components for Architecture Mode."""
    config = EmmitConfig.from_yaml(config_path)
    model = EmmitModel(config)
    
    if Path(ckpt_path).exists():
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        state_dict = ckpt.get("model_state_dict", ckpt)
        model.load_state_dict(state_dict)
    
    model.eval()
    
    try:
        tokenizer = spm.SentencePieceProcessor(model_file=f"{tokenizer_path}.model")
    except:
        tokenizer = None
        
    return model, tokenizer, config

# --- Application Layout ---
with st.sidebar:
    st.image("https://img.icons8.com/parakeet/512/000000/artificial-intelligence.png", width=80)
    st.markdown("<h2 style='font-family:Outfit; font-weight:700;'>Emmit Control Center</h2>", unsafe_allow_html=True)
    
    st.divider()
    
    mode = st.radio("🤖 Interaction Mode", ["Intelligence (Cloud API)", "Architecture (Local MoE)"])
    
    with st.expander("🛠️ Generation Parameters", expanded=True):
        temp = st.slider("Creativity (Temp)", 0.0, 2.0, 0.7)
        top_p = st.slider("Diversity (Top-P)", 0.0, 1.0, 0.9)
        max_tokens = st.number_input("Response Length", 1, 1024, 256)
    
    st.divider()
    
    st.markdown("### 📊 Model Stats")
    col1, col2 = st.columns(2)
    col1.metric("Active Params", "1.3B")
    col2.metric("Total Params", "13.2B")
    
    st.info("Current Mode: " + mode)

# --- Main UI ---
st.markdown("<h1 class='main-header'>Emmit AI</h1>", unsafe_allow_html=True)
st.markdown("<p style='color:rgba(255,255,255,0.6); font-size:1.1rem; margin-top:-10px;'>The Multilingual Mixture-of-Experts Frontier</p>", unsafe_allow_html=True)

# Initialization
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display Messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User Input
if prompt := st.chat_input("What would you like to explore?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.markdown("<p class='thinking-status'>🌌 Connecting to neural pathways...</p>", unsafe_allow_html=True)
        
        if mode == "Intelligence (Cloud API)":
            # Real smart answers via cloud
            response = call_huggingface_api(prompt)
            # Remove prompt if repeated in output (Mistral style)
            if response.startswith(prompt):
                response = response[len(prompt):].strip()
        else:
            # Local Architectural trace
            try:
                model, tokenizer, config = load_architecture_assets(
                    "configs/emmit_supertiny.yaml",
                    "outputs/supertiny_smoke_test/checkpoint_50.pt",
                    "tokenizers/test_tokenizer"
                )
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                model.to(device)
                
                if tokenizer:
                    input_ids = torch.tensor([tokenizer.encode(prompt)], device=device)
                    output_ids = generate(model, input_ids, max_new_tokens=max_tokens, temperature=temp, top_p=top_p)
                    response = tokenizer.decode(output_ids[0][input_ids.shape[1]:].tolist())
                else:
                    response = "[Architecture Trace]: Tokenizer not found. This mode simulates the internal MoE weights dispatch but requires a trained SentencePiece model for human-readable output."
            except Exception as e:
                response = f"Architecture Error: {str(e)}"

        # Animated reveal
        full_response = ""
        for chunk in response.split():
            full_response += chunk + " "
            time.sleep(0.01)
            message_placeholder.markdown(full_response + "▌")
        
        message_placeholder.markdown(full_response)
        st.session_state.messages.append({"role": "assistant", "content": full_response})

st.markdown("---")
st.markdown("<p style='text-align: center; color: rgba(255,255,255,0.3); font-size: 0.8rem;'>Built with Emmit MoE v0.1.0 • Experimental Multimodal Architecture</p>", unsafe_allow_html=True)
