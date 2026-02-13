# 🌌 Emmit AI

**The Mixture-of-Experts Frontier for Multilingual Vision-Language Generation.**

Emmit AI is a state-of-the-art transformer architecture designed from the ground up for scalability, efficiency, and cross-lingual intelligence. It leverages a sparse Mixture-of-Experts (MoE) backbone combined with integrated vision capabilities and an advanced multilingual tokenization pipeline.

---

## 🚀 Features

- **Sparse MoE Backbone**: 13.2B total parameters with 1.3B active parameters per token, optimized using Top-2 soft routing and auxiliary load-balancing losses.
- **Vision-Language Integration**: Patch-based vision encoder (ViT) seamlessly interleaved with textual tokens for multimodal reasoning.
- **Multilingual Excellence**: Native support for 50+ languages, including specialized normalization for Indic scripts (Devanagari, Dravidian, etc.).
- **Premium Interactive UI**: A high-end glassmorphic Chat UI powered by Streamlit, featuring real-time AI intelligence and architectural tracing.
- **Efficiency First**: Implemented with Grouped-Query Attention (GQA), Rotary Positional Embeddings (RoPE), and Flash Attention 2 support.

---

## 🛠️ Tech Stack

- **Core**: PyTorch, SentencePiece, NumPy
- **Distributed Training**: PyTorch FSDP (Fully Sharded Data Parallel)
- **UI/UX**: Streamlit, Custom CSS (Glassmorphism)
- **Infrastructure**: YAML-based configuration, Hugging Face Inference API

---

## 📦 Installation

```bash
# Clone the repository
git clone https://github.com/harshittrikaai-gif/EmMi.git
cd EmMi

# Install dependencies
pip install .

# Or install from the built wheel (v0.2.0)
pip install dist/emmit-0.2.0-py3-none-any.whl
```

---

## 🚦 Quick Start

### 1. Launch the Premium Experience
Run the interactive chat interface with cloud intelligence and architectural mode:
```bash
python -m streamlit run scripts/app.py
```

### 2. Verify Infrastructure (Smoke Test)
Train a super-tiny model locally to verify the MoE and data pipelines:
```bash
python scripts/train.py --config configs/emmit_tiny.yaml --generate_sample_data
```

### 3. Run Unit Tests
Validate all core modules:
```bash
![alt text](image.png)

---

## 📂 Repository Structure

- `emmit/model/` - Core transformer (MoE, GQA, RoPE, RMSNorm)
- `emmit/vision/` - ViT Encoder and multimodal preparation
- `emmit/tokenizer/` - Multilingual normalization and stratified sampling
- `emmit/training/` - FSDP Trainer and checkpointing logic
- `configs/` - Model definitions (Tiny to 13B-MoE)
- `scripts/` - Entry points for training, generation, and UI
- `dist/` - Built distribution artifacts (v0.2.0)

---

## 🌌 Project Status: v0.2.0
Emmit AI is currently in an experimental phase. The architecture is fully verified via unit tests and tiny-scale training. Large-scale weights are under development.

Built with ✨ by the Emmit AI Team.
