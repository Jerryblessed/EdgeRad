# 🏥 EdgeRad: The Offline AI Radiologist

> *Bringing specialist-grade diagnostics to rural clinics — no internet required.*

[![MedGemma](https://img.shields.io/badge/Powered%20by-MedGemma--4B--IT-blue?logo=google)](https://huggingface.co/google/medgemma-4b-it)
[![HAI-DEF](https://img.shields.io/badge/Google-HAI--DEF-green)](https://health.google/health-research/applied-health-ai/)
[![License](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey)](https://creativecommons.org/licenses/by/4.0/)
[![Kaggle](https://img.shields.io/badge/Kaggle-MedGemma%20Impact%20Challenge-orange?logo=kaggle)](https://kaggle.com/competitions/med-gemma-impact-challenge)

---

## 🌍 The Problem

In rural healthcare facilities across West Africa, two critical resources are consistently scarce: **specialist radiologists** and **reliable internet connectivity**.

- Patients wait **weeks** for X-ray interpretations that must be physically transported to city hospitals
- Cloud-based AI solutions **vanish** the moment the network drops
- Sending sensitive patient images to the cloud raises **data privacy concerns** in regions with developing data protection infrastructure

> *We don't just need "AI in Healthcare." We need AI at the Edge.*

---

## 💡 The Solution

**EdgeRad** is a lightweight, privacy-first diagnostic assistant powered by Google's HAI-DEF **MedGemma-4B-IT** model. It runs **100% offline** on consumer hardware, allowing nurses and general practitioners to upload medical images (chest X-rays, skin lesions, CT scans) and receive immediate clinical-grade analysis — with or without an internet connection.

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        EDGERAD SYSTEM                           │
│                                                                 │
│  ┌──────────────┐     ┌──────────────────────────────────────┐  │
│  │   FRONTEND   │     │            BACKEND CORE              │  │
│  │              │     │                                      │  │
│  │   Gradio UI  │────▶│  ┌─────────────────────────────┐    │  │
│  │              │     │  │   Adaptive Loading Engine   │    │  │
│  │  • Image     │     │  │                             │    │  │
│  │    Upload    │     │  │  VRAM ≥ 14GB → Float16      │    │  │
│  │  • Clinical  │     │  │  VRAM 8-14GB → 8-bit        │    │  │
│  │    Question  │     │  │  VRAM < 8GB  → 4-bit NF4   │    │  │
│  │  • Analysis  │     │  │  No GPU      → CPU Fallback │    │  │
│  │    Output    │     │  └──────────────┬──────────────┘    │  │
│  │              │     │                 │                    │  │
│  └──────────────┘     │  ┌──────────────▼──────────────┐    │  │
│                        │  │     MedGemma-4B-IT Model    │    │  │
│                        │  │   (google/medgemma-4b-it)   │    │  │
│                        │  │                             │    │  │
│                        │  │  • Vision Encoder           │    │  │
│                        │  │  • Language Decoder         │    │  │
│                        │  │  • Multimodal Fusion        │    │  │
│                        │  └──────────────┬──────────────┘    │  │
│                        │                 │                    │  │
│                        │  ┌──────────────▼──────────────┐    │  │
│                        │  │     Inference Pipeline      │    │  │
│                        │  │                             │    │  │
│                        │  │  AutoProcessor (chat tmpl)  │    │  │
│                        │  │  torch.inference_mode()     │    │  │
│                        │  │  Deterministic generation   │    │  │
│                        │  └─────────────────────────────┘    │  │
│                        └──────────────────────────────────────┘  │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                  DEPLOYMENT TARGETS                      │   │
│  │                                                          │   │
│  │   ☁️  Google Colab T4      🖥️  RTX 3060 Laptop           │   │
│  │   🤖  NVIDIA Jetson Orin  💻  Any CUDA-capable device    │   │
│  │                   🔌  Fully Offline                      │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

### Data Flow

```
  Medical Image (X-ray / CT / Skin Lesion)
          │
          ▼
  ┌───────────────┐
  │  PIL Image    │  ← Gradio handles upload & conversion
  └───────┬───────┘
          │
          ▼
  ┌───────────────────────────┐
  │  apply_chat_template()    │  ← Formats image + question into
  │  [image] + [text prompt]  │    MedGemma's expected input format
  └───────────┬───────────────┘
              │
              ▼
  ┌───────────────────────────┐
  │   MedGemma-4B-IT          │  ← Multimodal inference
  │   Vision + Language       │    (image & text processed together)
  └───────────┬───────────────┘
              │
              ▼
  ┌───────────────────────────┐
  │  Token Decoding           │  ← Strips prompt tokens, decodes output
  │  skip_special_tokens=True │
  └───────────┬───────────────┘
              │
              ▼
      Clinical Analysis Text
```

---

## ⚡ Adaptive GPU Loading

EdgeRad automatically detects available hardware and loads the model at the optimal precision — no configuration needed.

| Hardware | VRAM | Mode | Memory Used | Quality |
|---|---|---|---|---|
| A100 / H100 / RTX 4090 | 40–80 GB | Float16 | ~8 GB | ⭐⭐⭐ Best |
| T4 / RTX 3060–3090 | 14–24 GB | Float16 | ~8 GB | ⭐⭐⭐ Best |
| Mid-range GPU | 8–14 GB | 8-bit | ~4.5 GB | ⭐⭐ Good |
| Entry-level GPU | < 8 GB | 4-bit NF4 | ~3.6 GB | ⭐⭐ Solid |
| No GPU | CPU RAM | Float32 | ~16 GB RAM | ⭐ Slow |

---

## 🚀 Quick Start

### Option 1 — Google Colab (Recommended)

1. Open the notebook in Colab
2. Run all cells
3. Authenticate with your Hugging Face token when prompted
4. Open the Gradio public URL and start analyzing

### Option 2 — Local Setup

```bash
# Clone the repo
git clone https://github.com/Jerryblessed/EdgeRad.git
cd EdgeRad

# Install dependencies
pip install torch transformers accelerate bitsandbytes gradio huggingface_hub pillow

# Run
python app.py
```

### Requirements

- Python 3.9+
- CUDA-compatible GPU recommended (CPU fallback available)
- Hugging Face account with MedGemma access approved
  - Request access at: https://huggingface.co/google/medgemma-4b-it

---

## 🔐 Authentication

EdgeRad uses your Hugging Face token to download the MedGemma model on first run. The model is then cached locally — subsequent runs work **fully offline**.

```python
# You will be prompted to enter your token on first run
# Generate a token at: https://huggingface.co/settings/tokens
# Select "Read" permission only
```

> ⚠️ **Never commit your token to GitHub.** Use environment variables or Colab Secrets.

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Model | `google/medgemma-4b-it` (HAI-DEF) |
| Inference | PyTorch + Hugging Face Transformers |
| Quantization | bitsandbytes (4-bit NF4 / 8-bit) |
| Acceleration | Hugging Face Accelerate |
| Frontend | Gradio |
| Deployment | Docker-ready, fully offline |

---

## 📊 Performance

| Metric | Value |
|---|---|
| Model Parameters | 4 Billion |
| VRAM (4-bit mode) | ~3.6 GB |
| VRAM (float16 mode) | ~8 GB |
| Avg. Inference Time | 5–15 seconds (T4 GPU) |
| Internet Required | ❌ After initial download |

---

## 🌐 Impact

EdgeRad is specifically designed for deployment in **low-resource clinical settings**:

- **Rural clinics in West Africa** with intermittent connectivity
- **Field hospitals** and mobile medical units
- **Community health centers** without radiology departments
- Any setting where **patient data privacy** is a priority

A single deployment requires one internet connection to download the model. After that, the ethernet cable can be unplugged permanently.

---

## ⚠️ Disclaimer

EdgeRad is a research and demonstration tool developed for the MedGemma Impact Challenge. All AI-generated outputs **must be reviewed and verified by a qualified medical professional** before being used in any clinical decision-making context.

---

## 📄 License

This project is licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) in accordance with the MedGemma Impact Challenge rules.

---

## 🏆 Competition

Submitted to the **MedGemma Impact Challenge** on Kaggle by Google Research.
- Main Track
- The Edge AI Prize Track

---

*Built with ❤️ for the clinics that need it most.*
