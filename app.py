# ==========================================
# MEDGEMMA IMPACT CHALLENGE
# Adaptive GPU Loading — Runs on 2GB to 80GB
# ==========================================

import os
import sys
import subprocess

# --- STEP 1: INSTALL DEPENDENCIES ---
print("Installing dependencies... (This may take 1-2 minutes)")
pkgs = [
    "torch",
    "transformers",
    "accelerate",
    "bitsandbytes",
    "gradio",
    "huggingface_hub",
    "pillow"
]
for pkg in pkgs:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "-U", pkg])

print("Dependencies installed!\n")

# --- STEP 2: IMPORTS ---
import torch
from huggingface_hub import login
from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig
import gradio as gr
from PIL import Image

# --- STEP 3: AUTHENTICATION ---
print("--- AUTHENTICATION REQUIRED ---")
print("Please paste your Hugging Face Access Token when prompted.")
login()

# --- STEP 4: ADAPTIVE LOADING CONFIG ---
def get_load_config():
    """
    Automatically selects the best precision based on available GPU VRAM.
    - 14GB+  → Full float16     (~8GB used)  — best quality
    - 8-14GB → 8-bit            (~4.5GB used) — good quality
    - <8GB   → 4-bit NF4        (~3.6GB used) — compressed but still strong
    - No GPU → CPU fallback      (slow but works)
    """
    if not torch.cuda.is_available():
        print("⚠️  No GPU detected — falling back to CPU (inference will be slow)")
        return {"dtype": torch.float32, "device_map": "cpu"}, None

    total_vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
    gpu_name = torch.cuda.get_device_name(0)
    print(f"Detected GPU: {gpu_name} ({total_vram:.1f} GB VRAM)")

    if total_vram >= 14:
        print("→ Loading in full Float16 (best quality)")
        return {"dtype": torch.float16, "device_map": "auto"}, None

    elif total_vram >= 8:
        print("→ Loading in 8-bit (good quality, reduced memory)")
        bnb_config = BitsAndBytesConfig(load_in_8bit=True)
        return {"device_map": "auto"}, bnb_config

    else:
        print("→ Loading in 4-bit NF4 (compressed, fits low VRAM)")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16
        )
        return {"device_map": "auto"}, bnb_config

# --- STEP 5: LOAD MODEL ---
MODEL_ID = "google/medgemma-4b-it"
print(f"\nLoading {MODEL_ID}...")

load_kwargs, bnb_config = get_load_config()
if bnb_config:
    load_kwargs["quantization_config"] = bnb_config

try:
    processor = AutoProcessor.from_pretrained(MODEL_ID, use_fast=False)

    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID,
        **load_kwargs
    )

    torch.cuda.empty_cache()

    if torch.cuda.is_available():
        used = torch.cuda.memory_allocated(0) / 1024**3
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"\n✅ Model loaded! GPU Memory: {used:.2f} GB / {total:.1f} GB used")
    else:
        print("\n✅ Model loaded on CPU!")

except Exception as e:
    print(f"\n❌ ERROR loading model: {e}")
    print("Try: Runtime > Restart Session, then run again.")
    sys.exit()

# --- STEP 6: INFERENCE FUNCTION ---
def analyze_medical_case(image, question):
    if image is None:
        return "⚠️ Please upload a medical image (X-ray, CT Scan, Skin Lesion, etc) to analyze."

    if not question.strip():
        question = "Describe this medical image in detail."

    print(f"\nProcessing: {question}")

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text",  "text": question}
            ]
        }
    ]

    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt"
    ).to(model.device)

    input_len = inputs["input_ids"].shape[-1]

    with torch.inference_mode():
        generation = model.generate(
            **inputs,
            max_new_tokens=250,
            do_sample=False
        )
        generation = generation[0][input_len:]
        decoded = processor.decode(generation, skip_special_tokens=True)

    return decoded

# --- STEP 7: GRADIO UI ---
custom_css = """
#component-0 {max-width: 800px; margin: auto;}
.gradio-container {background-color: #f0f2f6}
"""

with gr.Blocks(title="MedGemma Diagnostic Assistant") as demo:
    gr.Markdown(
        """
        # 🏥 MedGemma Impact Challenge: Diagnostic Assistant
        **Powered by Google MedGemma-4B-IT — Adaptive Precision Loading**

        *Upload a medical image (X-ray, CT, Skin Lesion) and ask a clinical question.*
        """
    )

    with gr.Row():
        with gr.Column():
            img_input = gr.Image(type="pil", label="Upload Medical Image")
            txt_input = gr.Textbox(
                label="Clinical Question",
                placeholder="e.g., 'What abnormalities are visible in this X-ray?'",
                lines=2
            )
            submit_btn = gr.Button("Analyze Case", variant="primary")

        with gr.Column():
            output_text = gr.Textbox(
                label="MedGemma Analysis",
                lines=10,
                interactive=False
            )

    submit_btn.click(
        fn=analyze_medical_case,
        inputs=[img_input, txt_input],
        outputs=output_text
    )

    gr.Markdown("---\n*⚠️ Disclaimer: AI outputs must be verified by a qualified medical professional. This is a hackathon demonstration only.*")

# --- STEP 8: LAUNCH ---
print("\nLaunching application...")
demo.launch(share=True, debug=True, css=custom_css)
