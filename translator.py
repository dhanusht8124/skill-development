# Install dependencies
!pip install -q gradio transformers sentencepiece

# Import libraries
from transformers import MarianMTModel, MarianTokenizer
import gradio as gr

# Define supported translation directions
language_pairs = {
    "English to French": ("en", "fr"),
    "French to English": ("fr", "en"),
    "English to German": ("en", "de"),
    "German to English": ("de", "en"),
    "English to Spanish": ("en", "es"),
    "Spanish to English": ("es", "en"),
    "English to Russian": ("en", "ru"),
    "Russian to English": ("ru", "en"),
    "English to Japanese": ("en", "ja"),
    "Japanese to English": ("ja", "en"),
}

# Function to load model/tokenizer
def load_model(src_lang, tgt_lang):
    model_name = f"Helsinki-NLP/opus-mt-{src_lang}-{tgt_lang}"
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)
    return tokenizer, model

# Translation function
def translate(text, direction):
    if direction not in language_pairs:
        return "Translation direction not supported."
    src, tgt = language_pairs[direction]
    try:
        tokenizer, model = load_model(src, tgt)
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        outputs = model.generate(**inputs)
        return tokenizer.decode(outputs[0], skip_special_tokens=True)
    except Exception as e:
        return f"Error during translation: {e}"

# Gradio interface
iface = gr.Interface(
    fn=translate,
    inputs=[
        gr.Textbox(label="Enter Text"),
        gr.Dropdown(choices=list(language_pairs.keys()), label="Translation Direction"),
    ],
    outputs=gr.Textbox(label="Translated Text"),
    title="🌐 Multi-Language AI Translator",
    description="Translate between English, French, German, Spanish, Russian, and Japanese using pre-trained AI models.",
)

iface.launch()
