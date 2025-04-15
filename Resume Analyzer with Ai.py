# Install required libraries
!pip install -q gradio transformers pymupdf

# Import libraries
import fitz  # PyMuPDF
import gradio as gr
from transformers import pipeline

# Load AI pipelines
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
ner = pipeline("ner", grouped_entities=True)

# Function to extract text from PDF
def extract_text_from_pdf(pdf_file):
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text

# Function to analyze resume
def analyze_resume(pdf_file):
    try:
        # Extract text
        text = extract_text_from_pdf(pdf_file)
        if len(text) < 100:
            return "Resume text is too short or unreadable. Please upload a valid resume."

        # Summarize resume
        summary = summarizer(text[:1024], max_length=150, min_length=40, do_sample=False)[0]['summary_text']

        # Named entity recognition to get key skills / topics
        entities = ner(text[:1000])
        keywords = list(set([ent['word'] for ent in entities if ent['entity_group'] in ["ORG", "MISC", "SKILL", "JOB", "LOC"]]))
        
        feedback = "Looks like you have experience with: " + ", ".join(keywords[:10]) + "."

        return f"🔍 **Summary:**\n{summary}\n\n🛠️ **Key Skills/Entities:**\n{', '.join(keywords[:10])}\n\n💡 **Feedback:**\n{feedback}"

    except Exception as e:
        return f"Error: {str(e)}"

# Gradio Interface
iface = gr.Interface(
    fn=analyze_resume,
    inputs=gr.File(label="Upload Your Resume (PDF)"),
    outputs=gr.Markdown(label="AI Resume Analysis"),
    title="📄 AI Resume Analyzer",
    description="Upload your resume and get a summary, skill extraction, and feedback using AI.",
)

iface.launch()
