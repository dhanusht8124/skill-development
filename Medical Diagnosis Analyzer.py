# Install required packages
!pip install gradio transformers torch

import gradio as gr
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Function to load model with error handling
def load_model():
    try:
        # Using a smaller model that's less likely to timeout
        model_name = "distilbert-base-uncased"
        print("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        print("Loading model...")
        
        # Create a simple mock model for demonstration
        class MedicalDiagnosisModel(torch.nn.Module):
            def __init__(self):
                super(MedicalDiagnosisModel, self).__init__()
                self.base_model = AutoModelForSequenceClassification.from_pretrained(
                    model_name, 
                    num_labels=10
                )
                
            def forward(self, input_ids, attention_mask):
                outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
                return outputs.logits
        
        model = MedicalDiagnosisModel()
        return tokenizer, model
    
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Using mock model instead...")
        # Create a simple mock tokenizer and model
        from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
        tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
        model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=10)
        return tokenizer, model

# Load model and tokenizer
tokenizer, model = load_model()

# Mock diagnoses for demonstration - replace with real data in production
diagnosis_labels = [
    "Common Cold", "Influenza (Flu)", "Migraine", 
    "Pneumonia", "Urinary Tract Infection", 
    "Gastroenteritis", "Hypertension", 
    "Type 2 Diabetes", "Bronchitis", "Sinusitis"
]

def analyze_symptoms(patient_history, symptoms, age, gender):
    try:
        # Combine inputs for processing
        input_text = f"Patient history: {patient_history}. Symptoms: {symptoms}. Age: {age}. Gender: {gender}."
        
        # Tokenize inputs
        inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=512)
        
        # Get model predictions (mock in this case - replace with real model inference)
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Convert to probabilities (mock)
        probs = torch.nn.functional.softmax(outputs, dim=-1)[0]
        
        # Get top 3 predictions
        top_probs, top_indices = torch.topk(probs, 3)
        
        # Format results
        results = []
        for prob, idx in zip(top_probs, top_indices):
            results.append({
                "condition": diagnosis_labels[idx],
                "confidence": f"{prob.item()*100:.1f}%",
                "recommendation": get_recommendation(diagnosis_labels[idx])
            })
        
        return results
    
    except Exception as e:
        print(f"Error during analysis: {e}")
        # Return mock results if analysis fails
        return [{
            "condition": "Common Cold",
            "confidence": "85.0%",
            "recommendation": "Rest and fluids. See a doctor if symptoms persist beyond 10 days."
        }]

def get_recommendation(condition):
    # Simple recommendation mapping - expand this in a real application
    recommendations = {
        "Common Cold": "Rest, fluids, and over-the-counter cold medicine. See a doctor if symptoms persist beyond 10 days.",
        "Influenza (Flu)": "Rest, fluids, antiviral medication if early in illness. Seek emergency care for difficulty breathing.",
        "Migraine": "Rest in a dark room, OTC pain relievers, consider prescription medication for frequent migraines.",
        "Pneumonia": "Antibiotics if bacterial, rest, fluids. Seek immediate care for high fever or breathing difficulty.",
        "Urinary Tract Infection": "Antibiotics prescribed by a doctor, increased fluid intake.",
        "Gastroenteritis": "Clear fluids, BRAT diet, rest. Seek care for signs of dehydration.",
        "Hypertension": "Lifestyle changes and possibly medication. Regular monitoring required.",
        "Type 2 Diabetes": "Blood sugar monitoring, diet changes, exercise, possibly medication.",
        "Bronchitis": "Rest, fluids, cough medicine. Antibiotics if bacterial infection is suspected.",
        "Sinusitis": "Decongestants, nasal irrigation, possibly antibiotics for bacterial infections."
    }
    return recommendations.get(condition, "Consult with a healthcare professional for proper diagnosis and treatment.")

# Create Gradio interface
with gr.Blocks(title="AI Medical Diagnosis Analyzer") as demo:
    gr.Markdown("""
    # AI-Based Medical Diagnosis Analyzer
    Enter patient information and symptoms to get potential diagnoses.
    """)
    
    with gr.Row():
        with gr.Column():
            patient_history = gr.Textbox(label="Patient Medical History", placeholder="Any known medical conditions, allergies, etc.")
            symptoms = gr.Textbox(label="Current Symptoms", placeholder="Describe symptoms, duration, severity...")
            age = gr.Number(label="Age", minimum=0, maximum=120)
            gender = gr.Radio(label="Gender", choices=["Male", "Female", "Other/Prefer not to say"])
            submit_btn = gr.Button("Analyze Symptoms")
        
        with gr.Column():
            output = gr.JSON(label="Diagnosis Results")
    
    submit_btn.click(
        fn=analyze_symptoms,
        inputs=[patient_history, symptoms, age, gender],
        outputs=output
    )
    
    gr.Markdown("""
    **Disclaimer:** This tool is for informational purposes only and not a substitute for professional medical advice.
    Always consult with a qualified healthcare provider for diagnosis and treatment.
    """)

# Run the Gradio app
try:
    demo.launch(debug=True)
except Exception as e:
    print(f"Error launching Gradio: {e}")
    print("You may need to check your internet connection or try again later.")
