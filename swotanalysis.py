import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import re
import os
import pandas as pd
import google.generativeai as genai

# Load Gemini API Key from environment variable
gemini_api_key = os.getenv("GEMINI_API_KEY")
if not gemini_api_key:
    st.error("GEMINI_API_KEY not found. Please set it in your environment variables.")
    st.stop()

genai.configure(api_key=gemini_api_key)

# List available models
def get_available_models():
    try:
        models = genai.list_models()
        return [model.name for model in models]
    except Exception as e:
        st.error(f"❌ Error listing models: {e}")
        return []

# Prioritize newer models over older ones
available_models = get_available_models()
preferred_models = [
    "models/gemini-2.0-pro-exp",
    "models/gemini-2.0-flash",
    "models/gemini-1.5-pro",
    "models/gemini-1.5-flash",
]

# Select the best model from the preferred list
model_name = next((m for m in preferred_models if m in available_models), None)

if not model_name:
    st.error("❌ No suitable Gemini models available. Check your API key and account permissions.")
    st.stop()

def extract_swot_data(swot_text):
    """Extracts SWOT elements from AI-generated text with correct classification."""
    swot_dict = {"Strengths": [], "Weaknesses": [], "Opportunities": [], "Threats": []}
    
    categories = list(swot_dict.keys())
    
    for i, category in enumerate(categories):
        # Define regex pattern to capture only the relevant section
        next_category = categories[i + 1] if i + 1 < len(categories) else None
        if next_category:
            pattern = rf"{category}:\s*(.*?)(?=\n{next_category}:|$)"
        else:
            pattern = rf"{category}:\s*(.*)"
        
        match = re.search(pattern, swot_text, re.DOTALL)
        
        if match:
            content = match.group(1).strip()
            items = re.findall(r"-\s*(.+)", content)  # Extract bullet points
            swot_dict[category] = items  # Assign correctly to category
    
    return swot_dict

# Streamlit UI
st.title("AI-Powered SWOT Analysis")
st.write("Enter company information below to generate SWOT analysis.")

company_info = st.text_area("Company Details:")

swot_prompt = """
Analyze the following company's details and generate a structured SWOT analysis:

{company_info}

Provide responses in the format:
Strengths:
- Strength 1
- Strength 2

Weaknesses:
- Weakness 1
- Weakness 2

Opportunities:
- Opportunity 1
- Opportunity 2

Threats:
- Threat 1
- Threat 2
"""

if st.button("Generate SWOT Analysis"):
    model = genai.GenerativeModel(model_name)
    response = model.generate_content(swot_prompt.format(company_info=company_info))
    
    if response:
        swot_analysis = response.text
    else:
        swot_analysis = "Error generating response."
    
    st.text_area("SWOT Analysis Output", swot_analysis, height=300)
    
    # Extract SWOT Data
    swot_dict = extract_swot_data(swot_analysis)

    # Convert to DataFrame for Table Display
    max_length = max(len(swot_dict["Strengths"]), len(swot_dict["Weaknesses"]), 
                     len(swot_dict["Opportunities"]), len(swot_dict["Threats"]))
    
    swot_data = {
        "Strengths": swot_dict["Strengths"] + [""] * (max_length - len(swot_dict["Strengths"])),
        "Weaknesses": swot_dict["Weaknesses"] + [""] * (max_length - len(swot_dict["Weaknesses"])),
        "Opportunities": swot_dict["Opportunities"] + [""] * (max_length - len(swot_dict["Opportunities"])),
        "Threats": swot_dict["Threats"] + [""] * (max_length - len(swot_dict["Threats"])),
    }
    
    swot_df = pd.DataFrame(swot_data)

    st.write("### SWOT Analysis Table")
    st.table(swot_df)

    # Prepare Data for Visualization (Fixed Values)
    labels = list(swot_dict.keys())
    values = [len(swot_dict[key]) for key in labels]

    if sum(values) > 0:
        fig, ax = plt.subplots(figsize=(6, 4))
        colors = ['green', 'red', 'blue', 'orange']
        bars = ax.bar(labels, values, color=colors)
        
        ax.set_ylabel("Number of Insights")
        ax.set_title("SWOT Analysis Summary")
        ax.set_xticks(np.arange(len(labels)))
        ax.set_xticklabels(labels, rotation=30, ha="right")
        
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2, height, f'{height}', ha='center', va='bottom')
        
        st.pyplot(fig)
    else:
        st.warning("No valid SWOT data extracted. Please check the AI response format.")


